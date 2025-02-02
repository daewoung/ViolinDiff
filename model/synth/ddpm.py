import torch
import torch.nn as nn 
import numpy as np
import math 
from collections import namedtuple
from torch.cuda.amp import autocast
import torch.nn.functional as F

from functools import partial
from random import random
from tqdm.auto import tqdm
from einops import reduce, repeat

from model.module.wav2mel import get_scaler


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5



def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps, scale=False):
    if scale:
        scale = 1000 / timesteps
        beta_start = scale * 0.0001
        #beta_end = scale * 0.02
        beta_end = scale * 0.06
    else:
        beta_start = 0.0001
        beta_end = 0.06 # 0.02
        
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def get_enc_pretrained(pt_pth):
    enc_pt = torch.load(pt_pth, map_location='cpu')['model']
    return enc_pt


def linear_map_to_minus_one_one(left, right):
    a = 2 / (right - left)
    b = (-right -left) / (right - left)

    c = (right - left) / 2
    d = (right + left) / 2
    return lambda x: a * x + b, lambda x: c * x + d


class GaussianDiffusion1D(nn.Module):
  def __init__(
        self,
        model,
        beta_schedule = 'cosine',
        timesteps = 1000,
        objective = 'pred_noise',
        self_condition = False,
        loss = 'l2',
        cfg_dropout = None,
        n_mels = 80,
        mel_length = 448,
        vocoder_name = 'diffwave',
        use_enc_mel_train = False,
    ):
    super().__init__()
    self.model = model
    self.cfg_dropout = cfg_dropout
    self.use_enc_mel_train = use_enc_mel_train
    
    self.self_condition = self_condition
    
    self.channels = n_mels
    self.seq_length = mel_length

    self.objective = objective
    self.loss = loss
    
    self.vocoder_name = vocoder_name
    
    self.get_scaler = get_scaler()
    
    assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

    if beta_schedule == 'linear':
        betas = linear_beta_schedule(timesteps)
    elif beta_schedule == 'cosine':
        betas = cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f'unknown beta schedule {beta_schedule}')

    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)

    # sampling related parameters

    # self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

    # assert self.sampling_timesteps <= timesteps
    # self.is_ddim_sampling = self.sampling_timesteps < timesteps
    # self.ddim_sampling_eta = ddim_sampling_eta

    # helper function to register buffer from float64 to float32

    register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

    register_buffer('betas', betas)
    register_buffer('alphas_cumprod', alphas_cumprod)
    register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

    # calculations for diffusion q(x_t | x_{t-1}) and others

    register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
    register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
    register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
    register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
    register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

    # calculations for posterior q(x_{t-1} | x_t, x_0)

    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

    register_buffer('posterior_variance', posterior_variance)

    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

    register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
    register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
    register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    # calculate loss weight

    snr = alphas_cumprod / (1 - alphas_cumprod)

    if objective == 'pred_noise':
        loss_weight = torch.ones_like(snr)
    elif objective == 'pred_x0':
        loss_weight = snr
    elif objective == 'pred_v':
        loss_weight = snr / (snr + 1)

    register_buffer('loss_weight', loss_weight)

      # whether to autonormalize

      # self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
      # self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    # self.clip_min = 1e-5
    # self.clip_max = 1e8
    # self.scale_forward, self.scale_backward = linear_map_to_minus_one_one(np.log(self.clip_min), np.log(self.clip_max))

  def predict_start_from_noise(self, x_t, t, noise):
    return (
        extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

  def predict_noise_from_start(self, x_t, t, x0):
    return (
        (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
        extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
    )

  def predict_v(self, x_start, t, noise):
    return (
        extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
        extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
    )

  def predict_start_from_v(self, x_t, t, v):
    return (
        extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
        extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
    )

  def q_posterior(self, x_start, x_t, t):

    posterior_mean = (
        extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
        extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = extract(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

  def model_predictions(self, x, t, cond, 
                      x_self_cond = None, cfg_scale = None, 
                      clip_x_start = False, rederive_pred_noise = False):
    
    if cfg_scale is not None:
        model_output = self.model.forward_with_cond(x, t, cond, x_self_cond, cfg_scale)
    # else:
    #     model_output = self.model(x, t, cond, x_self_cond, cfg_dropout = 1)

    maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

    if self.objective == 'pred_noise':
        pred_noise = model_output
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start = maybe_clip(x_start)

        if clip_x_start and rederive_pred_noise:
            pred_noise = self.predict_noise_from_start(x, t, x_start)

    elif self.objective == 'pred_x0':
        x_start = model_output
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

    elif self.objective == 'pred_v':
        v = model_output
        x_start = self.predict_start_from_v(x, t, v)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)

    return ModelPrediction(pred_noise, x_start)

  def p_mean_variance(self, x, t, cond, x_self_cond = None, cfg_scale=None, clip_denoised = True):
    preds = self.model_predictions(x, t, cond, x_self_cond, cfg_scale = cfg_scale)
    x_start = preds.pred_x_start

    if clip_denoised:
        x_start.clamp_(-1., 1.)

    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
    return model_mean, posterior_variance, posterior_log_variance, x_start

  @torch.no_grad()
  def p_sample(self, x, t: int, cond, x_self_cond = None, cfg_scale = None, clip_denoised = True):
    b, *_, device = *x.shape, x.device
    batched_times = torch.full((b,), t, device = x.device, dtype = torch.long)
    model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, t = batched_times, cond=cond, x_self_cond = x_self_cond, cfg_scale=cfg_scale, clip_denoised = clip_denoised)
    noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
    pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
    return pred_img, x_start

  @torch.no_grad()
  def p_sample_loop(self, cond, num_batches =1, cfg_scale =None):
    device = self.betas.device

    img = torch.randn((num_batches, self.channels, self.seq_length), device=device)

    x_start = None

    for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        self_cond = x_start if self.self_condition else None
        img, x_start = self.p_sample(img, t, cond, self_cond, cfg_scale)
    if self.vocoder_name == 'soundstream':
        img = self.get_scaler.reverse(img)
    elif self.vocoder_name == 'diffwave':
        img = self.get_scaler.reverse_0_1(img)
        
    return img

  @torch.no_grad()
  def p_sample_loop2(self, cond, seq_leng, cfg_scale =None):
    device = self.betas.device

    img = torch.randn((1, self.channels, seq_leng), device=device)
    
    x_start = None
    for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        self_cond = x_start if self.self_condition else None
        img, x_start = self.p_sample(img, t, cond, self_cond, cfg_scale)
    if self.vocoder_name == 'soundstream':
        img = self.get_scaler.reverse(img)
    elif self.vocoder_name == 'diffwave':
        img = self.get_scaler.reverse_0_1(img)

    return img


  @torch.no_grad()
  def ddim_sample(self, 
                total_time_steps,
                sampling_time_steps, cond, num_batches, eta=0., clip_denoised = True):
    device, total_timesteps, sampling_timesteps, eta, objective = self.betas.device, total_time_steps, sampling_time_steps, eta, self.objective

    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
    img = torch.randn((num_batches, self.channels, self.seq_length), device = device)

    x_start = None

    for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
        time_cond = torch.full((num_batches,), time, device=device, dtype=torch.long)
        self_cond = x_start if self.self_condition else None
        pred_noise, x_start, *_ = self.model_predictions(img, time_cond, cond, self_cond, clip_x_start = clip_denoised)
        
        if time_next < 0:
            img = x_start
            continue

        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(img)

        img = x_start * alpha_next.sqrt() + \
            c * pred_noise + \
            sigma * noise

    if self.vocoder_name == 'soundstream':
        img = self.get_scaler.reverse(img)
    elif self.vocoder_name == 'diffwave':
        img = self.get_scaler.reverse_0_1(img)
    
    return img
  
  @torch.no_grad()
  def smoothing(self, x_start, blend_weight, overlap_seq):
    device = self.betas.device
    batch_size = x_start.shape[0]
    
    interpolated = x_start.clone()
    
    for index in range(1, batch_size):
        overlap1 = interpolated[index-1, :, -overlap_seq:]
        overlap2 = interpolated[index, :, :overlap_seq]
        # print(torch.max(overlap1), torch.min(overlap1))
        # blend_weight = torch.linspace(1, 0, overlap_seq, device='cuda').unsqueeze(0).unsqueeze(0)  # .repeat((1, model_output.shape[1], 1))

        blended = overlap1 * blend_weight + overlap2 * (1 - blend_weight)
        interpolated[index-1, :, -overlap_seq:] = blended
        interpolated[index, :, :overlap_seq] = blended
    torch.clamp_(interpolated, -1., 1.)    
    return interpolated

  @torch.no_grad()
  def trasitions(self, out, overlap_seq):
    batch_size, mel_freq, segment_length = out.shape[0], out.shape[1], out.shape[2]
    
    traisition_data = torch.zeros([mel_freq, segment_length * batch_size - (overlap_seq * (batch_size - 1))], device=out.device)


    first_trasition = out[0, :, -overlap_seq:] #right
    traisition_data[:, :segment_length] = out[0, :, :]
    start_pos = segment_length
    for i in range(1, batch_size):
        end_pos = (start_pos + segment_length) - overlap_seq
        traisition_data[:, start_pos:end_pos] = out[i, :, overlap_seq:] 
        start_pos = end_pos

    
    return traisition_data[:, :end_pos]

        
  @torch.no_grad()
  def long_sampling(self, cond, overlap_len, cfg_scale, clip_denoised=True):
    device = self.betas.device
    batch_size = cond[0].shape[0]
    
    
    x = torch.randn((self.channels, self.seq_length), device=device)
    x = repeat(x, 'D T -> B D T', B = batch_size)


    # x = torch.randn((batch_size ,self.channels, self.seq_length), device=device)

    x_start = None

    blend_weight = torch.linspace(1, 0, overlap_len, device=device)


    for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling synth time step', total = self.num_timesteps):
        batched_times = torch.full((batch_size,), t, device = x.device, dtype = torch.long)

        self_cond = x_start if self.self_condition else None
        preds = self.model_predictions(x, batched_times, cond, self_cond, cfg_scale = cfg_scale)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)
            
        x_start = self.smoothing(x_start, blend_weight, overlap_len)
        
        model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = batched_times)
                
        noise = torch.randn((self.channels, self.seq_length), device=device) if t > 0 else 0. # no noise if t == 0
        if t > 0:
            noise = repeat(noise, 'D T -> B D T', B = batch_size)
            
        x = model_mean + (0.5 * model_log_variance).exp() * noise


    img = x

    if self.vocoder_name == 'soundstream':
        img = self.get_scaler.reverse(img) #+ 1e-7
    elif self.vocoder_name == 'diffwave':
        img = self.get_scaler.reverse_0_1(img)
    concat_out = self.trasitions(img, overlap_len)
    return img, concat_out


  @torch.no_grad()
  def sample(self, batch_size = 16):
    seq_length, channels = self.seq_length, self.channels
    sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
    return sample_fn((batch_size, channels, seq_length))

  @torch.no_grad()
  def interpolate(self, x1, x2, t = None, lam = 0.5):
    b, *_, device = *x1.shape, x1.device
    t = default(t, self.num_timesteps - 1)

    assert x1.shape == x2.shape

    t_batched = torch.full((b,), t, device = device)
    xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

    img = (1 - lam) * xt1 + lam * xt2

    x_start = None

    for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
        self_cond = x_start if self.self_condition else None
        img, x_start = self.p_sample(img, i, self_cond)

    return img

  @autocast(enabled=False)
  def q_sample(self, x_start, t, noise=None):
    noise = default(noise, lambda: torch.randn_like(x_start))

    return (
        extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )

  def p_losses(self, x_start, t, cond, valid_mode, noise = None):
    b, c, n = x_start.shape
    noise = default(noise, lambda: torch.randn_like(x_start))

    # noise sample

    x = self.q_sample(x_start = x_start, t = t, noise = noise)

    # if doing self-conditioning, 50% of the time, predict x_start from current set of times
    # and condition with unet with that
    # this technique will slow down training by 25%, but seems to lower FID significantly

    x_self_cond = None
    
    if self.self_condition:
        if random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, cond, cfg_scale=1).pred_x_start
                x_self_cond.detach_()

    # predict and take gradient step
    if valid_mode:
        cfg_drop = 1
    else:
        cfg_drop = self.cfg_dropout
        
    model_out, enc_pred_mel = self.model(x, t, cond, x_self_cond, cfg_dropout=cfg_drop)

    if self.objective == 'pred_noise':
        target = noise
    elif self.objective == 'pred_x0':
        target = x_start
    elif self.objective == 'pred_v':
        v = self.predict_v(x_start, t, noise)
        target = v
    else:
        raise ValueError(f'unknown objective {self.objective}')

    if self.loss == 'l2':
        loss = F.mse_loss(model_out, target, reduction = 'none')
    elif self.loss == 'l1':
        loss = F.l1_loss(model_out, target, reduction = 'none')
        
    loss = reduce(loss, 'b ... -> b', 'mean')

    loss = loss * extract(self.loss_weight, t, loss.shape)
    
    if self.use_enc_mel_train:
        enc_mel_loss = F.mse_loss(enc_pred_mel, x_start)
        return loss.mean(), enc_mel_loss
    else:
        return loss.mean() 

  def forward(self, img, cond, valid_mode=False):
    b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
    assert n == seq_length, f'seq length must be {seq_length}'
    t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

    img = self.get_scaler(img)
    return self.p_losses(img, t, cond, valid_mode)



  @torch.no_grad()
  def sample_loop(self, cond, num_batches =1, cfg_scale =None):
    device = self.betas.device

    img = torch.randn((num_batches, self.channels, self.seq_length), device=device)

    x_start = None

    for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        self_cond = x_start if self.self_condition else None
        img, x_start = self.p_sample(img, t, cond, self_cond, cfg_scale)
        
    return img

  @torch.no_grad()
  def org_sampling(self, x, pred_noise, t):
    return (
        extract(self.sqrt_recip_alphas_cumprod, t, pred_noise.shape) *
        (x - extract(self.betas, t, x.shape) / extract(self.sqrt_one_minus_alphas_cumprod, t, pred_noise.shape))

    )


  @torch.no_grad()
  def outpaint_sample_loop(self, cond, known_part, cfg_scale):
    # must [1]~
    total_mel_len = known_part.shape[-1]
    known_part = known_part[:, :, total_mel_len//2:]
    device = self.betas.device

    x = torch.randn((1, self.channels, self.seq_length), device=device)

    for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
        

        batched_times = torch.full((1,), t, device = x.device, dtype = torch.long)
        batched_times_known_part = torch.full((1,), t - 1, device=x.device, dtype=torch.long)
        
        self_cond = x_start if self.self_condition else None
        preds = self.model_predictions(x, batched_times, cond, self_cond, cfg_scale = cfg_scale)
        x_start = preds.pred_x_start
        
        x_start = x_start.clamp(-1, 1)
        
 
        model_mean, _, model_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = batched_times)
                
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        
        x = model_mean + (0.5 * model_log_variance).exp() * noise
        # T = 200 180쯤 
        # REsample 180 -> 190 -> 180 reverse
        known_part_noised = self.q_sample(known_part, batched_times_known_part) if t > 0 else known_part
        x[:, :, :total_mel_len // 2] = known_part_noised

    return x
    
  @torch.no_grad()
  def single_forward_step(self, x, t):

    times = torch.full((1,), t, device = x.device, dtype = torch.long)
    
    return self.q_sample(x, times)

  
  @torch.no_grad()
  def single_reverse_step(self, x, t, cond, cfg_scale, self_cond):

    batched_times = torch.full((1,), t, device=x.device, dtype=torch.long)
    preds = self.model_predictions(x, batched_times, cond, self_cond, cfg_scale=cfg_scale)
    x_start = preds.pred_x_start
    x_start = x_start.clamp(-1, 1)

    model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=batched_times)
    noise = torch.randn_like(x) if t > 0 else 0.
    return model_mean + (0.5 * model_log_variance).exp() * noise
  
  @torch.no_grad()
  def resample(self, x, cond, cfg_scale, t, j, self_cond):
    # j 스텝 앞으로 이동 후 다시 뒤로 돌아오도록 리샘플링 수행
    max_timesteps = self.num_timesteps-1  # 최대 타임스텝 (200)
    # forward_t가 최대 타임스텝을 넘지 않도록 보장
    for forward_t in range(min(t + j, max_timesteps), t, -1):
        x = self.single_forward_step(x, forward_t)
    # 뒤로 돌아오는 과정에서도 최대 타임스텝을 넘지 않도록 보장
    for backward_t in range(t + 1, min(t + j + 1, max_timesteps + 1)):
        x = self.single_reverse_step(x, backward_t, cond, cfg_scale, self_cond)
    return x


  @torch.no_grad()
  def outpaint_sampling(self, cond, cfg_scale, clip_denoised=True):
    device = self.betas.device
    first_condition = [cond[0][0:1], cond[1][0:1], cond[2][0:1], cond[3][0:1], cond[4][0:1], cond[5][0:1]]
    
    first_x0 = self.sample_loop(first_condition, num_batches=1, cfg_scale=cfg_scale)
    total_mel_len = first_x0.shape[-1] #[B, M, T]
    batch_size = cond[0].shape[0]
    
    total_mel = first_x0
    known_part = first_x0
    for t in tqdm((range(1, batch_size)), desc = 'sampling loop time step', total = batch_size):
        current_condition = [cond[0][t:t+1], cond[1][t:t+1], cond[2][t:t+1], cond[3][t:t+1], cond[4][t:t+1], cond[5][t:t+1]]
        known_part = self.outpaint_sample_loop(current_condition, known_part, cfg_scale)
        total_mel = torch.cat([total_mel, known_part[:, :, total_mel_len//2:]], dim=-1)
    
    img = total_mel

    if self.vocoder_name == 'soundstream':
        img = self.get_scaler.reverse(img) #+ 1e-7
    elif self.vocoder_name == 'diffwave':
        img = self.get_scaler.reverse_0_1(img)

    return img
