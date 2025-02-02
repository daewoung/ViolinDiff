from omegaconf import DictConfig, OmegaConf

from .encoder import CondEncoder
from .denoiser import Denoiser
from .wavediff import WaveDiff
from .ddpm import GaussianDiffusion1D as GaussianDiffusion1D_mask

def get_model(cfg : DictConfig, mel_config):

  encoder = CondEncoder(
            cfg.encoder.dim,
            cfg.encoder.fft_out,
            cfg.encoder.num_heads,
            cfg.encoder.num_layers,
            cfg.encoder.dropout,
            cfg.encoder.pitch_num,
            cfg.encoder.pre_norm
  )
  
  wavediff = WaveDiff(
            cfg.encoder.pitch_num,
            cfg.wavenet.res_ch,
            cfg.wavenet.cond_dim,
            cfg.wavenet.n_layers,
            cfg.wavenet.dilation,
            cfg.wavenet.use_Film
  )
  
  denoiser = Denoiser(
            wavediff,
            encoder,
            cfg.bend_diff.self_condition
  )
    
  diff = GaussianDiffusion1D_mask(
    denoiser,
    cfg.bend_diff.beta_schedule,
    cfg.bend_diff.timesteps,
    cfg.bend_diff.objective,
    cfg.bend_diff.self_condition,
    cfg.bend_diff.loss_fn,
    cfg.bend_diff.cfg_dropout,
    cfg.encoder.pitch_num,
    mel_config.mel_len,
    cfg.vocoder,
    cfg.bend_diff.use_enc_mel_train
  )
  
  return diff

