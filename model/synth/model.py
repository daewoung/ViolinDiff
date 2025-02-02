from omegaconf import DictConfig, OmegaConf

from .encoder import CondModel
from .denoiser import Denoiser
from .wavediff import WaveDiff
from .ddpm import GaussianDiffusion1D

def get_model(cfg : DictConfig, mel_config):
  encoder = CondModel(
            cfg.encoder.dim,
            cfg.encoder.fft_out,
            cfg.encoder.num_heads,
            cfg.encoder.num_layers,
            cfg.encoder.dropout,
            cfg.encoder.pitch_num,
            mel_config.n_mels,
            cfg.encoder.pre_norm
  )

  wavediff = WaveDiff(
            mel_config.n_mels,
            cfg.wavenet.res_ch,
            cfg.wavenet.cond_dim,
            cfg.wavenet.n_layers,
            cfg.wavenet.dilation,
            cfg.wavenet.use_Film
  )
  
  denoiser = Denoiser(
            wavediff,
            encoder,
            cfg.diff.self_condition
  )
  
  diff = GaussianDiffusion1D(
    denoiser,
    cfg.diff.beta_schedule,
    cfg.diff.timesteps,
    cfg.diff.objective,
    cfg.diff.self_condition,
    cfg.diff.loss_fn,
    cfg.diff.cfg_dropout,
    mel_config.n_mels,
    mel_config.mel_len,
    cfg.vocoder,
    cfg.diff.use_enc_mel_train
  )
  

  return diff