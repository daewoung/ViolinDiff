import math

import torch
import torch.nn as nn
import torchaudio
import pickle
from pathlib import Path


class MelConverter(nn.Module):
  def __init__(self, sr, win_length, hop_length, n_fft, f_min, f_max, n_mels, clip_value_min, clip_value_max):
    super().__init__()
    FILE_DIR = Path(__file__).resolve().parent
    pkl_path = FILE_DIR / "MEL_BASIS.pkl"

    with open(pkl_path, 'rb') as f:
      MEL_BASIS = pickle.load(f)
    
    self.mel_basis = torch.tensor(MEL_BASIS)
    self.spec = torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length, 
                                                  power=None, normalized=False, center=True, pad_mode='constant')
    self.clip_value_min = clip_value_min
    self.clip_value_max = clip_value_max
    
  def forward(self, audio):
    spectrogram = self.spec(audio)
    spectrogram = spectrogram.transpose(-1, -2)
    spectrogram = torch.abs(spectrogram)
    
    mel = torch.matmul(spectrogram, self.mel_basis.to(spectrogram.device))
    if len(spectrogram.shape) == 3:
      mel = mel[:, 1:, :]
    elif len(spectrogram.shape) == 2:
      mel = mel[1:, :]
    mel = mel.clamp(self.clip_value_min, self.clip_value_max)
    mel = torch.log(mel)
    return mel.transpose(-1, -2)

class Scaler(nn.Module):
  def __init__(self, init_min: float = math.inf, init_max: float = -math.inf):
    super().__init__()
    self.register_buffer("min", torch.tensor(init_min))
    self.register_buffer("max", torch.tensor(init_max))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.min) / (self.max - self.min) * 2 - 1  # 0-1 to -1 to 1

  def reverse(self, x: torch.Tensor, max = None) -> torch.Tensor:
    if max is not None:
      return (x + 1) / 2 * (max - self.min) + self.min
    else:
      return (x + 1) / 2 * (self.max - self.min) + self.min
    
  def reverse_0_1(self, x: torch.Tensor) -> torch.Tensor:
    spectrogram = self.reverse(x) # input muset be -1 to 1
    #spectrogram = (x - self.min) / (self.max - self.min)
    spectrogram = torch.clamp((spectrogram + 100) / 100, 0.0, 1.0)
    
    return spectrogram
  
def adaptive_update_hook(module: Scaler, input):
    x = input[0]
    if module.training:
        module.min.fill_(torch.min(module.min, x.min()))
        module.max.fill_(torch.max(module.max, x.max()))


def get_scaler(adaptive: bool = True, **kwargs) -> Scaler:
    scaler = Scaler(**kwargs)
    if adaptive:
        scaler.register_forward_pre_hook(adaptive_update_hook)
    return scaler