import torch
import torch.nn as nn 

from einops import rearrange

from model.module.time_emb import SinusoidalPosEmb
from model.module.fftblock import FFTBlock
  

class CondEncoder(nn.Module):
  def __init__(self, dim, fft_out, num_heads, num_layers, dropout, pitch_num, pre_norm):
    super().__init__()
    self.dim = dim

    self.pitch_gru = nn.GRU(pitch_num, dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
    self.onset_gru = nn.GRU(pitch_num, dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
    self.offset_gru = nn.GRU(pitch_num, dim // 2, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        
    self.pos_emb = SinusoidalPosEmb(dim)
    
    self.encoders = nn.ModuleList([FFTBlock(dim, fft_out, num_heads, dropout, pre_norm) for _ in range(num_layers)])
    
  def forward(self, cond):
    pitch, onset, _, _, perf, offset = cond
    pitch_length = pitch.size(-1)
    
    t = torch.arange(0, pitch_length, device=pitch.device)

    pitch = rearrange(pitch, 'b d t -> b t d')
    onset = rearrange(onset, 'b d t -> b t d')
    offset = rearrange(offset, 'b d t -> b t d')
        
    pitch, _ = self.pitch_gru(pitch)
    onset, _ = self.onset_gru(onset)
    offset, _ = self.offset_gru(offset)
    
    cond = pitch + onset + offset

    cond = cond + self.pos_emb(t)
    
    for encoder in self.encoders:
      cond = encoder(cond)
    return cond