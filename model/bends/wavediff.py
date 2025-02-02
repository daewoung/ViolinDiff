import torch.nn as nn 
import torch
import math 
from einops import rearrange

from model.module.time_emb import SinusoidalPosEmb

class WaveDiff(nn.Module):
  def __init__(self, n_mels, res_ch, cond_dim, n_layers, dilation, use_Film):
    super().__init__()
    in_dims = n_mels
    print(res_ch)
    self.input_proj = nn.Sequential(nn.Conv1d(in_dims, res_ch, 1),
                                    nn.ReLU())
    
    sinu_pos_emb = SinusoidalPosEmb(res_ch)
    
    self.time_mlp = nn.Sequential(
      sinu_pos_emb,
      nn.Linear(res_ch, res_ch * 4),
      nn.SiLU(),
      nn.Linear(res_ch * 4, res_ch) 
    )
    
    self.perf_mlp = nn.Sequential(nn.Linear(res_ch, res_ch * 4),
                                  nn.SiLU(),
                                  nn.Linear(res_ch * 4, res_ch))
    
    
    self.res_blocks = nn.ModuleList([
      ResidualBlock(res_ch, cond_dim, 2 ** (i % dilation), use_Film) for i in range(n_layers)
    ])
    
    self.skip_proj = nn.Sequential(
                  nn.Conv1d(res_ch, res_ch, 1),
                  nn.ReLU())
    
    self.out_proj = nn.Conv1d(res_ch, in_dims, 1)
    
  def forward(self, x, t, cond, perf_emb):
    t = self.time_mlp(t)
    perf_emb = self.perf_mlp(perf_emb)

    x = self.input_proj(x)

    skips = []
    
    for block in self.res_blocks:
      x, skip = block(x, t, cond, perf_emb)
      skips.append(skip)
      
    x = torch.sum(torch.stack(skips), dim =0) / math.sqrt(len(self.res_blocks))
    x = self.skip_proj(x)
    x = self.out_proj(x)

    return x
  

class ResidualBlock(nn.Module):
  def __init__(self, res_ch, cond_dim, dilation, use_Film_Perf):
    super().__init__()
    
    self.use_film = use_Film_Perf
    
    if use_Film_Perf:
      self.perf_proj = nn.Sequential(nn.Linear(res_ch, res_ch *2), nn.SiLU())
    #else: 
    self.time_step = nn.Linear(res_ch, res_ch)
    
    self.conditional_proj = nn.Conv1d(cond_dim, res_ch * 2, 1)
    
    self.dilated_conv = nn.Conv1d(res_ch, res_ch * 2, 3, dilation=dilation, padding=dilation)
    self.output_proj = nn.Conv1d(res_ch, res_ch * 2, 1)
    
  def forward(self, x, time, cond, perf_emb):
    time = self.time_step(time)
    time = rearrange(time, 'b d -> b d 1')
    y = x + time 
    
    if self.use_film:
      perf_emb = self.perf_proj(perf_emb)
      perf_emb = rearrange(perf_emb, 'b d -> b d 1')
      scale, shift = perf_emb.chunk(2, dim=1)
      y = y * (scale + 1) + shift

    
    cond = self.conditional_proj(cond)
    y = self.dilated_conv(y) + cond
    
    gate, filter = y.chunk(2, dim=1)
    y = torch.sigmoid(gate) * torch.tanh(filter)
    y = self.output_proj(y)
    residual, skip = y.chunk(2, dim=1)
    return (x + residual) / math.sqrt(2.0), skip