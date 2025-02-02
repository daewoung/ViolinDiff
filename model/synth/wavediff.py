import torch
import torch.nn as nn
import math 
from model.module.time_emb import SinusoidalPosEmb
from einops import rearrange



class Film2(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    
    self.mlp = nn.Sequential(nn.Conv1d(in_dim, out_dim * 2, 3, padding=1), 
                             nn.SiLU()) 
    
    self.conv1 = nn.Conv1d(in_dim, out_dim, 3, padding=1)
    self.conv2 = nn.Conv1d(out_dim, out_dim, 3, padding=1)
    
    self.res_conv = nn.Conv1d(in_dim, out_dim, 1)
    self.act = nn.SiLU()
    self.time_step = nn.Linear(out_dim, in_dim)

    
  def forward(self, x, enc_mel, t):
    time = self.time_step(t)
    time = rearrange(time, 'b d -> b d 1')

    h = x + time
    h = self.conv1(h)
    h = self.act(h)
    
    enc_mel = self.mlp(enc_mel)
    scale, shift = enc_mel.chunk(2, dim=1)
    h = h * (scale + 1) + shift
    
    h = self.conv2(h)
    h = self.act(h)
    
    return h + self.res_conv(x)
  

class ResidualBlock(nn.Module):
  def __init__(self, res_ch, cond_dim, dilation, use_Film_Perf):
    super().__init__()
    
    self.use_film = use_Film_Perf
    
    if use_Film_Perf:
      self.perf_proj = nn.Sequential(nn.Linear(res_ch, res_ch * 2), nn.SiLU())
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
    
    
class WaveDiff(nn.Module):
  def __init__(self, n_mels, res_ch, cond_dim, n_layers, dilation, use_Film):
    super().__init__()
    in_dims = n_mels


    self.input_proj = Film2(in_dims, res_ch)
   
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
    
    noised_x, enc_pred = x    
    x = self.input_proj(noised_x, enc_pred, t)

    perf_emb = self.perf_mlp(perf_emb)


    skips = []
    
    for block in self.res_blocks:
      x, skip = block(x, t, cond, perf_emb)
      skips.append(skip)
      
    x = torch.sum(torch.stack(skips), dim =0) / math.sqrt(len(self.res_blocks))
    x = self.skip_proj(x)
    x = self.out_proj(x)

    return x
  