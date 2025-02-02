import torch
import torch.nn as nn

from einops import rearrange, repeat



def prob_mask_like(shape, prob, device):
  if prob == 1:
      return torch.ones(shape, device = device, dtype = torch.bool)
  elif prob == 0:
      return torch.zeros(shape, device = device, dtype = torch.bool)
  else:
      return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob


class Denoiser(nn.Module):
  def __init__(self,
              denoise_model,
              cond_model,
              self_condition=False,
              device = 'cuda'):
    super().__init__()

    self.self_condition = self_condition
    
    self.denoise_model = denoise_model
    self.cond_model = cond_model
    
    
    self.perf_emb = nn.Embedding(22, cond_model.dim)
    
    self.null_cond = torch.zeros(1, 54, 256).to(device)

    # self.null_perf = torch.tensor([22], dtype=torch.long).to(device)
    self.null_perf = torch.tensor([21], dtype=torch.long).to(device)


  def forward_with_cond(self, x, t, cond, x_self_cond = None, cfg_scale = 1.5):
    
  
    output, _ = self.forward(x, t, cond, x_self_cond=x_self_cond, cfg_dropout=1) # it means no dropout
    
    if cfg_scale == 1:
      return output
    
    uncond_output, _ = self.forward(x, t, None, x_self_cond=None, cfg_dropout=None)
    
    scaled_output = uncond_output + (output - uncond_output) * cfg_scale
    
    return scaled_output
    
  def forward(self, x, t, cond, x_self_cond = None, cfg_dropout = None):
    
    batch_size = x.shape[0]
    seq_len = x.shape[-1]
    if cond == None:
      if self.null_cond.shape[-1] != seq_len:
        null_cond = torch.zeros(1, 54, seq_len).to(x[0].device)
        null_cond = (null_cond, null_cond, null_cond, null_cond, self.null_perf, null_cond)
      else:
        null_cond = (self.null_cond, self.null_cond, self.null_cond, self.null_cond, self.null_perf, self.null_cond)
  
      null_cond_features = self.cond_model(null_cond)
      
      null_cond_features = null_cond_features.squeeze(0)

      cond = repeat(null_cond_features, 't c -> b t c', b = batch_size)
      
      perf_emb = repeat(self.perf_emb(self.null_perf), '1 c -> b c', b = batch_size)
      

    else:
      _, _, _, _, perf, _  = cond
      perf_emb = self.perf_emb(perf)
      cond_features = self.cond_model(cond)
      
      # print(cond_features.shape)

      
      if cfg_dropout > 0:
        null_cond = (self.null_cond, self.null_cond, self.null_cond, self.null_cond, self.null_perf, self.null_cond)
        
        if self.null_cond.shape[2] != cond_features.shape[1]:
          null_cond = torch.zeros(1, 54, cond_features.shape[1]).to(cond_features.device)
          null_cond = (null_cond, null_cond, null_cond, null_cond, self.null_perf, null_cond)

        null_cond_features = self.cond_model(null_cond)
        # print(null_cond_features.shape)
        null_cond_features = null_cond_features.squeeze(0)
        
        drop_mask = prob_mask_like(batch_size, cfg_dropout, cond_features.device)  #만약 0.8이면 8개는 True, 그대로 사용 / False면 null_emb로 교체
        
        drop_mask_cond = rearrange(drop_mask, 'b -> b 1 1')
        null_emb = repeat(null_cond_features, 't c -> b t c', b = batch_size)
        
        cond = torch.where(drop_mask_cond, cond_features, null_emb)
       
        null_perf_emb = repeat(self.perf_emb(self.null_perf), '1 c -> b c', b = batch_size)
        drop_mask_2 = prob_mask_like(batch_size, cfg_dropout, cond_features.device)  #만약 0.8이면 8개는 True, 그대로 사용 / False면 null_emb로 교체
        drop_mask_perf = rearrange(drop_mask_2, 'b -> b 1')
        perf_emb = torch.where(drop_mask_perf, perf_emb, null_perf_emb)
        
        
    cond = rearrange(cond, 'b t c -> b c t')
    x = self.denoise_model(x, t, cond, perf_emb)
       
    return x, None