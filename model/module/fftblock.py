import torch.nn as nn 

class FFTBlock(nn.Module):
  def __init__(self, dim_in, dim_out, num_heads, dropout, pre_norm):
    super().__init__()
    
    self.mha = nn.MultiheadAttention(dim_in, num_heads, batch_first=True)
    self.ff = nn.Sequential(
        nn.Linear(dim_in, dim_out),
        nn.GELU(),
        nn.Linear(dim_out, dim_in),
    )
    self.LayerNorm1 = nn.LayerNorm(dim_in)
    self.LayerNorm2 = nn.LayerNorm(dim_in)
    
    self.pre_norm = pre_norm
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    
  def forward(self, cond):
    if self.pre_norm:
      tmp = self.LayerNorm1(cond)
      attn_output, _ = self.mha(tmp, tmp, tmp)
      out = self.dropout1(attn_output) + cond
      tmp = self.LayerNorm2(out)
      out = self.dropout2(self.ff(tmp)) + out
    else:
      attn_output, _ = self.mha(cond, cond, cond)
      attn_output = self.dropout1(attn_output)
      out = self.LayerNorm1(attn_output + cond)
      out = self.dropout2(self.ff(out)) + out
      out = self.LayerNorm2(out)
    return out