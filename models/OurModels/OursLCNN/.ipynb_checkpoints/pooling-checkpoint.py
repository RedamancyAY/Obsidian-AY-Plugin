import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from einops import rearrange

class Classic_Attention(nn.Module):
    def __init__(self,input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
    
    def forward(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2) # (B, embed_dim, 1)
        attention_weights = F.tanh(lin_out.bmm(v_view).squeeze())
        attention_weights_normalized = F.softmax(attention_weights,1)
        return attention_weights_normalized


class AttentiveStatisticsPooling1D(nn.Module):
    """This class implements a attenstive statistic pooling layer.


    Args:
        input_dim: the channel dimesion
        embed_dim: the hidden dim in attention layer
        dim: it should be the temporal dim index, for example, 1 for (B, T, C) input and
            2 for (B, C, T) input.
    """

    def __init__(self, input_dim, embed_dim, dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.dim = dim
        
        self.attention = Classic_Attention(input_dim, embed_dim)


    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance


    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        mean = torch.mean(el_mat_prod,1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling

    def forward(self, x):
        if self.dim == 2:
            x = rearrange(x, 'b c t -> b t c')
        
        attn_weights = self.attention(x)
        x = self.stat_attn_pool(x,attn_weights)
        return x
