# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import torch
import torch.nn as nn
from einops import rearrange


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim, self.output_dim = input_dim, output_dim

        self.first_conv = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, 3, padding=1),
            nn.BatchNorm1d(output_dim),
        )
        self.conv_block = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, 3, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Conv1d(output_dim, output_dim, 3, padding=1),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x, *args, **kwargs):
        x = self.first_conv(x)
        x = x + self.conv_block(x)
        return x


class AudioCSModule(nn.Module):
    def __init__(self, N, sr):
        super().__init__()
        self.N, self.sr = N, sr
        m = int(N * sr)
        # self.phi = nn.Linear(N, m, bias=False)
        # self.psi = nn.Linear(m, N, bias=False)
        phi = torch.nn.init.orthogonal_(torch.randn(m, N))
        self.phi = nn.Parameter(phi.T)
        self.psi = nn.Parameter(phi)
        self.blocks = nn.ModuleList([
            ConvBlock(1, 64), 
            ConvBlock(64, 64), 
            ConvBlock(64, 128), 
            ConvBlock(128, 128), 
        ])
        self.final_conv = nn.Conv1d(128, 1, 3, padding=1)

    def sample(self, x):
        _x = rearrange(x, "b c (l N) -> b c l N", N=self.N)   
        y = _x  @ self.phi
        return y
    def init_re(self, y, psi=None):
        psi = self.psi if psi is None else psi
        x0 = y @ psi
        x0 = rearrange(x0, "b c l N -> b c (l N)")
        return x0
    
    def block_wise_projection(self, x, y):
        _y = y - self.sample(x)
        _x0 = self.init_re(_y, psi=self.phi.T)
        x = x + _x0
        return x
    
    def forward(self, x):
        y = self.sample(x)
        x0 = self.init_re(y)
        
        x = x0
        for block in self.blocks:
            x = block(x)
            x = self.block_wise_projection(x, y)
        x_re = self.final_conv(x) + x0
        return x_re, x0, y

# + tags=["style-solution", "active-ipynb"]
# model = AudioCSModule(N=1600, sr=0.25)
# x = torch.randn(2, 1, 48000)
# model(x).shape
#
# from ay2.tools import summary_torch_model
# summary_torch_model(model)
