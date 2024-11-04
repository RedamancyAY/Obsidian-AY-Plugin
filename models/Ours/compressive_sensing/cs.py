# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# %load_ext autoreload
# %autoreload 2

# +
import os

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
# -

from ay2.math import logistic_map


def generate_measurement_matrix(m, n):
    N = logistic_map(0.999131, m * n)
    N = np.array(N)
    N = np.where(N > 0.5, 1, -1)
    matrix = N * (1 / np.sqrt(m))
    matrix = np.reshape(matrix, (m,n)).astype(np.float32)
    return matrix.T


class AudioCS(nn.Module):

    def __init__(self, m, n):
        super().__init__()
        
        self.m = m
        self.n = n
        matrix = generate_measurement_matrix(m, n)
        self.register_buffer('matrix', torch.tensor(matrix))

    def __call__(self, x):

        short_cut = x
        
        flag_ndim3 = False
        if len(x.shape) == 3:
            flag_ndim3 = True
            x = x[:, 0, :]

        x = rearrange(x, 'b (n l) -> b n l', l=self.n)
        # print(self.matrix.shape, x.shape)
        x = torch.matmul(x, self.matrix)
        x = torch.matmul(x, self.matrix.transpose(0, 1))
        x = rearrange(x, 'b n l -> b (n l)')

        if flag_ndim3:
            x = x[:, None, :]

        return x
        # return x - short_cut
        

# +
# module = AudioCS(100, 1600)
# x = torch.randn(2, 1, 48000)
# module(x)
