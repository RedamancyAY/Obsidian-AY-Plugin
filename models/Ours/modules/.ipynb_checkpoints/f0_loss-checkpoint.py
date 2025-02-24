import math

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms
import torchyin
from einops import rearrange, repeat


def get_f0_loss(x, pred_f0):
    """
    Assume that the input audio x is with shape (B, 1, 48000). If its length is not equal to 48000,
    you may have to change th frame stride (second).
    """
    pitch = torchyin.estimate(
        x[:, 0, :],
        sample_rate=16000,
        pitch_min=20,
        pitch_max=9000,
        frame_stride=0.01513,  # actually is 0.015625
    ) / 9000

    loss = F.mse_loss(pred_f0, pitch)
    return loss

# +
# x = torch.randn(2, 1, 48000)
# f0 = torch.randn(2, 192)
# # get_f0_loss(x, f0)
