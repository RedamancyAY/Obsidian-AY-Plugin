# %load_ext autoreload
# %autoreload 2

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torchaudio
from copy import deepcopy

from .v2 import RawNet2 as org_model

RAW_NET2_CONFIG = {
    "nb_samp": 48000,
    "first_conv": 1024,  # no. of filter coefficients
    "in_channels": 1,  # no. of filters channel in residual blocks
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 1,
}


def RawNet2():
    return org_model(deepcopy(RAW_NET2_CONFIG))

# +
# model = RawNet2(deepcopy(RAW_NET2_CONFIG))

# x = torch.randn(3, 48000)
# model(x)
