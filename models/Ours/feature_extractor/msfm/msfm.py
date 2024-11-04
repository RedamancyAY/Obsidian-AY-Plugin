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

# ## Import

# +
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from ay2.torch.nn import LambdaFunctionModule

# + editable=true slideshow={"slide_type": ""}
try:
    from .conv_attention import MLP, Attention
    from .utils import AdaptiveConv1d, DepthwiseSeparableConv1d, Multi_Head_Attention
except ImportError:
    from conv_attention import MLP, Attention
    from utils import AdaptiveConv1d, DepthwiseSeparableConv1d, Multi_Head_Attention


# + editable=true slideshow={"slide_type": ""}
class MultiScaleFusion2D(nn.Module):
    def __init__(self, n_dim, n_head=1, scales=[1, 5, 10], samples_per_frame=400):
        super().__init__()

        self.n_dim = n_dim
        self.norm = nn.BatchNorm2d(n_dim)

        scales = [1, 2, 3]

        self.down_samples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AvgPool2d(scales[i] * 3, stride=scales[i], padding=scales[i])
                    # nn.Conv2d(n_dim, n_dim, 3, stride=scales[i], padding=1, bias=True)
                    if i > 0
                    else nn.Identity(),
                    # nn.BatchNorm2d(n_dim),
                    # nn.ReLU(),
                    nn.Conv2d(
                        n_dim, n_dim, 3, stride=1, padding=1, groups=1, bias=True
                    ),
                    nn.BatchNorm2d(n_dim)
                )
                for i in range(3)
            ]
        )

        # self.conv_attention = Attention(dim=n_dim)
        self.conv_attention = nn.Sequential(
            Attention(dim=n_dim), MLP(dim=n_dim, mlp_ratio=2.0)
        )

        self.up_samples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(
                        scale_factor=scales[i], mode="bilinear", align_corners=True
                    )
                    # nn.ConvTranspose2d(in_channels=n_dim, out_channels=n_dim, kernel_size=3, stride=scales[i], padding=1)
                    if i > 0
                    else nn.Identity(),
                    # nn.BatchNorm2d(n_dim),
                    # nn.ReLU(),
                    nn.Conv2d(
                        n_dim, n_dim, 3, stride=1, padding=1, groups=1, bias=True
                    ),
                    nn.BatchNorm2d(n_dim)
                )
                for i in range(3)
            ]
        )

        # self.final_proj = nn.Sequential(
        #     nn.Conv2d(n_dim*3, n_dim, 1, bias=False),
        #     nn.BatchNorm2d(n_dim),
        #     nn.Dropout(0.1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(n_dim, n_dim, 1, bias=False)
        # )
        # self.final_proj = nn.Conv2d(n_dim*3, n_dim, 1, bias=False)

        self.register_parameter("alpha1", nn.Parameter(torch.ones(1, n_dim, 1, 1)))
        self.register_parameter("alpha2", nn.Parameter(torch.ones(1, n_dim, 1, 1)))
        self.register_parameter("alpha3", nn.Parameter(torch.ones(1, n_dim, 1, 1)))
        self.register_parameter("alpha", nn.Parameter(torch.ones(1, n_dim, 1, 1)))

    def forward(self, x):
        B, C, H, W = x.shape
        short_cut = x
        x = self.norm(x)

        frame_feat = []
        ms_feat = []
        for i in range(3):
            y = self.down_samples[i](x)
            y = self.conv_attention(y)
            # print("scale %d : " % i, y.shape)
            ms_feat.append(y)

        rec_feat = []
        for i in range(3):
            y = self.up_samples[i](ms_feat[i])
            _H, _W = y.shape[-2], y.shape[-1]
            y = F.pad(y, (0, W - _W, 0, H - _H))
            # print(y.shape)
            rec_feat.append(y)

        # rec_feat = (rec_feat[0] + rec_feat[1] + rec_feat[2]) / 3
        rec_feat = (
            self.alpha1 * rec_feat[0]
            + self.alpha2 * rec_feat[1]
            + self.alpha3 * rec_feat[2]
        ) / 3
        # rec_feat = self.final_proj(torch.concat(rec_feat, dim=1))
        x = x + self.alpha * rec_feat
        # x = x + rec_feat
        return x

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# module = MultiScaleFusion2D(n_dim=64)
# x = torch.randn(2, 64, 224, 224)
# module(x).shape

# + editable=true slideshow={"slide_type": ""}
# spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
# x = torch.randn(2, 1, 48000)
# spectrogram(x).shape
