# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

# + editable=true slideshow={"slide_type": ""}
from .utils import AdaptiveConv1d, DepthwiseSeparableConv1d, Multi_Head_Attention


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# from utils import AdaptiveConv1d, DepthwiseSeparableConv1d, Multi_Head_Attention

# + editable=true slideshow={"slide_type": ""}
class MultiScaleFusion(nn.Module):
    def __init__(self, n_dim, kernel_size, samples_per_frame=400):
        super().__init__()

        self.samples_per_frame = samples_per_frame

        self.norm = nn.BatchNorm1d(n_dim)
        
        # strides = [1, kernel_size, kernel_size * 2]
        kernel_sizes = [5, 5, 25]
        strides = [1, 5, 25]
        assert samples_per_frame % strides[-1] == 0, samples_per_frame
        self.adap_conv_blocks = nn.ModuleList(
            [
                AdaptiveConv1d(
                    n_dim=n_dim,
                    kernel_size=kernel_sizes[i],
                    stride=strides[i],
                    reduction=strides[i],
                    groups=n_dim,
                    conv_transpose="upsample",
                )
                for i in range(3)
            ]
        )
        self.conv_blocks = nn.ModuleList(
            [
                DepthwiseSeparableConv1d(
                    n_dim, n_dim, kernel_size=3, stride=1, padding="same"
                )
                for i in range(6)
            ]
        )

        self.mha = Multi_Head_Attention(max_k=80, embed_dim=n_dim, num_heads=1)
        self.attn_upsamples = nn.ModuleList(
            [
                nn.Upsample(scale_factor=samples_per_frame // strides[i])
                for i in range(3)
            ]
        )

        self.register_parameter('alpha', nn.Parameter(torch.ones(1, n_dim, 1)) )

    def forward(self, x):
        short_cut = x
        x = self.norm(x)
        n_frames = x.shape[-1] // self.samples_per_frame
        avg_pool = partial(F.adaptive_avg_pool1d, output_size=n_frames)
        max_pool = partial(F.adaptive_max_pool1d, output_size=n_frames)

        frame_feat = []
        ms_feat = []
        for i in range(3):
            y = self.adap_conv_blocks[i](x)
            # y = self.conv_blocks[i](y)
            # print("scale", i, y.shape)
            ms_feat.append(y)
            attn = avg_pool(y) + max_pool(y)
            frame_feat.append(attn.transpose(1, 2))  # (B, n_frames, n_dim)

        v, k, q = frame_feat
        attn = self.mha(q, k, v)
        attn = attn.transpose(1, 2)  # (B, n_dim, n_frames)
        # print("attn shape: ", attn.shape)

        rec_feat = []
        for i in range(3):
            _attn = self.attn_upsamples[i](attn)
            y = ms_feat[i] * _attn
            y = self.adap_conv_blocks[i].reverse(y)
            # y = self.conv_blocks[i + 3](y)
            rec_feat.append(y)

        rec_feat = rec_feat[0] + rec_feat[1] + rec_feat[2]
        x = x + self.alpha * rec_feat
        return x


# -

def build_stage(
    n_dim_in, n_dim_out, n_blocks, kernel_size, samples_per_frame, downsample_factor=1
):
    # print(n_dim_in, n_dim_out)
    conv1 = nn.Conv1d(n_dim_in, n_dim_out, 3, stride=1, padding=1)
    conv_blocks = [
        MultiScaleFusion(
            n_dim=n_dim_out,
            kernel_size=kernel_size,
            samples_per_frame=samples_per_frame,
        )
        for i in range(n_blocks)
    ]
    module = nn.Sequential(conv1, *conv_blocks)
    if downsample_factor > 1:
        module.add_module(
            "down-sample", nn.Conv1d(n_dim_out, n_dim_out, 5, stride=2, padding=2)
        )
    return module


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-activity"]
# modle = build_stage(
#     n_dim_in=32,
#     n_dim_out=128,
#     n_blocks=3,
#     kernel_size=25,
#     samples_per_frame=400,
#     downsample_factor=2,
# )
# with torch.autograd.profiler.profile(enabled=True) as prof:
#     x = torch.randn(16, 32, 24000)
#     _ = modle(x).shape
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# -

class AudioModel(nn.Module):
    def __init__(
        self, dims=[32, 64, 128, 256], n_blocks=[2, 2, 6, 2], samples_per_frame=400
    ):
        super().__init__()

        self.samples_per_frame = samples_per_frame
        self.conv_head = nn.Sequential(
            nn.Conv1d(1, dims[0], 2, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv1d(dims[0], dims[0], 3, stride=1, padding=1)
        )

        self.stages = nn.ModuleList(
            [
                build_stage(
                    n_dim_in=dims[max(i - 1, 0)],
                    n_dim_out=dims[i],
                    n_blocks=n_blocks[i],
                    kernel_size=25,
                    samples_per_frame=400 // (2 * (2**i)),
                    downsample_factor=1 if i == 0 else 2,
                )
                for i in range(4)
            ]
        )

        self.cls_head = nn.Linear(dims[-1], 1, bias=False)

    def forward(self, x):
        audio_length = x.shape[-1]
        audio_frames = audio_length // self.samples_per_frame

        x = self.conv_head(x)
        # print(x.shape)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # print("Output of the %d-th stage"%(i+1), x.shape)

        x = F.adaptive_avg_pool1d(x, 1)
        # x = F.adaptive_avg_pool1d(x, audio_frames)
        x = self.cls_head(x.transpose(1, 2))
        x = torch.mean(x, dim=1)
        return x

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = AudioModel()
# x = torch.randn(32, 1, 48000)
# model(x)
# with torch.autograd.profiler.profile(enabled=True) as prof:
#     x = torch.randn(16, 1, 48000)
#     _ = model(x).shape
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model.to("cuda:1")
#
# import torch
# from torch.autograd import Variable
#
# x = torch.randn(16, 1, 48000)
# y = Variable(x, requires_grad=True).to("cuda:1")

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     z = model(y)
#     print(y.shape)
#     z = torch.sum(z)
#     z.backward()
# # NOTE: some columns were removed for brevityM
# print(prof.key_averages().table(sort_by="self_cuda_time_total"))
