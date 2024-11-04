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
# -

def weight_init(m):
    from timm.models.layers import DropPath, to_2tuple, trunc_normal_

    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.Conv3d, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight, gain=math.sqrt(2.0))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d, nn.LayerNorm)):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


# # Feature Model
#
# ## Multi-Scale Fusion Module

# + editable=true slideshow={"slide_type": ""}
class MultiScaleFusion(nn.Module):
    def __init__(self, n_dim, n_head=1, samples_per_frame=400):
        super().__init__()

        self.n_dim = n_dim
        self.samples_per_frame = samples_per_frame
        self.norm = nn.BatchNorm1d(n_dim)

        scales = [1, 5, 25]
        assert samples_per_frame % scales[-1] == 0, samples_per_frame

        self.down_samples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AvgPool1d(scales[i] * 3, stride=scales[i], padding=scales[i])
                    if i > 0
                    else nn.Identity(),
                    nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    # nn.GELU(),
                    nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                )
                for i in range(3)
            ]
        )

        self.up_samples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=scales[i]) if i > 0 else nn.Identity(),
                    nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                    # nn.GELU(),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                )
                for i in range(3)
            ]
        )

        self.conv_fusion = nn.Sequential(
            nn.Conv1d(n_dim * 3, n_dim, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_dim, n_dim * 3, 3, stride=1, padding=1),
        )
        self.mha = Multi_Head_Attention(
            max_k=80, embed_dim=n_dim, num_heads=n_head, dropout=0.1
        )
        self.attn_upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=samples_per_frame // scales[i]),
                    nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(n_dim, n_dim, 3, stride=1, padding=1),
                )
                for i in range(3)
            ]
        )

        self.register_parameter("alpha", nn.Parameter(torch.ones(1, n_dim, 1)))
        self.register_parameter("beta", nn.Parameter(torch.ones(1, n_dim * 3, 1)))

    def forward(self, x):
        short_cut = x
        x = self.norm(x)
        n_frames = x.shape[-1] // self.samples_per_frame
        avg_pool = partial(F.adaptive_avg_pool1d, output_size=n_frames)
        max_pool = partial(F.adaptive_max_pool1d, output_size=n_frames)

        frame_feat = []
        ms_feat = []
        for i in range(3):
            y = self.down_samples[i](x)
            # print("scale %d : "%i, y.shape)
            ms_feat.append(y)
            attn = avg_pool(y) + max_pool(y) # (B, n_dim, n_frames)
            frame_feat.append(attn)  
            # frame_feat.append(attn.transpose(1, 2))  # (B, n_frames, n_dim)

        frame_feat = torch.concat(frame_feat, dim=1) # (B, 3*n_dim, n_frames)
        frame_feat = self.conv_fusion(frame_feat)
        frame_feat = torch.split(frame_feat, self.n_dim, dim=1)
        frame_feat = [x.transpose(1, 2) for x in frame_feat]
        
        v, k, q = frame_feat
        attn = self.mha(q, k, v)
        attn = attn.transpose(1, 2)  # (B, n_dim, n_frames)
        # print("attn shape: ", attn.shape)

        rec_feat = []
        for i in range(3):
            _attn = self.attn_upsamples[i](attn)
            # y = ms_feat[i] + ms_feat[i] * _attn
            y = (
                ms_feat[i]
                + self.beta[:, i * self.n_dim : (i + 1) * self.n_dim, :] * _attn
            )
            y = self.up_samples[i](y)
            rec_feat.append(y)

        rec_feat = rec_feat[0] + rec_feat[1] + rec_feat[2]
        x = x + self.alpha * rec_feat
        return x


# -

model = MultiScaleFusion(n_dim=32)
x = torch.randn(2, 32, 4000)
model(x)


def build_stage(
    n_dim_in, n_dim_out, n_blocks, samples_per_frame, n_head=1, downsample_factor=1
):
    # print(n_dim_in, n_dim_out)
    conv1 = nn.Conv1d(n_dim_in, n_dim_out, 3, stride=1, padding=1)
    conv_blocks = [
        MultiScaleFusion(
            n_dim=n_dim_out,
            n_head=n_head,
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
#     samples_per_frame=400,
#     downsample_factor=2,
# )
# with torch.autograd.profiler.profile(enabled=True) as prof:
#     x = torch.randn(16, 32, 24000)
#     _ = modle(x).shape
# print(prof.key_averages().table(sort_by="self_cpu_time_total"))
# -

# ## Feature Model

class FeatureModel(nn.Module):
    def __init__(
        self,
        dims=[32, 64, 128],
        n_blocks=[2, 6, 2],
        n_heads=[1, 2, 4],
        samples_per_frame=400,
    ):
        super().__init__()

        self.samples_per_frame = samples_per_frame
        self.conv_head = nn.Sequential(
            nn.Conv1d(1, dims[0], 4, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv1d(dims[0], dims[0], 3, stride=1, padding=1),
        )

        self.stages = nn.ModuleList(
            [
                build_stage(
                    n_dim_in=dims[max(i - 1, 0)],
                    n_dim_out=dims[i],
                    n_blocks=n_blocks[i],
                    n_head=n_heads[i],
                    samples_per_frame=400 // (4 * (2**i)),
                    downsample_factor=2 if i < 2 else 1,
                )
                for i in range(3)
            ]
        )

        from asteroid_filterbanks import Encoder, ParamSincFB
        from ay2.torchaudio.preprocess import PreEmphasis

        self.audio_preprocess = nn.Sequential(
            PreEmphasis(), nn.InstanceNorm1d(1, eps=1e-4, affine=True)
        )
        self.time_freq_repr = Encoder(
            ParamSincFB(n_filters=256, kernel_size=251, stride=1), padding=125
        )

        self.apply(weight_init)

    def preprocess_audio(self, x):
        x = self.audio_preprocess(x)
        x = torch.abs(self.time_freq_repr(x))
        x = torch.log(x + 1e-6)
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return x

    def get_feature(self, x):
        audio_length = x.shape[-1]
        audio_frames = audio_length // self.samples_per_frame

        x = self.conv_head(x)
        for i, stage in enumerate(self.stages):
            # print("Input of the %d-th stage"%(i+1), x.shape)
            x = stage(x)  # (B, C, frames)
            # print("Output of the %d-th stage"%(i+1), x.shape)

        # classfication
        feature = torch.mean(x, dim=-1)
        return feature

    def forward(self, x):
        feature = self.get_feature(x)
        return feature


# # Audio Model

class AudioModel(nn.Module):
    def __init__(
        self,
        dims=[32, 64, 128],
        n_blocks=[2, 6, 2],
        n_heads=[1, 2, 4],
        samples_per_frame=400,
    ):
        super().__init__()

        self.feature_model = FeatureModel(
            dims=dims,
            n_blocks=n_blocks,
            n_heads=n_heads,
            samples_per_frame=samples_per_frame,
        )

        self.mlp_v, self.mlp_e = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dims[-1], dims[-1]),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(dims[-1], dims[-1]),
                )
                for _ in range(2)
            ]
        )

        self.dropout = nn.Dropout(p=0.1)
        self.emotion_head = nn.Linear(dims[-1], 13, bias=False)
        self.vocoder_head = nn.Linear(dims[-1], 8, bias=False)
        self.final_head = nn.Linear(dims[-1] * 2, 1, bias=False)

        self.apply(weight_init)

    def forward(self, x, stage="test"):

        
        feature = self.feature_model.get_feature(x)

        vocoder_feature = self.mlp_v(feature)
        emotion_feature = self.mlp_e(feature)

        # feature = self.dropout(feature)
        emotion_logit = self.emotion_head(self.dropout(emotion_feature)).squeeze()
        vocoder_logit = self.vocoder_head(self.dropout(vocoder_feature))

        if stage == "train":
            noise = torch.randn(vocoder_feature.shape).to(vocoder_feature.device)
            vocoder_feature = vocoder_feature + noise * 0.1

        logit = self.final_head(
            self.dropout(
                torch.concat(
                    [emotion_feature, vocoder_feature],
                    dim=-1,
                )
            )
        ).squeeze()

        adv_logit = self.final_head(
            self.dropout(
                torch.concat(
                    [emotion_feature, vocoder_feature[torch.randperm(x.shape[0])]],
                    dim=-1,
                )
            )
        ).squeeze()

        return {
            "feature": feature,
            "emotion_feature": emotion_feature,
            "vocoder_feature": vocoder_feature,
            "logit": logit,
            "vocoder_logit": vocoder_logit,
            "emotion_logit": emotion_logit,
            "adv_logit":adv_logit
        }

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
