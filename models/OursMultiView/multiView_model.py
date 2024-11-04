# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import math
import random
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from ay2.torch.nn import LambdaFunctionModule
from einops import rearrange


# %%
from torchvision.transforms import v2

# %% editable=true slideshow={"slide_type": ""}
try:
    from .main_stream import MainStream
    from .rawnet.rawnet2 import RawNet2
    from .resnet import ResNet
    from .resnet1d import ResNet1D
    from .utils.attention import CBAM, Attention1D, ChannelAttention
except ImportError:
    from main_stream import MainStream
    from rawnet.rawnet2 import RawNet2
    from resnet import ResNet
    from resnet1d import ResNet1D
    from utils.attention import CBAM, Attention1D, ChannelAttention


# %%
class FusionModule(nn.Module):
    def __init__(self, verbose=0, cfg=None, args=None, **kwargs):
        super().__init__()
        self.verbose = verbose

        self.init_para = 0.5
        self.kernel = 1
        self.padding = 0

        self.configure_1D_stream_modules()
        self.configure_2D_stream_modules()

    def configure_1D_stream_modules(self):
        self.spectrogram_transforms = nn.ModuleList(
            [
                torchaudio.transforms.Spectrogram(n_fft=512 // ((2**i) * 4), hop_length=187 // ((2**i) * 4))
                for i in range(4)
            ]
        )
        self.alphas = nn.ParameterList(
            [nn.Parameter(torch.ones(1, _c, 1) * self.init_para) for _c in [64, 128, 256, 512]]
        )
        self.convs_for_2D_to_1D = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(_c, _c, self.kernel, padding=self.padding),
                    nn.BatchNorm1d(_c),
                    nn.ReLU(True),
                    nn.Conv1d(_c, _c, self.kernel, padding=self.padding),
                    nn.BatchNorm1d(_c),
                )
                for _c in [64, 128, 256, 512]
            ]
        )
        self.attn1D = nn.ModuleList([Attention1D(_c) for _c in [64, 128, 256, 512]])

    def configure_2D_stream_modules(self):
        self.convs_for_1D_to_2D = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(_c, _c, self.kernel, padding=self.padding),
                    nn.BatchNorm2d(_c),
                    nn.ReLU(True),
                    nn.Conv2d(_c, _c, self.kernel, padding=self.padding),
                    nn.BatchNorm2d(_c),
                )
                for _c in [64, 128, 256, 512]
            ]
        )
        self.BN2Ds = nn.ModuleList([nn.BatchNorm2d(_c) for _c in [64, 128, 256, 512]])

        self.betas = nn.ParameterList(
            [nn.Parameter(torch.ones(1, _c, 1, 1) * self.init_para) for _c in [64, 128, 256, 512]]
        )
        self.attn2D = nn.ModuleList([CBAM(_c, reduction_ratio=4) for _c in [64, 128, 256, 512]])

    def forward(self, x):
        return x

    def fuse_2Dfeat_for_1D(self, feat1D, feat2D, idx):
        h, w = feat2D.shape[-2:]
        L = feat1D.shape[-1]
        scale_factor = L / (h * w)

        if self.verbose:
            print("Fuse 2D featuer for 1D:", feat1D.shape, feat2D.shape, scale_factor)

        feat2D = rearrange(feat2D, "b c h w -> b c (w h)")
        feat2D = F.upsample(feat2D, scale_factor=scale_factor + 0.0001, mode="linear")
        feat2D = self.convs_for_2D_to_1D[idx](feat2D)

        feat1D = self.alphas[idx] * feat1D + (1 - self.alphas[idx]) * feat2D

        # feat1D = self.attn1D[idx](feat1D) + feat1D

        return feat1D, feat2D

    def transform_audio_into_spectorgram(self, x, transform, idx):
        x = transform(x)
        x = torch.log(x + 1e-7)
        # x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9)
        x = (x - torch.mean(x, dim=(2, 3), keepdim=True)) / (torch.std(x, dim=(2, 3), keepdim=True) + 1e-9)
        # x = self.BN2Ds[idx](x)
        return x

    def fuse_1Dfeat_for_2D(self, feat1D, feat2D, idx):
        # print(feat1D.shape, feat2D.shape, idx)
        feat1D = torch.concat(
            [
                self.transform_audio_into_spectorgram(x, self.spectrogram_transforms[idx], idx)
                for x in torch.split(feat1D, 64, dim=0)
            ],
            dim=0,
        )
        H, W = feat2D.shape[-2:]
        feat1D = feat1D[:, :, :H, :W]

        if self.verbose:
            print("Fuse 1D featuer for 2D:", feat1D.shape, feat2D.shape)

        feat1D = self.convs_for_1D_to_2D[idx](feat1D)
        feat2D = self.betas[idx] * feat2D + (1 - self.betas[idx]) * feat1D

        # feat2D = self.attn2D[idx](feat2D) + feat2D

        return feat1D, feat2D


# %%
# spectrogram_transforms = nn.ModuleList(
#     [
#         torchaudio.transforms.Spectrogram(
#             n_fft=512 // (2 ** (i) * 4), hop_length=187 // (2 ** (i) * 4)
#         ).cuda()
#         for i in range(4)
#     ]
# )
# x = torch.randn(127, 512,47).cuda()
# spectrogram_transforms[-1](x).shape

# %% editable=true slideshow={"slide_type": ""}
class MultiViewModel(nn.Module):
    def __init__(self, verbose=0, cfg=None, args=None, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.args = args

        self.feature_model2D = ResNet()
        self.main_stream = MainStream(transform_type=cfg.transform_type)
        self.feature_model1D = ResNet1D(
            in_channels=1,
            base_filters=64,
            kernel_size=3,
            downsample_stride=4,
            groups=1,
            n_block=8,
            n_classes=0,
            downsample_gap=2,
            increasefilter_gap=2,
            verbose=1,
        )
        feat_dim = 512
        final_dim = 512
        self.fusion_module = FusionModule(verbose=verbose)

        ## build classifiers
        self.dropout = nn.Dropout(0.1)
        self.cls_final = nn.utils.weight_norm(nn.Linear(final_dim, 1, bias=False))
        self.cls1D, self.cls2D = [nn.utils.weight_norm(nn.Linear(feat_dim, 1, bias=False)) for _ in range(2)]
        # self.cls_final = nn.Linear(512, 1, bias=False)
        # self.cls1D, self.cls2D = [nn.Linear(final_dim, 1, bias=False) for _ in range(2)]
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.set_verbose(verbose)

    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10.0
        code = torch.div(code, code_norm)
        return code

    def print_shape(self, *args):
        for x in args:
            print(x.shape)

    def set_verbose(self, verbose):
        self.feature_model1D.verbose = verbose
        self.feature_model2D.verbose = verbose
        self.fusion_module.verbose = verbose
        # self.main_stream.verbose = verbose
        self.verbose = verbose

    def forward(self, x, stage="test", batch=None, spec_aug=None, **kwargs):
        batch_size = x.shape[0]
        res = {}
        # _input = x.clone()

        feat1_1 = self.feature_model1D.compute_stage1(x)
        feat2_1 = self.feature_model2D.compute_stage1(x, spec_aug=spec_aug)

        fused_1D, feat1D_by_2D = self.fusion_module.fuse_2Dfeat_for_1D(feat1_1, feat2_1, 0)
        feat2D_by_1D, fused_2D = self.fusion_module.fuse_1Dfeat_for_2D(feat1_1, feat2_1, 0)
        feat = self.main_stream.compute_stage(feat1_1, feat2_1, 0)
        feat1_2 = self.feature_model1D.compute_stage2(fused_1D)
        feat2_2 = self.feature_model2D.compute_stage2(fused_2D)

        fused_1D, feat1D_by_2D = self.fusion_module.fuse_2Dfeat_for_1D(feat1_2, feat2_2, 1)
        feat2D_by_1D, fused_2D = self.fusion_module.fuse_1Dfeat_for_2D(feat1_2, feat2_2, 1)
        feat = self.main_stream.compute_stage(feat1_2, feat2_2, 1)
        feat1_3 = self.feature_model1D.compute_stage3(fused_1D)  # (B, 256, 188)
        feat2_3 = self.feature_model2D.compute_stage3(fused_2D)  # (B, 256, 19, 19)

        fused_1D, feat1D_by_2D = self.fusion_module.fuse_2Dfeat_for_1D(feat1_3, feat2_3, 2)
        feat2D_by_1D, fused_2D = self.fusion_module.fuse_1Dfeat_for_2D(feat1_3, feat2_3, 2)
        feat = self.main_stream.compute_stage(feat1_3, feat2_3, 2)
        feat1_4 = self.feature_model1D.compute_stage4(fused_1D)
        feat2_4 = self.feature_model2D.compute_stage4(fused_2D)

        fused_1D, feat1D_by_2D = self.fusion_module.fuse_2Dfeat_for_1D(feat1_4, feat2_4, 3)
        feat2D_by_1D, fused_2D = self.fusion_module.fuse_1Dfeat_for_2D(feat1_4, feat2_4, 3)
        feat = self.main_stream.compute_stage(feat1_4, feat2_4, 3)
        feat1 = self.feature_model1D.compute_latent_feature(fused_1D)
        feat2 = self.feature_model2D.compute_latent_feature(fused_2D)

        # res["feature"] = torch.concat([feat1, feat2], dim=-1)
        res["feature1D"] = feat1
        res["feature2D"] = feat2
        res["feature"] = self.main_stream.resnet.compute_latent_feature(feat)

        # if stage == "train":
        #     shuffle_id = torch.randperm(batch_size)
        #     feat1 =  exchange_mu_std(feat1, feat1[shuffle_id], dim=-1)
        #     feat2 =  exchange_mu_std(feat2, feat2[shuffle_id], dim=-1)
        #     res["feature"] = exchange_mu_std(res["feature"], res["feature"][shuffle_id], dim=-1)
            # feat1 = random_noise(feat1, noise_level=10)
            # feat2 = random_noise(feat2, noise_level=10)
            # res["feature"] = random_noise(res["feature"], noise_level=10)

        # res["feature"] = torch.concat([feat1, feat2], dim=-1)

        # print('mean', feat1.mean(), feat2.mean(), res["feature"].mean())
        # print('std', feat1.std(), feat2.std(), res["feature"].std())

        res["logit1D"] = self.cls1D(self.dropout(feat1)).squeeze(-1)
        res["logit2D"] = self.cls2D(self.dropout(feat2)).squeeze(-1)
        # weight1 = self.cls_final[0].weight[:, :512].to(feat1.device)
        # res["logit1D"] = torch.nn.functional.linear(self.dropout(feat1), weight1).squeeze(-1)
        # weight2 = self.cls_final[0].weight[:, 512:].to(feat2.device)
        # res["logit2D"] = torch.nn.functional.linear(self.dropout(feat2), weight2).squeeze(-1)

        res["logit"] = self.cls_final(self.dropout(res["feature"])).squeeze(-1)
        return res


# %% editable=true slideshow={"slide_type": ""} tags=["style-activity", "active-ipynb"]
# model = MultiViewModel(verbose=1)
# x = torch.randn(3, 1, 48000)
# _ = model(x)

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# # model = model.cuda()
# # x = x.cuda()
# # model(x)

# %%
def exchange_mu_std(x, y, dim=None):
    mu_x = torch.mean(x, dim=dim, keepdims=True)
    mu_y = torch.mean(y, dim=dim, keepdims=True)
    std_x = torch.std(x, dim=dim, keepdims=True)
    std_y = torch.std(y, dim=dim, keepdims=True)

    alpha = np.random.randint(50, 100) / 100
    target_mu = alpha * mu_x + (1 - alpha) * mu_y
    target_std = alpha * std_x + (1 - alpha) * std_y
    z = target_std * ((x - mu_x) / std_x) + target_mu

    # z = random_noise(z, noise_level=10)

    return z


def random_noise(x, noise_level=10):
    add_noise_level = np.random.randint(0, noise_level) / 100
    mult_noise_level = np.random.randint(0, noise_level) / 100
    z = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level)
    return z


def _noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    if add_noise_level > 0.0:
        add_noise = add_noise_level * np.random.beta(2, 5) * torch.FloatTensor(x.shape).normal_().to(x.device)
    if mult_noise_level > 0.0:
        mult_noise = (
            mult_noise_level * np.random.beta(2, 5) * (2 * torch.FloatTensor(x.shape).uniform_() - 1).to(x.device) + 1
        )
    return mult_noise * x + add_noise
