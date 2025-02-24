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
import sys
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from ay2.tools import freeze_modules
from ay2.torch.nn import LambdaFunctionModule
from einops import rearrange

# %%
from torchvision.transforms import v2

# %% editable=true slideshow={"slide_type": ""}
try:
    from .rawnet.rawnet2 import RawNet2
    from .resnet import ResNet
    from .lcnn import LCNN
except ImportError:
    from rawnet.rawnet2 import RawNet2
    from resnet import ResNet
    from lcnn import LCNN


# %%
try:
    from ...Aaasist.Aaasist.load_model import get_model  as load_AASIST
    from ...WaveLM.wavlm import BaseLine as WavLM
except ImportError:
    sys.path.append("../../Aaasist")
    sys.path.append("../../WaveLM")
    from Aaasist.load_model import get_model as load_AASIST
    from wavlm import BaseLine as WavLM


# %% [markdown]
# # 1D models

# %%
class WavLM_1D(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model1D = WavLM()
        self.n_dim = 768

    def forward(self, x):
        if x.ndim == 3:
            x = x[:, 0,:]
        feature = self.model1D.pretrain_model(x)[self.model1D.pretrain_feat] # (B, T, 768)
        return feature.mean(1)


# %% tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = WavLM_1D()
# x = torch.randn(2, 1, 48000)
# model(x).shape

# %%
class RawNet2_1D(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model1D = RawNet2()
        self.n_dim = 512

    def forward(self, x):
        y = self.model1D.model.get_hidden_state(x)
        y = self.model1D.model.get_final_feature(y)
        return y


# %% tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = RawNet2_1D()
# x = torch.randn(2, 1, 48000)
# model(x).shape

# %%
class AASIST_1D(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model1D = load_AASIST("AASIST")
        self.n_dim = 160

    def forward(self, x):
        if x.ndim == 3:
            audio = x[:, 0, :]

        feat, logit = self.model1D(audio)
        return feat


# %% tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = AASIST_1D()
# x = torch.randn(2, 1, 48000)
# model(x).shape

# %%
class ResNet_2D(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model2D = ResNet(pretrained=True)
        self.n_dim = 512

    def forward(self, x):
        x = self.model2D.preprocess(x)
        y = self.model2D.get_hidden_state(x)
        y, conv_feat = self.model2D.get_final_feature(y)
        return y


# %% tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = ResNet_2D()
# x = torch.randn(2, 1, 48000)
# model(x).shape

# %%
class LCNN_2D(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        from torchaudio.transforms import LFCC
        self.lfcc = LFCC(
            n_lfcc=60,
            speckwargs={"n_fft": 400, "hop_length": 160, "center": False},
        )
        
        self.model2D = LCNN()
        self.n_dim = 64

    def forward(self, x):
        x = self.lfcc(x)
        y = self.model2D.extract_feature(x)
        return y


# %% tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = LCNN_2D()
# x = torch.randn(2, 1, 48000)
# model(x).shape

# %% editable=true slideshow={"slide_type": ""}
class MultiViewCombine(nn.Module):
    def __init__(self, verbose=0, cfg=None, args=None, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.args = args

        if cfg.model1D == "rawnet":
            self.feature_model1D = RawNet2_1D()
        elif cfg.model1D == "aasist":
            self.feature_model1D = AASIST_1D()
        elif cfg.model1D.lower() == "wavlm":
            self.feature_model1D = WavLM_1D()
        else:
            raise ValueError('cfg.model1D should be rawnet or aasist , but is ', cfg.model1D)
        
        if cfg.model2D == "resnet":
            self.feature_model2D = ResNet_2D()
        elif cfg.model2D == 'LCNN':
            self.feature_model2D = LCNN_2D()
        else:
            raise ValueError('cfg.model2D should be resnet or LCNN , but is ', cfg.model1D)
        
        in_dim = self.feature_model1D.n_dim + self.feature_model2D.n_dim
        dim = 512
        self.cls_final = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

        self.cls1D = nn.Linear(self.feature_model1D.n_dim, 1, bias=False)
        self.cls2D = nn.Linear(self.feature_model2D.n_dim, 1, bias=False)
        

    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10.0
        code = torch.div(code, code_norm)
        return code



    def forward(self, x, stage="test", batch=None, spec_aug=None, **kwargs):
        batch_size = x.shape[0]
        res = {}

        feat1 = self.feature_model1D(x)
        feat2 = self.feature_model2D(x)

        res["feature1D"] = feat1
        res["feature2D"] = feat2

        res["feature"] = torch.concat([feat1, feat2], dim=-1)

        res["logit1D"] = self.cls1D(res["feature1D"]).squeeze(-1)
        res["logit2D"] = self.cls2D(res["feature2D"]).squeeze(-1)
        res["logit"] = self.cls_final(res["feature"]).squeeze(-1)
        return res

# %% editable=true slideshow={"slide_type": ""} tags=["style-activity", "active-ipynb"]
# from argparse import Namespace
# cfg = Namespace(model1D ='rawnet')
#
# model = MultiViewCombine(cfg=cfg, verbose=1)
# x = torch.randn(3, 1, 48000)
# y = model(x)
# print(y)
