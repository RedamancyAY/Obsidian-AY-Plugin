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
import torchaudio
from ay2.torch.nn import LambdaFunctionModule
from ay2.torchaudio.transforms import SpecAugmentBatchTransform

# + editable=true slideshow={"slide_type": ""}
try:
    from .gru_head import GRU_Head
    from .msfm import  MultiScaleFusion2D
except ImportError:
    from gru_head import GRU_Head
    from msfm import MultiScaleFusion2D


# -

# ## Build stage

def build_stage2D(
    n_dim_in, n_dim_out, n_blocks, samples_per_frame, n_head=1, downsample_factor=1
):
    # print(n_dim_in, n_dim_out)
    conv1 = nn.Conv2d(n_dim_in, n_dim_out, 3, stride=1, padding=1, bias=False)
    conv_blocks = [
        MultiScaleFusion2D(
            n_dim=n_dim_out,
            n_head=n_head,
            samples_per_frame=samples_per_frame,
        )
        for i in range(n_blocks)
    ]
    module = nn.Sequential(conv1, *conv_blocks)
    if downsample_factor > 1:
        module.add_module(
            "down-sample",
            # nn.Conv2d(n_dim_out, n_dim_out, 3, stride=2, padding=2)
            nn.Sequential(
                nn.Conv2d(n_dim_out, n_dim_out, 1, stride=2, bias=False),
                nn.BatchNorm2d(n_dim_out),
            ),
        )
    return module


# ## FeatureExtractor

class MultiAudioTransforms(nn.Module):
    def __init__(
        self, specaug_policy='ss'
    ):
        super().__init__()
        self.t1 = torchaudio.transforms.Spectrogram(n_fft=400, hop_length=160)
        self.t2 = torchaudio.transforms.LFCC(
            n_lfcc=201,
            speckwargs={"n_fft": 400, "hop_length": 160, "center": True},
        )
        self.t3 = torchaudio.transforms.MFCC(
            n_mfcc=201,
            melkwargs={
                "n_fft": 400,
                "n_mels": 201,
                "hop_length": 160,
                "mel_scale": "htk",
            },
        )
        self.transform = SpecAugmentBatchTransform.from_policy(specaug_policy)
        

    def norm(self, x):
        # return (x - torch.mean(x)) / (torch.std(x) + 1e-9)

        return (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )

    def __call__(self, x, stage='test'):
        ## spectrogram
        y1 = self.t1(x)
        y1 = torch.log(y1 + 1e-7)
        y1 = self.norm(y1)

        
        ## LFCC
        y2 = self.t2(x)
        # y2 = self.norm(y2)
        ## MFCC
        # y3 = self.t3(x)

        
        if stage == 'train':
            y1 = self.transform(y1)
            y2 = self.transform(y2)
            # print(stage, y1.shape, y2.shape)
        
        res = torch.concat([y1, y2], dim=1)
        # res = self.norm(res)
        return res



# + editable=true slideshow={"slide_type": ""}
class FeatureExtractor2D(nn.Module):
    def __init__(
        self,
        dims=[32, 64, 64, 128],
        n_blocks=[1, 1, 2, 1],
        n_heads=[1, 2, 2, 4],
        samples_per_frame=400,
        use_gru_head=False,
        gru_node=512,
        gru_layers=3,
        args=None,
        **kwargs
    ):
        super().__init__()

        self.dims = dims
        self.samples_per_frame = samples_per_frame
        self.conv_head = nn.Sequential(
            nn.Conv2d(2, dims[0], 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(),
            nn.Conv2d(dims[0], dims[0], 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(dims[0]), ####
        )

        # print(dims)
        self.stages = nn.ModuleList(
            [
                build_stage2D(
                    n_dim_in=dims[max(i - 1, 0)],
                    n_dim_out=dims[i],
                    n_blocks=n_blocks[i],
                    n_head=n_heads[i],
                    samples_per_frame=samples_per_frame // (4 * (2**i)),
                    downsample_factor=2 if i < len(dims) - 1 else 1,
                    # downsample_factor=2,
                )
                for i in range(len(dims))
            ]
        )

        # self.conv_tail = nn.Sequential(
        #     nn.Conv2d(dims[-1], dims[-1], 3, stride=1, padding=1),
        #     # nn.BatchNorm2d(dims[-1]),
        # )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # self.transform = SpecAugmentBatchTransform.from_policy("ld")

        self.debug = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
        self.spectrogram = torchaudio.transforms.LFCC(
            n_lfcc=60 * 2,
            speckwargs={"n_fft": 400, "hop_length": 160, "center": False},
        )
        self.spectrogram = MultiAudioTransforms(specaug_policy=args.specaug)

    def get_content_stream_modules(self, ):
        return [self.conv_head, self.stages[0], self.stages[1], self.stages[2]]
    
    
    def preprocess(self, x, stage='test'):
        x = self.spectrogram(x, stage=stage)
        return x
        
        # x = torch.log(x + 1e-7)
     
        
        # if self.debug:
        #     print('log-scale feature : ', x.shape)

        
        # # if stage=='train':
        # #     x = self.transform(x)

        # x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
        #     torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        # )
        # return x

    def build_final_block(self):
        from copy import deepcopy
        return deepcopy(self.stages[3])

    
    def copy_final_stage(self):
        self.stage3_copy = self.build_final_block()

    def get_final_block_parameters(self):
        return self.stages[3].parameters()
        
    def get_copied_final_block_parameters(self):
        return self.stage3_copy.parameters()

    
    def get_main_stem(self):
        return [self.conv_head, self.stages[0], self.stages[1], self.stages[2]]

    def get_content_stem(self):
        return [self.stages[3]]

    def get_vocoder_stem(self):
        return [self.stage3_copy]
    
    def get_hidden_state(self, x):
        x = self.conv_head(x)
        x = self.stages[0](x)
        x = self.stages[1](x)
        x = self.stages[2](x)
        return x

    def pool_reshape(self, x):
        # print(x.shape)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        return x
    
    def get_final_feature(self, x):
        x = self.stages[3](x)
        x = self.pool_reshape(x)
        return x

    def get_final_feature_copyed(self, x): 
        x = self.stage3_copy(x)
        x = self.pool_reshape(x)
        return x
