# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from ay2.tools.torch_model import freeze_modules
from einops import rearrange
from transformers import AutoFeatureExtractor, WavLMModel

try:
    from .resnet import ResNet
    from .utils.attention import CBAM, ChannelAttention
except ImportError:
    from resnet import ResNet
    from utils.attention import CBAM, ChannelAttention


# %%
def set_conv_stride_to_1(module):
    """
    Set the stride of all convolutional layers in the module to 1.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            # Update stride to 1 for Conv2d layers
            child.stride = (1, 1)
        elif isinstance(child, nn.Module):
            # Recursively apply to child modules
            set_conv_stride_to_1(child)


# %%
def add_noise(x):
    noise_level = 5
    add_noise_level = np.random.randint(1, noise_level) / 100
    mult_noise_level = np.random.randint(1, noise_level) / 100
    z = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level)
    return z


def _noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    if add_noise_level > 0.0:
        add_noise = add_noise_level * np.random.beta(2, 5) * torch.zeros_like(x).normal_()
    if mult_noise_level > 0.0:
        mult_noise = mult_noise_level * np.random.beta(2, 5) * (2 * torch.zeros_like(x).uniform_() - 1) + 1
    return mult_noise * x + add_noise


# %%
class MainStream(nn.Module):
    def __init__(
        self,
        transform_type="LFCC",
        embed_dims=[64, 128, 256, 512],
        verbose=0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.transform_type = transform_type
        # self.transform_type = "Log"
        if transform_type not in ["LFCC", "MFCC", "Log"]:
            raise ValueError(
                "Error!!!!, the transform type in the main stream should be MFCC or LFCC, but your input is ",
                transform_type,
            )
        self.embed_dims = embed_dims
        self.verbose = verbose

        self.resnet = ResNet()
        set_conv_stride_to_1(self.resnet)

        n_mels = [65, 33, 17, 9]
        self.MFCC_transforms = nn.ModuleList(
            [
                torchaudio.transforms.MFCC(
                    sample_rate=16000 // (16 * (4**i)),
                    n_mfcc=n_mels[i],
                    melkwargs={
                        "n_fft": 512 // ((2**i) * 4),
                        "hop_length": 187 // ((2**i) * 4),
                        "n_mels": n_mels[i],
                        "center": True,
                    },
                    log_mels=False,
                )
                for i in range(4)
            ]
        )
        self.LFCC_transforms = nn.ModuleList(
            [
                torchaudio.transforms.LFCC(
                    sample_rate=16000 // (16 * (4**i)),
                    n_lfcc=n_mels[i],
                    speckwargs={
                        "n_fft": 512 // ((2**i) * 4),
                        "hop_length": 187 // ((2**i) * 4),
                        "center": True,
                    },
                )
                for i in range(4)
            ]
        )
        self.Log_transforms = nn.ModuleList(
            [
                torchaudio.transforms.Spectrogram(n_fft=512 // ((2**i) * 4), hop_length=187 // ((2**i) * 4))
                for i in range(4)
            ]
        )

        self.fuse_attn = nn.ModuleList(
            [
                ChannelAttention(embed_dims[max(i - 1, 0)], reduction=4)
                for i in range(4)
                # CBAM(embed_dims[max(i - 1, 0)], reduction_ratio=4)
                # for i in range(4)
            ]
        )
        self.fuse_conv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dims[i] * 2, embed_dims[max(i - 1, 0)], 3, padding=1),
                    nn.BatchNorm2d(embed_dims[max(i - 1, 0)]),
                    nn.ReLU(),
                    nn.Conv2d(embed_dims[max(i - 1, 0)], embed_dims[max(i - 1, 0)], 3, padding=1),
                    nn.BatchNorm2d(embed_dims[max(i - 1, 0)]),
                )
                for i in range(4)
            ]
        )
        self.conv2Ds = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(_c, _c, 1, padding=0),
                    nn.BatchNorm2d(_c),
                    nn.ReLU(),
                    nn.Conv2d(_c, _c, 1, padding=0),
                    nn.BatchNorm2d(_c),
                )
                for _c in embed_dims
            ]
        )

        self.bn2Ds = nn.ModuleList([nn.BatchNorm2d(_c) for _c in embed_dims])

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(_c, _c, 3, stride=2, padding=1),
                    nn.BatchNorm2d(_c),
                    nn.ReLU(),
                    nn.Conv2d(_c, _c, 1),
                    nn.BatchNorm2d(_c),
                    nn.Dropout2d(0.1),
                )
                for i, _c in enumerate(embed_dims[0:-1])
            ]
        )
        self.proj_attn = nn.ModuleList([CBAM(_c, reduction_ratio=4) for i, _c in enumerate(embed_dims)])

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def compute_stage(self, feat1D: torch.Tensor, feat2D: torch.Tensor, stage_idx: int) -> torch.Tensor:
        """_summary_

        Args:
            feat1D (torch.Tensor): _description_
            feat2D (torch.Tensor): _description_
            stage_idx (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        if self.verbose:
            print(
                f"Main Stream => stage {stage_idx+1} input shape",
                feat1D.shape,
                feat2D.shape,
            )

        # if stage_idx > 2:
        #     feat1D = add_noise(feat1D)
        #     feat2D = add_noise(feat2D)

        if self.transform_type == "LFCC":
            feat1D = self.LFCC_transforms[stage_idx](feat1D)
        elif self.transform_type == "MFCC":
            feat1D = self.MFCC_transforms[stage_idx](feat1D)
        elif self.transform_type == "Log":
            feat1D = self.Log_transforms[stage_idx](feat1D)
            feat1D = torch.log(feat1D + 1e-7)
            # print(torch.mean(feat1D), torch.mean(feat2D), stage_idx)

        feat1D = (feat1D - torch.mean(feat1D, dim=(2, 3), keepdim=True)) / (
            torch.std(feat1D, dim=(2, 3), keepdim=True) + 1e-9
        )

        h, w = feat1D.shape[-2:]
        feat1D = torch.nn.functional.interpolate(feat1D, (h, h))
        feat1D = self.conv2Ds[stage_idx](feat1D)

        # feat2D = self.bn2Ds[stage_idx](feat2D)
        feat = torch.concat([feat1D, feat2D], dim=1)
        feat = self.fuse_conv[stage_idx](feat)
        feat = self.fuse_attn[stage_idx](feat) + feat

        # print(torch.mean(feat1D), torch.mean(feat2D), torch.mean(feat), stage_idx)
        # print(torch.std(feat1D), torch.std(feat2D), torch.std(feat), stage_idx)

        if stage_idx == 0:
            out = self.resnet.compute_stage1(feat, first_conv=False, preprocess=False)
            self.previous_out = out
        else:
            feat = self.projections[stage_idx - 1](self.previous_out) + feat
            # feat = self.proj_attn[stage_idx - 1](feat) + feat
            self.previous_out = self.resnet.compute_stage(feat, stage_idx + 1)

        self.previous_out = self.proj_attn[stage_idx](self.previous_out) + self.previous_out

        # print(torch.std(feat1D), torch.std(feat2D), torch.std(feat), torch.std(self.previous_out), stage_idx)

        if self.verbose:
            print(
                f"Main Stream => stage {stage_idx+1} output shape",
                self.previous_out.shape,
            )

        return self.previous_out


# %%
# model = MainStream()
# x = torch.randn(2, 64, 3000)
# y = torch.randn(2, 64, 65, 65)
# z = model.compute_stage(x, y, 0)
# x = torch.randn(2, 128, 750)
# y = torch.randn(2, 128, 33, 33)
# z = model.compute_stage(x, y, 1)
# x = torch.randn(2, 128, 750)
# y = torch.randn(2, 128, 33, 33)
# z = model.compute_stage(x, y, 1)
