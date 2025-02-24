# %%
import torch
import torch.nn as nn
import torchaudio
from einops.layers.torch import Rearrange

try:
    from .base import FeatureExtractor
except ImportError:
    from base import FeatureExtractor

# %%
class Mfm2(nn.Module):
    def forward(self, x):
        out1, out2 = torch.chunk(x, 2, 1)
        return torch.max(out1, out2)
        # return out1 * out2


# %%
class MultiAudioTransforms(nn.Module):
    def __init__(
        self,
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

    def norm(self, x):
        # return (x - torch.mean(x)) / (torch.std(x) + 1e-9)

        return (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )

    def __call__(self, x):
        ## spectrogram
        y1 = self.t1(x)
        y1 = torch.log(y1 + 1e-7)
        y1 = self.norm(y1)

        ## LFCC
        y2 = self.t2(x)
        # y2 = self.norm(y2)
        ## MFCC
        # y3 = self.t3(x)

        res = torch.concat([y1, y2], dim=1)
        # res = self.norm(res)
        return res



# %%
# x = torch.randn(2, 1, 48000)
# m = MultiAudioTransforms()
# m(x)

# %%
class LCNN(FeatureExtractor):
    def __init__(self):
        super().__init__()

        # self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
        self.spectrogram = torchaudio.transforms.LFCC(
            n_lfcc=60 * 2,
            speckwargs={"n_fft": 400, "hop_length": 160, "center": False},
        )
        in_channels = 1

        self.spectrogram = MultiAudioTransforms()
        in_channels = 2

        # (1, H, W) -> (32, H/4, W/4)
        self.conv_head = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=(5, 5),
                padding=(2, 2),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        # (32, H/4, W/4) -> (48, H/8, W/8)
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=96,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(48),
        )

        # (48, H/8, W/8) -> (64, H/16, W/16)
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=48,
                out_channels=96,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.BatchNorm2d(48),
            nn.Conv2d(
                in_channels=48,
                out_channels=128,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        # (64, H/16, W/16) -> (32, 16, 8)
        self.final_hw = (7, 7)
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.BatchNorm2d(32),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),  ####
            ),
            Mfm2(),
            nn.AdaptiveMaxPool2d(self.final_hw),
        )

        # (32, 16, 8) -> (64)
        self.block4 = self.build_final_block()

    def preprocess(self, x, stage="test"):
        x = self.spectrogram(x)
        # x = (x - torch.mean(x)) / (torch.std(x) + 1e-9)

        return x

        # x = torch.log(x + 1e-7)

        # if self.debug:
        # print('log-scale feature : ', x.shape)

        # if stage=='train':
        #     x = self.transform(x)

        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )
        return x

    def build_final_block(self):
        return nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(32 * self.final_hw[0] * self.final_hw[1], 128),
            Mfm2(),
            nn.BatchNorm1d(64),
        )

    def get_final_block_parameters(self):
        return self.block4.parameters()

    def get_copied_final_block_parameters(self):
        return self.block4_copied.parameters()

    def copy_final_stage(self):
        self.block4_copied = self.build_final_block()

    def debug_print(self, *args):
        if hasattr(self, "debug") and self.debug:
            print(*args)

    def get_hidden_state(self, x):
        self.debug_print("Input: ", x.shape)
        x = self.conv_head(x)
        self.debug_print("Conv head:", x.shape)
        x = self.block1(x)
        self.debug_print("Block 1:", x.shape)
        x = self.block2(x)
        self.debug_print("Block 2:", x.shape)
        x = self.block3(x)
        self.debug_print("Block 3:", x.shape)
        return x

    def get_final_feature(self, x):
        x = self.block4(x)
        return x

    def get_final_feature_copyed(self, x):
        x = self.block4_copied(x)
        return x


# %%
x = torch.randn(2, 1, 48000)
model = LCNN()
model.debug = 1
x = model.preprocess(x)
model(x).shape
