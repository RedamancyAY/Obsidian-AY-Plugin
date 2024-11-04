# %%
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torchaudio

try:
    from .base import FeatureExtractor
except ImportError:
    from base import FeatureExtractor

# %%
class Mfm2(nn.Module):
    def forward(self, x):
        out1, out2 = torch.chunk(x, 2, 1)
        return torch.max(out1, out2)


# %%
class LCNN(FeatureExtractor):
    def __init__(self):
        super().__init__()

        # (1, H, W) -> (32, H/4, W/4)
        self.conv_head = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(5, 5),
                padding=(2, 2),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        # (32, H/4, W/4) -> (64, H/8, W/8)
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=128,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(64),
        )

        # (64, H/8, W/8) -> (128, H/16, W/16)
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=256,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        # (128, H/16, W/16) -> (32, 16, 8)
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=(1, 1),
                padding=(0, 0),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(3, 3),
                padding=(1, 1),
                stride=(1, 1),
            ),
            Mfm2(),
            nn.AdaptiveMaxPool2d((16, 8)),
        )

        # (64, 16, 8) -> (64)
        self.block4 = self.build_final_block()


        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)

    
    
    def preprocess(self, x, stage='test'):
        x = self.spectrogram(x)
        x = torch.log(x + 1e-7)
        
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
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(64 * 16 * 8, 256),
            Mfm2(),
            nn.BatchNorm1d(128),
        )


    def get_final_block_parameters(self):
        return self.block4.parameters()
        
    def get_copied_final_block_parameters(self):
        return self.block4_copied.parameters()
    
    
    def copy_final_stage(self):
        self.block4_copied = self.build_final_block()

    
    def get_hidden_state(self, x):
        x = self.conv_head(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x

    def get_final_feature(self, x):
        x = self.block4(x)
        return x

    def get_final_feature_copyed(self, x): 
        x = self.block4_copied(x)
        return x

# %%
# x = torch.randn(2, 1, 224, 224)
# model = LCNN()
# model(x).shape
