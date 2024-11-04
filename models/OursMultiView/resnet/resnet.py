# %load_ext autoreload
# %autoreload 2

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torchaudio
from copy import deepcopy
import torchaudio
from .model.encoder import ResNetExtractor

MODEL_URL = "https://github.com/descriptinc/lyrebird-wav2clip/releases/download/v0.1.0-alpha/Wav2CLIP.pt"


def get_model(device="cpu", pretrained=True, frame_length=None, hop_length=None):
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            MODEL_URL, map_location=device, progress=True
        )
        model = ResNetExtractor(
            checkpoint=checkpoint,
            scenario="finetune",  # frozen
            transform=True,
            frame_length=frame_length,
            hop_length=hop_length,
        )
    else:
        model = ResNetExtractor(
            scenario="supervise", frame_length=frame_length, hop_length=hop_length
        )
    model.to(device)
    return model


class ResNet(nn.Module):
    def __init__(self, verbose=0):
        super().__init__()

        self.model = get_model(pretrained=True).encoder
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
        # self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=353) # original

        self.verbose = verbose
    
    def preprocess(self, x, stage="test"):
        # x = self.model.spectrogram(x)
        x = self.spectrogram(x)
        x = torch.log(x + 1e-7)
        # x = (x - torch.mean(x)) / (torch.std(x) + 1e-9)

        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )
        return x

    def compute_stage1(self, x, preprocess=True, first_conv=True, spec_aug=None):
        if preprocess:    
            x = self.preprocess(x)

            if spec_aug is not None:
                # print('use spec aug', x.shape)
                x = spec_aug.batch_apply(x)
        
        if first_conv:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
        if self.verbose:
            print('ResNet Stage 1: input after first conv shape', x.shape)
        
        x = self.model.layer1(x)
        if self.verbose:
            print('ResNet Stage 1: output shape', x.shape)
        return x

    def compute_stage2(self, x):
        x = self.model.layer2(x)
        if self.verbose:
            print('ResNet Stage 2: output shape', x.shape)
        return x

    def compute_stage3(self, x):
        x = self.model.layer3(x)
        if self.verbose:
            print('ResNet Stage 3: output shape', x.shape)
        return x

    def compute_stage4(self, x):
        x = self.model.layer4(x)
        if self.verbose:
            print('ResNet Stage 4: output shape', x.shape)
        return x


    def compute_stage(self, x, idx):
        if idx == 2:
            return self.compute_stage2(x)
        if idx == 3:
            return self.compute_stage3(x)
        if idx == 4:
            return self.compute_stage4(x)
        
    
    def compute_latent_feature(self, x):
        x = self.model.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.feature_norm(x)
        if self.verbose:
            print("ResNet Latent Feature: output shape", x.shape)
        return x

    
    def get_hidden_state(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        return x

    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10
        code = torch.div(code, code_norm)
        return code

    def extract_feature(self, x):
        x = self.get_hidden_state(x)
        x, conv_feat = self.get_final_feature(x)
        return x

    
    
    def get_final_feature(self, x):
        conv_feat = self.model.layer4(x)
        x = self.model.avgpool(conv_feat)
        x = x.reshape(x.size(0), -1)

        x = self.feature_norm(x)

        return x, conv_feat
