# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: audio_df
#     language: python
#     name: python3
# ---

# +
import math
import random
from argparse import Namespace
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
# -

from transformers import AutoFeatureExtractor, WavLMModel
import pytorch_lightning as pl


# # 1D modules
#

# We use the original WavLM as the backbone to extract general 1D features from 1D waveform
#

class WavLM(nn.Module):
    """A PyTorch module for extracting features using the WavLM model.

    This class wraps the WavLM model from the Hugging Face Transformers library and provides
    functionality to extract either the last hidden state or the intermediate features from the model.

    Example:
        ```python
        # Initialize the WavLM model
        wavlm = WavLM(pretrain_feat="last_hidden_state")

        # Extract features from an input tensor
        input_tensor = torch.randn(1, 16000)  # Example input tensor (batch_size, sequence_length)
        features = wavlm(input_tensor)
        ```

    Args:
        pretrain_path (str): The path to the pretrained WavLM model.
        pretrain_feat (str): The type of feature to extract from the WavLM model.
                                Must be either "last_hidden_state" or "extract_features".
                                Defaults to "last_hidden_state". "extract_features" is the intermediate features from the 1D CNN, while "last_hidden_state" is the final output of the transformer.
        **kwargs: Additional keyword arguments passed to the WavLMModel.from_pretrained method.

    Attributes:
        pretrain_feat (str): The type of feature to extract.
        pretrain_model (WavLMModel): The pretrained WavLM model loaded from the specified path.

    Raises:
        AssertionError: If `pretrain_feat` is not one of ["last_hidden_state", "extract_features"].
    """

    def __init__(
        self, pretrained_path:str, pretrain_feat: str = "last_hidden_state", **kwargs
    ):
        super().__init__()

        assert pretrain_feat in ["last_hidden_state", "extract_features"]
        self.pretrain_feat = pretrain_feat

        self.pretrain_model = WavLMModel.from_pretrained(pretrained_path)

    def extract_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the input tensor using the WavLM model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The extracted features of shape (batch_size, channels, time_steps).
        """

        if x.ndim == 3 and x.size(1) == 1:
            _input = x[:, 0, :]
        elif x.ndim == 2:
            _input = x
        else:
            raise ValueError(
                f"Input tensor for WavLM must be of shape (batch_size, sequence_length) or (batch_size, 1, sequence_length)., but got {x.shape}"
            )

        feature = self.pretrain_model(_input)[self.pretrain_feat]
        feature = torch.transpose(feature, 1, 2)  ## (B, T, C) -> (B, C, T)
        return feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the WavLM model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: The extracted features of shape (batch_size, channels, time_steps).
        """
        return self.extract_feature(x)


class Model_1D(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model1D = WavLM(pretrained_path="/usr/local/ay_data/0-model_weights/microsoft_wavlm-base",
                             pretrain_feat="last_hidden_state")
        self.n_dim = 768
    
    def forward(self, x:torch.Tensor):
        feat = self.model1D(x)
        return feat


# + tags=["active-ipynb"]
# model = Model_1D()
# x = torch.randn(2, 48000)
# model(x).shape
# -

# # 2D Modules
#
# We use the following modules as the backbone to extract general features from 2D spectrogram:
#

from transformers import AutoModel


class AudioMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            "hance-ai/audiomae",
            cache_dir="/usr/local/ay_data/0-model_weights",
            trust_remote_code=True,
        ).encoder.to('cpu')
        self.n_dim = 768

    def extract_feature(self, x: torch.Tensor, profiler=None):

        if x.ndim == 3 and x.size(1) == 1:
            _input = x[:, 0, :]
        elif x.ndim == 2:
            _input = x
        else:
            raise ValueError(
                f"Input tensor for WavLM must be of shape (batch_size, sequence_length) or (batch_size, 1, sequence_length)., but got {x.shape}"
            )

        if profiler is None:
            profiler = pl.profilers.PassThroughProfiler()

        with profiler.profile("AudioMAE: generate melspec from input audio waveform"):
            melspec = [
                self.model.waveform_to_melspec(_input[i][None])
                for i in range(_input.shape[0])
            ]
            # (b, length, n_freq_bins) = (b, 1024, 128)
            melspec = torch.stack(melspec, dim=0)
            # (b, 1, length, n_freq_bins) = (b, 1, 1024, 128)
            melspec = melspec.unsqueeze(1)

        with profiler.profile("AudioMAE: generate spectrogram feature from encoder"):
            # melspec = self.model.waveform_to_melspec(x)  # (length, n_freq_bins) = (1024, 128)
            # melspec = melspec[None,None,:,:]  # (1, 1, length, n_freq_bins) = (1, 1, 1024, 128)
            z = self.model.forward_features(melspec)  # (b, 1+n, d); d=768
            z = z[:, 1:, :]  # (b n d); remove [CLS], the class token

        b, c, w, h = melspec.shape  # w: temporal dim; h:freq dim
        wprime = round(
            w / self.model.patch_embed.patch_size[0]
        )  # width in the latent space
        hprime = round(
            h / self.model.patch_embed.patch_size[1]
        )  # height in the latent space

        # reconstruct the temporal and freq dims
        z = rearrange(z, "b (w h) c -> b c h w", h=hprime)  # (b c h' w')
        return z  


# + tags=["active-ipynb"]
# model = AudioMAE()
# x = torch.randn(2, 48000)
# model.extract_feature(x).shape
# -

class Model_2D(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model2D = AudioMAE()
        self.n_dim = 768
    
    def forward(self, x:torch.Tensor):
        feat = self.model2D.extract_feature(x)
        return feat


# + tags=["active-ipynb"]
# model = Model_2D()
# x = torch.randn(2, 48000)
# model(x).shape
# -

# # Text Modules

# We use ASR modules to extract the text features.
#
# - Whisper from OpenAI: https://huggingface.co/models?search=openai/whisper

class Model_Text(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model1D = WavLM(pretrained_path="/usr/local/ay_data/0-model_weights/models--patrickvonplaten--wavlm-libri-clean-100h-base-plus",
                             pretrain_feat="last_hidden_state")
        self.n_dim = 768
    
    def forward(self, x:torch.Tensor):
        feat = self.model1D(x)
        return feat

# + tags=["active-ipynb"]
# model = Model_Text()
# x = torch.randn(2, 48000)
# model(x).shape
