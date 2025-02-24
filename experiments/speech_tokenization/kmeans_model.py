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
#     display_name: torch
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# # Import necessary libraries

import numpy as np  # for numerical computations
import matplotlib.pyplot as plt  # for plotting (optional)

# +
import os
from argparse import Namespace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# -

import time
import joblib
from sklearn.cluster import MiniBatchKMeans

import torchaudio
from torch.utils.data import ConcatDataset, DataLoader, Subset

# # KMeansTokenizer

# 使用WavLM模型和LibriSpeech 960h训练数据训练Kmeans模型后，我们可以用它将音频表示聚类成不同的units。

from einops import rearrange
from sklearn.cluster import MiniBatchKMeans


# 为了更好地融合进Speech tokenizer，我们将Kmeans进行封装，使其既能处理单个输入，也能处理**批量输入**。

class KMeansTokenizer:
    """
    Tokenizer based on kmeans clustering. This class can deal with batch input!!
    
    Args:
        vocab_size (int, optional): the size of the vocabulary (number of clusters). Defaults to 200.
    """

    def __init__(self, vocab_size:int=200):
        
        self.model = self.load_kmeans_model(n_cluster=vocab_size)
        
    def load_kmeans_model(self, n_cluster=200):
        """load kmeans model from file

        Args:
            n_cluster (int, optional): the size of the vocabulary (number of clusters). Defaults to 200.

        Returns:
            kmeans_model: the MinibatchKMeans model
        """
        dataset_root_path = "/home/ay/data2/datasets/Lib"
        kmeans_path = os.path.join(dataset_root_path, f"kmeans_model-{n_cluster}.pkl")
        kmeans_model = joblib.load(kmeans_path)
        return kmeans_model

    def extract_unique_units(self, arr):
        """extract unique units from the input array

        Args:
            arr (np.ndarray): the input array

        Returns:
            np.ndarray: the unique units
        """
        return torch.unique_consecutive(torch.from_numpy(arr)).numpy()


    def predict_batch_units(self, batch_input: np.ndarray):
        """predict units using kmeans model for batch input

        Args:
            batch_input (np.ndarray): the batch input with shape of (B, L, C)

        Raises:
            ValueError: the input shape is not supported

        Returns:
            np.ndarray: the predicted units with shape of (B, L)
        """
        if isinstance(batch_input, torch.Tensor):
            batch_input = batch_input.detach().cpu().numpy()
        
        assert batch_input.ndim == 3, "input shape should be (B, L, C)"
        
        batch_size = batch_input.shape[0]
        _input = rearrange(batch_input, 'b l c -> (b l) c')
        units = self.model.predict(_input)
        batch_units = rearrange(units, '(b l) -> b l', b=batch_size)
        # print(batch_units.shape)
        unique_units = [self.extract_unique_units(u) for u in batch_units]
        return unique_units, batch_units
    
    def __call__(self, x):
        if x.ndim == 3:
            return self.predict_batch_units(x)
        else:
            return self.model.predict(x)


# # SpeechEncoder

# SpeechEncode contains two modules:
# 1. Audio feature extraction model, using WavLM here
# 2. Tokenizer, using Kmeans here

class CustomSpeechEncoder(nn.Module):
    def __init__(self, dense_model_name="wavlm", quantizer_name="kmeans", vocab_size=200):
        super().__init__()
        self.load_SpeechEncoder(dense_model_name, quantizer_name=quantizer_name, vocab_size=vocab_size)

    def load_SpeechEncoder(self, dense_model_name, quantizer_name="kmeans", vocab_size=500):
        
        if dense_model_name == "wavlm":
            from transformers import WavLMModel
            model = WavLMModel.from_pretrained("microsoft/wavlm-base")
            model.lm_head = nn.Identity()
        elif dense_model_name == "wav2vec2":
            raise NotImplementedError
        self.model = model
        
        if quantizer_name == "kmeans":
            self.quantizer_model = KMeansTokenizer(vocab_size=vocab_size)


    def forward(self, x):
        return self.encode_speech(x)

    def encode_speech(self, x:torch.Tensor, attention_mask=None):
        # res is a dict with keys ('dense', 'units', 'durations').
        # It can also contain 'f0' if SpeechEncoder was initialized
        # with need_f0=True flag.
        """
        Return:
            a dict with {
                "units" : a list [ (L1), (L2), (L3) ], each element is a unit id list, with no repetition,
                "dense" : a tensor with shape (B, T, 768)
                "original_units": unit id lists with shape (B, T)
            }
        """
        
        assert x.ndim == 2 or (x.ndim == 3 and x.size(1) == 1), 'Input shape must be (B, L) or (B, 1, L)'
        
        if x.ndim == 3: # change shape from (B, 1, L) to (B, L) 
            x = x[:, 0, :]
        
        res = {}
        feats = self.model(x, output_hidden_states=True, attention_mask=attention_mask)
        res['dense'] = feats.last_hidden_state # (B, T, C)
        
        
        # list of (B, T, C). Note that the last_hidden_state is the same as 
        # the last element of hidden_states
        res['hidden_states'] = feats.hidden_states
        
        res['units'], res['original_units'] = self.quantizer_model.predict_batch_units(res['dense'])
        return res

# + tags=["active-ipynb"]
# model = CustomSpeechEncoder()
# audio = torch.randn(4, 1, 48000)
# res = model(audio)
# print(res.keys())
