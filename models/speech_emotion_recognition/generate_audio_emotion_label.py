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

import sys

sys.path.append('/home/ay/zky/Coding/0-Audio')

# +
import json, os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tqdm import tqdm
from rich.progress import track
import pytorch_lightning as pl


# + editable=true slideshow={"slide_type": ""} tags=["style-activity", "active-ipynb"]
# from models.speech_emotion_recognition.train_model import WaveLM_lit
# from data.datasets import WaveFake, LibriSeVoc
# from data.tools import WaveDataset
#
# -

# ## Our's model

def get_pretrained_model():
    ckpt_path = "/home/ay/data/DATA/1-model_save/0-Audio/speech_emotion_recognition/version_0/checkpoints/best-epoch=110-val-loss=0.16.ckpt"
    model1 = WaveLM_lit(num_classes=13)
    model1 = model1.load_from_checkpoint(ckpt_path).to("cpu")
    return model1.model


# + editable=true slideshow={"slide_type": ""} tags=["style-activity", "active-ipynb"]
# dataset = WaveFake()
# data = dataset.data

# + editable=true slideshow={"slide_type": ""}
_ds = WaveDataset(
    data,
    sample_rate=16000,
    trim=False,
    max_wave_length=-1,
    transform=None,
    is_training=False,
)
_dl = torch.utils.data.DataLoader(
    _ds,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    shuffle=False,
    prefetch_factor=2,
)
# -

model = get_pretrained_model()
model = model.to('cuda:0')

# + editable=true slideshow={"slide_type": ""}
output_file = os.path.join(dataset.root_path, 'emotion.json')

try:
    with open(output_file, "r") as file:
        json_data = json.load(file)
        res = dict(json_data)
except FileNotFoundError:
    res = {}



for i, x in track(enumerate(_dl), total=len(_dl)):
    name = x['name'][0]
    if name in res.keys():
        continue

    x = x['audio'].squeeze_(dim=1)
    y = model(x.to('cuda:0'))
    labels = list(torch.max(y, dim=1).indices.detach().cpu().numpy())

    res[name] = int(labels[0])
    if i % 1000 == 0 or i == len(_dl) - 1:
        with open(output_file, "w") as file:
            json.dump(res, file)
