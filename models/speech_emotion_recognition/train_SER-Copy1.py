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

# +
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import IPython.display as ipd
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from ay2.datasets import ASVP_ESD
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tqdm import tqdm
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from transformers.file_utils import ModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)
# -

from data.tools import WaveDataset

import pytorch_lightning as pl
from ay2.torch.deepfake_detection import DeepfakeAudioClassification
from ay2.torch.lightning.callbacks import (
    ACC_Callback,
    APCallback,
    AUC_Callback,
    Color_progress_bar,
    EER_Callback,
)
from models import AudioModel_lit
from models.WaveLM.wavlm import BaseLine as WaveLM
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# ## 数据集

from ay2.tools.pandas import DF_spliter

dataset = ASVP_ESD()
data = dataset.data
data = data.query("isSpeech == 1")

train, val, test = DF_spliter.split_by_percentage(data, splits=[0.8, 0.1, 0.1])

# + editable=true slideshow={"slide_type": ""}
datasets = []
data_loaders = []
for i, _d in enumerate([train, val, test]):
    _ds = WaveDataset(
        data,
        sample_rate=16000,
        trim=False,
        max_wave_length=48000,
        transform=None,
        is_training=True if i == 0 else False,
    )
    _dl = torch.utils.data.DataLoader(
        _ds,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        prefetch_factor=2,
    )
    data_loaders.append(_dl)


# -

# ## 预训练模型

class WaveLM_lit(DeepfakeAudioClassification):
    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.model = WaveLM(num_classes=13)
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=0.0001
        )
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        out = self.model(audio)
        return {
            "logit": out
        }


# ### callback & trainer

# +
def make_callbacks():
    callbacks = [
        Color_progress_bar(),
        ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            monitor="val-loss",
            mode="min",
            save_last=True,
            filename="best-{epoch}-{val-loss:.2f}",
        ),
        EarlyStopping(
            monitor="val-auc",
            min_delta=0.0001,
            patience=5,
            mode="max",
            stopping_threshold=0.999,
            verbose=True,
        ),
        AUC_Callback(batch_key="label", output_key="logit", num_classes=13),
    ]
    return callbacks


ROOT_DIR = "/home/ay/data/DATA/1-model_save/0-Audio"
trainer = pl.Trainer(
    max_epochs=300,
    accelerator="gpu",
    devices=[0],
    logger=pl.loggers.CSVLogger(
        ROOT_DIR,
        name="SER",
        version=0,
    ),
    check_val_every_n_epoch=1,
    callbacks=make_callbacks(),
    default_root_dir=ROOT_DIR,
)
# -

model = WaveLM_lit(num_classes=13)

trainer.fit(model, data_loaders[0], val_dataloaders=data_loaders[1])

# ## Our's model

ckpt_path = "/home/ay/data/DATA/1-model_save/0-Audio/SER/version_0/checkpoints/best-epoch=94-val-loss=0.12.ckpt"
model1 = WaveLM_lit(num_classes=13)
model1 = model1.load_from_checkpoint(ckpt_path).to("cpu")

from models.Ours.model import AudioModel


# ### Loss function

class CLIPLoss1D(nn.Module):
    def __init__(self):
        super(CLIPLoss1D, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_image = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        batch_size = image_features.shape[0]
        ground_truth = torch.arange(
            batch_size, dtype=torch.long, device=image_features.device
        )
        return (
            self.loss_image(logits_per_image, ground_truth)
            + self.loss_text(logits_per_text, ground_truth)
        ) / 2


class Our_lit(DeepfakeAudioClassification):
    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.model1 = model1.model
        self.model2 = AudioModel()
        self.mlp = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 512)
        )
        self.loss_fn = CLIPLoss1D()
        self.save_hyperparameters()

    def calcuate_loss(self, batch_res, batch):
        loss = self.loss_fn(batch_res["feat_org"], batch_res["feat_tar"])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model2.parameters(), lr=0.0001, weight_decay=0.0001
        )
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        # if len(audio.shape) == 3:
            # audio = audio[:, 0, :]
            
        with torch.no_grad():
            feature_org = self.model1.get_feature(audio[:, 0, :])
        feature_tar = self.model2.get_feature(audio)
        feature_tar = self.mlp(feature_tar)
        
        return {"feat_org": feature_org, "feat_tar": feature_tar}


# +
def make_callbacks():
    callbacks = [
        Color_progress_bar(),
        ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            monitor="val-loss",
            mode="min",
            save_last=True,
            filename="best-{epoch}-{val-loss:.2f}",
        ),
        EarlyStopping(
            monitor="val-loss",
            min_delta=0.001,
            patience=10,
            mode="min",
            stopping_threshold=0.1,
            verbose=True,
        ),
    ]
    return callbacks


trainer = pl.Trainer(
    max_epochs=300,
    accelerator="gpu",
    devices=[0],
    logger=pl.loggers.CSVLogger(
        ROOT_DIR,
        name="SER",
        version=1,
    ),
    check_val_every_n_epoch=1,
    callbacks=make_callbacks(),
    default_root_dir=ROOT_DIR,
)
model = Our_lit()
# -

trainer.fit(model, data_loaders[0], val_dataloaders=data_loaders[1])

torch.save(model.model2.state_dict(), "/home/ay/data/DATA/1-model_save/0-Audio/SER/version_1/model2.ckpt")

# +
model = AudioModel()

ckpt = torch.load("/home/ay/data/DATA/1-model_save/0-Audio/SER/version_1/model2.ckpt")
model.load_state_dict(ckpt)
