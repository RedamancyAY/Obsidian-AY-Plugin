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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tqdm import tqdm
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

# +
import pytorch_lightning as pl
from ay2.torch.deepfake_detection import DeepfakeAudioClassification
from ay2.torch.lightning.callbacks import (
    ACC_Callback,
    APCallback,
    AUC_Callback,
    Color_progress_bar,
    EER_Callback,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.WaveLM.wavlm import BaseLine as WaveLM
# -

# ## 数据集

from ._data import get_data


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
        return {"logit": out}


# ### callback & trainer

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


def train_SER(gpu=0):
    dl = get_data()

    ROOT_DIR = "/home/ay/data/DATA/1-model_save/0-Audio"
    
    trainer = pl.Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices=[gpu],
        logger=pl.loggers.CSVLogger(
            ROOT_DIR,
            name="speech_emotion_recognition",
            version=None,
        ),
        check_val_every_n_epoch=1,
        callbacks=make_callbacks(),
        default_root_dir=ROOT_DIR,
    )

    model = WaveLM_lit(num_classes=13)
    trainer.fit(model, dl.train, val_dataloaders=dl.val)
    trainer.test(model, dl.test)
