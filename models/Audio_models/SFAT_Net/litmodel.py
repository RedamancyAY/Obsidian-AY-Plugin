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

# + editable=true slideshow={"slide_type": ""}
# %load_ext autoreload
# %autoreload 2

# + editable=true slideshow={"slide_type": ""}
import math

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms
import torchyin
from einops import rearrange, repeat

# + editable=true slideshow={"slide_type": ""}
from ay2.torch.deepfake_detection import DeepfakeAudioClassification

# + editable=true slideshow={"slide_type": ""}
from .model import SFATNet, F0ReconstructionLoss


# + editable=true slideshow={"slide_type": ""}
class SFATNet_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = SFATNet()
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()
        self.f0_loss = F0ReconstructionLoss()
        self.save_hyperparameters()

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        cls_loss = self.cls_loss(batch_res["logit"], label.type(torch.float32))
        spec_rec_loss = self.mse_loss(batch_res['spec'], batch_res['pred_spec'])
        f0_rec_loss = self.f0_loss(batch["audio"], batch_res['pred_f0'])
        loss = cls_loss + spec_rec_loss + f0_rec_loss
        
        return {
            'loss' : loss,
            'cls_loss' : cls_loss,
            'f0_rec_loss' : f0_rec_loss,
            'spec_rec_loss': spec_rec_loss,
            'aux_loss' : f0_rec_loss + spec_rec_loss
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        return [optimizer], [scheduler]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        batch_res = self.model(audio)
        batch_res['pred'] = (torch.sigmoid(batch_res['logit']) + 0.5).int()
        return batch_res

