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

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ay2.torch.deepfake_detection import DeepfakeAudioClassification

from torchaudio.transforms import LFCC

from .lcnn import LCNN


class LCNN_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LCNN(num_class=1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

        self.lfcc = LFCC(
            n_lfcc=60,
            speckwargs={"n_fft": 400, "hop_length": 160, "center": False},
        )

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=0.0001
        )
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        audio=self.lfcc(audio)
        
        # batch_out = self.model(audio).squeeze()
        feature = self.model.extract_feature(audio)
        batch_out = self.model.make_prediction(feature).squeeze(-1)
        batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
        return {"logit": batch_out, "pred": batch_pred, "feature" : feature}

