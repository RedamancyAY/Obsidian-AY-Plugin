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

# %load_ext autoreload
# %autoreload 2

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ay2.torch.deepfake_detection import DeepfakeAudioClassification

from .Aaasist.load_model import get_model
from .Aaasist.utils import cosine_annealing


class AASIST_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = get_model("AASIST")
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=None)
        self.save_hyperparameters()

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.0001,
            betas=[0.9, 0.999],
            weight_decay=0.0001,
            amsgrad=False,
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                200000, #total_steps
                1,  # since lr_lambda computes multiplicative factor
                0.000005 / 0.0001,
            ),
        )

        return [optimizer], [scheduler]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        feat, logit = self.model(audio)
        logit = logit.squeeze()
        batch_pred = (torch.sigmoid(logit) + 0.5).int()
        return {"logit": logit, "pred": batch_pred, "feature":feat}
