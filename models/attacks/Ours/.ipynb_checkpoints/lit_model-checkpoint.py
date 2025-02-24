# ---
# jupyter:
#   jupytext:
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

import torch
import torch.nn as nn
from ay2.tools import freeze_modules

import pytorch_lightning as pl

try:
    from ._cs_models import AudioCSModule
except ImportError:
    from _cs_models import AudioCSModule


class AudioAttackModel(pl.LightningModule):
    def __init__(self, audio_detection_model=None, cfg=None, args=None, **kwargs):
        super().__init__()
        self.N = 1600
        self.sr = 0.25
        self.cs_model = AudioCSModule(N=self.N, sr=self.sr)
        self.audio_detection_model = audio_detection_model
        if audio_detection_model is not None:
            freeze_modules(self.audio_detection_model)

        self.cfg = cfg
        self.args = args
        self.pretrain_CS = args.pretrain_CS
        
        self.configure_loss_fn()

        # self.save_hyperparameters()

    def configure_optimizers(
        self,
    ):
        optimizer = torch.optim.Adam(self.cs_model.parameters(), lr=0.0003, weight_decay=0.0001)
        return [optimizer]

    def configure_loss_fn(self):
        self.loss_fn = nn.MSELoss()
        self.ce_loss = nn.BCEWithLogitsLoss()

    def calcuate_loss(self, batch_res, batch):
        label = batch["audio"]
        re_loss = self.loss_fn(batch_res["x_re"], label) + self.loss_fn(
            batch_res["x0"], label
        )
        orth_loss = torch.mean(
            (
                (self.cs_model.phi @ self.cs_model.psi)
                - torch.eye(self.N).to(self.device)
            )
            ** 2
        )

        loss = re_loss + 0.1 * orth_loss
        losses = {
            "re_loss": re_loss,
            "orth_loss": orth_loss,
            "loss": loss,
        }

        if self.pretrain_CS:
            return losses
        
        logit = batch_res['logit']
        cls_loss = self.ce_loss(logit, 1 - logit)

        if self.current_epoch > 5:
            loss += 0.00001 * cls_loss

        losses.update({
            'cls_loss': cls_loss,
            "loss": loss,
        })
        
        return losses

    def configure_normalizer(self):
        return None

    def normalize_input(self, x):
        if not hasattr(self, "normalizer"):
            self.normalizer = self.configure_normalizer()

        if self.normalizer is not None:
            x = self.normalizer(x)
        return x

    def _shared_pred(self, batch, batch_idx):
        x = batch["audio"]
        x_re, x0, y = self.cs_model(x)

        if self.pretrain_CS:
            logit = batch['label']
        else:
            logit = self.audio_detection_model(x_re)
        # with torch.no_grad():
        #     org_logit = self.audio_detection_model(x)

        return {"x_re": x_re, "x0": x0, "y": y, "logit": logit, 'org_logit':logit}

    def _shared_eval_step(
        self, batch, batch_idx, stage="train", dataloader_idx=0, *args, **kwargs
    ):
        batch_res = self._shared_pred(batch, batch_idx)
        loss = self.calcuate_loss(batch_res, batch)

        if not isinstance(loss, dict):
            loss = {"loss": loss}

        suffix = "" if dataloader_idx == 0 else f"-dl{dataloader_idx}"
        self.log_dict(
            {f"{stage}-{key}{suffix}": loss[key] for key in loss},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=batch["audio"].shape[0],
        )
        batch_res.update(loss)
        return batch_res

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(
            batch, batch_idx, stage="val", dataloader_idx=dataloader_idx
        )

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(
            batch, batch_idx, stage="test", dataloader_idx=dataloader_idx
        )

    def prediction_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_eval_step(
            batch, batch_idx, stage="predict", dataloader_idx=dataloader_idx
        )
