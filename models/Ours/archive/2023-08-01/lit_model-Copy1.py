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
from ay2.torch.losses import BinaryTokenContrastLoss

from copy import deepcopy

# + editable=true slideshow={"slide_type": ""}
from .model import AudioModel


# + tags=["active-ipynb", "style-solution"] editable=true slideshow={"slide_type": ""}
# from model import AudioModel

# + editable=true slideshow={"slide_type": ""}
class AudioModel_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.model = AudioModel()
        ckpt = torch.load(
            "/home/ay/data/DATA/1-model_save/0-Audio/speech_emotion_recognition/version_1/model2.ckpt"
        )
        self.model.feature_model.load_state_dict(ckpt, strict=True)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.contrast_loss = BinaryTokenContrastLoss(alpha=0.4)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def calcuate_loss(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]

        label = batch["label"]
        label_32 = label.type(torch.float32)
        losses = {}
        losses["cls_loss"] = self.bce_loss(batch_res["logit"], label_32)
        losses["adv_loss"] = self.bce_loss(batch_res["adv_logit"], label_32)
        # losses["emo_contrast_loss"] = self.contrast_loss(
        #     batch_res["emotion_feature"], label_32
        # )
        # losses["emo_cls_loss"] = self.bce_loss(batch_res["emotion_logit"], label_32)
        losses["emo_cls_loss"] = self.cross_entropy_loss(
            batch_res["emotion_logit"], batch['emotion_label']
        )
        losses["vocoder_cls_loss"] = self.cross_entropy_loss(
            batch_res["vocoder_logit"], batch["vocoder_label"]
        )
        # losses["vocoder_contrast_loss"] = self.contrast_loss(
        #     batch_res["vocoder_feature"], label_32
        # )

        losses["ortho_loss"] = torch.mean(
            torch.abs(
                torch.matmul(
                    batch_res["emotion_feature"].view(B, 1, -1),
                    batch_res["vocoder_feature"].view(B, -1, 1),
                ).squeeze()
            )
        )

        self.log_dict(
            {f"{stage}-{key}": losses[key] for key in losses},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
        )

        loss = [
            losses["cls_loss"],
            losses['adv_loss'],
            # losses["emo_contrast_loss"],
            losses["emo_cls_loss"],
            losses["vocoder_cls_loss"],
            # losses["vocoder_contrast_loss"],
            losses["ortho_loss"],
        ]
        loss = torch.mean(torch.stack(loss))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.00005, weight_decay=0.0001
        )
        return [optimizer]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        batch_res = self.model(audio, stage=stage)
        batch_res["pred"] = (torch.sigmoid(batch_res["logit"]) + 0.5).int()
        return batch_res

    def _shared_eval_step(self, batch, batch_idx, stage="train"):
        batch_res = self._shared_pred(batch, batch_idx, stage=stage)

        label = batch["label"]
        loss = self.calcuate_loss(batch_res, batch, stage=stage)

        self.log_dict(
            {f"{stage}-loss": loss},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        batch_res["loss"] = loss
        # print(batch_res['pred'], batch_res['logit'], label)
        return batch_res
