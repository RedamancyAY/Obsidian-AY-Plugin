import os
import statistics
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ay2.torch.deepfake_detection import DeepfakeAudioClassification
from ay2.torch.losses import (
    BinaryTokenContrastLoss,
    CLIPLoss1D,
    LabelSmoothingBCE,
)


try:
    from .model import MultiView3DModel
except ImportError:
    from model import MultiView3DModel


class MultiView3DModel_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        self.model = MultiView3DModel(cfg=cfg, args=args)
        self.cfg = cfg
        self.args = args

        self.configure_loss_fn()
        self.save_hyperparameters()

    def configure_custom_callbacks(self):
        from ay2.torch.lightning.callbacks.metrics import BinaryACC_Callback,BinaryAUC_Callback
        from ay2.torch.lightning.callbacks import EER_Callback
    
        callbacks = [
            BinaryACC_Callback(batch_key="label", output_key="logit1", theme="view1"),
            BinaryACC_Callback(batch_key="label", output_key="logit2", theme="view2"),
            BinaryACC_Callback(batch_key="label", output_key="logit_text", theme="view_text"),
            BinaryAUC_Callback(batch_key="label", output_key="logit1", theme="view1"),
            BinaryAUC_Callback(batch_key="label", output_key="logit2", theme="view2"),
            BinaryAUC_Callback(batch_key="label", output_key="logit_text", theme="view_text"),
            EER_Callback(batch_key="label", output_key="logit1", theme="view1"),
            EER_Callback(batch_key="label", output_key="logit2", theme="view2"),
            EER_Callback(batch_key="label", output_key="logit_text", theme="view_text"),
        ]
        return callbacks

    def configure_loss_fn(
        self,
    ):
        self.bce_loss = LabelSmoothingBCE(label_smoothing=0.1)
        self.contrast_loss2 = BinaryTokenContrastLoss(alpha=0.1)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.clip_loss = CLIPLoss1D()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=0.0001,
            weight_decay=0.00001,
        )
        return [optimizer]

    def calcuate_loss(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]
        label = batch["label"]
        losses = {}


        losses["cls_loss1"] = self.bce_loss(batch_res["logit1"], label)
        losses["cls_loss2"] = self.bce_loss(batch_res["logit2"], label)
        losses["cls_loss_text"] = self.bce_loss(batch_res["logit_text"], label)
        losses["cls_loss"] = self.bce_loss(batch_res["logit"], label)
        losses["loss"] = losses["cls_loss"] + 0.1 * (losses["cls_loss1"] + losses["cls_loss2"] + losses["cls_loss_text"])

        return losses

    def _shared_pred(self, batch, batch_idx, stage="train"):
        """common predict step for train/val/test

        Note that the data augmenation is done in the self.model.feature_extractor.

        """
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        batch_res = self.model(audio)
        batch_res["pred"] = (torch.sigmoid(batch_res["logit"]) + 0.5).int()

        return batch_res
