# ---
# jupyter:
#   jupytext:
#     formats: py:light
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

# +
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
    Focal_loss,
    LabelSmoothingBCE,
    MultiClass_ContrastLoss,
    CLIPLoss1D
)
from ay2.torch.optim import Adam_GC
from ay2.torch.optim.selective_weight_decay import (
    Optimizers_with_selective_weight_decay,
    Optimizers_with_selective_weight_decay_for_modulelist,
)
from ay2.torchaudio.transforms import AddGaussianSNR
from ay2.torchaudio.transforms.self_operation import (
    AudioToTensor,
    CentralAudioClip,
    RandomAudioClip,
    RandomPitchShift,
    RandomSpeed,
)
from tqdm.auto import tqdm

# -

from ay2.tools import (
    find_unsame_name_for_file,
    freeze_modules,
    rich_bar,
    unfreeze_modules,
)

from ay2.torchaudio.transforms import SpecAugmentBatchTransform


# + editable=true slideshow={"slide_type": ""}
try:
    from .multiviewCombine import MultiViewCombine
except ImportError:
    from multiviewCombine import MultiViewCombine


# + editable=true slideshow={"slide_type": ""}
class MultiViewCombine_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        self.model = MultiViewCombine(cfg=cfg, args=args)
        self.cfg = cfg
        self.args = args

        self.configure_loss_fn()
        self.save_hyperparameters()

    def configure_loss_fn(
        self,
    ):
        self.bce_loss = LabelSmoothingBCE(label_smoothing=0.1)
        self.contrast_loss2 = BinaryTokenContrastLoss(alpha=0.1)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.cliploss = CLIPLoss1D()
        
    
    def configure_optimizers(self):
        optimizer = Optimizers_with_selective_weight_decay_for_modulelist(
            [self.model],
            optimizer="Adam",
            lr=0.0001,
            weight_decay=0.01,
        )

        return [optimizer]


    def calcuate_loss(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]
        label = batch["label"]
        losses = {}

        losses["cls_loss1D"] = self.bce_loss(batch_res["logit1D"], label)
        losses["cls_loss2D"] = self.bce_loss(batch_res["logit2D"], label)
        losses["cls_loss"] = self.bce_loss(batch_res["logit"], label)
        losses["loss"] = losses["cls_loss"] + 0.5 * (losses["cls_loss1D"] + losses["cls_loss2D"])
        return losses


    def _shared_pred(self, batch, batch_idx, stage="train"):
        """common predict step for train/val/test

        Note that the data augmenation is done in the self.model.feature_extractor.

        """
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        batch_res = self.model(
            audio,
            stage=stage,
            batch=batch if stage == "train" else None,
            trainer=self.trainer,
        )

        batch_res["pred"] = (torch.sigmoid(batch_res["logit"]) + 0.5).int()
        return batch_res
