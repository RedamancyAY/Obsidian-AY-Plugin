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

# +
from ay2.torchaudio.transforms import SpecAugmentBatchTransform


# + editable=true slideshow={"slide_type": ""}
try:
    from .multiView_model import MultiViewModel
except ImportError:
    from multiView_model import MultiViewModel


# + editable=true slideshow={"slide_type": ""}
class MultiViewModel_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        self.model = MultiViewModel(cfg=cfg, args=args)
        self.cfg = cfg

        self.spec_transform = SpecAugmentBatchTransform.from_policy(cfg.aug_policy)

        self.configure_loss_fn()
        self.save_hyperparameters()

    def configure_loss_fn(
        self,
    ):
        self.bce_loss = LabelSmoothingBCE(label_smoothing=0.1)
        self.contrast_loss2 = BinaryTokenContrastLoss(alpha=0.1)
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)


    def get_shuffle_ids(self, B):
        
        shuffle_ids = torch.randperm(B) 
        while 0 in (shuffle_ids- torch.arange(B)):
            shuffle_ids = torch.randperm(B)
        return shuffle_ids
    
    def calcuate_loss(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]
        label = batch["label"]
        losses = {}

        losses["cls_loss1D"] = self.bce_loss(batch_res["logit1D"], label)
        losses["cls_loss2D"] = self.bce_loss(batch_res["logit2D"], label)
        losses["cls_loss"] = self.bce_loss(batch_res["logit"], label)
        losses["contrast_loss"] = self.contrast_loss2(batch_res["feature"], label)
        losses["contrast_loss1D"] = self.contrast_loss2(batch_res["feature1D"], label)
        losses["contrast_loss2D"] = self.contrast_loss2(batch_res["feature2D"], label)

        losses["triplet_loss"] = self.triplet_loss(
            batch_res["feature1D"],
            batch_res["feature2D"],
            batch_res["feature1D"][self.get_shuffle_ids(B)],
        ) + self.triplet_loss(
            batch_res["feature2D"],
            batch_res["feature1D"],
            batch_res["feature2D"][self.get_shuffle_ids(B)],
        )

        if self.trainer.current_epoch  < -1:
            losses["loss"] = 0.5 * losses["triplet_loss"] + 0.5 * (losses["contrast_loss"] + losses["contrast_loss1D"] + losses["contrast_loss2D"])
        else:
            losses["loss"] = (
                losses["cls_loss"] 
                + 0.1 * losses["triplet_loss"]
                + 0.1 * (losses["contrast_loss"])
                + 0.1 * (losses["contrast_loss1D"] + losses["contrast_loss2D"])
                + 0.5 * (losses["cls_loss1D"] + losses["cls_loss2D"])
            )

        return losses

    def remove_parameters_from_total(self, total, removed):
        removed_ids = [id(x) for x in removed]
        new = []
        for x in total:
            if not id(x) in removed_ids:
                new.append(x)
        return new

    def configure_optimizers(self):
        optimizer = Optimizers_with_selective_weight_decay_for_modulelist(
            [self.model],
            optimizer="Adam",
            lr=0.0001,
            weight_decay=0.01,
        )


        return [optimizer]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        """common predict step for train/val/test

        Note that the data augmenation is done in the self.model.feature_extractor.

        """
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        batch_res = self.model(
            audio,
            stage=stage,
            batch=batch if stage == "train" else None,
            # spec_aug = self.spec_transform if stage == "train" else None,
            trainer=self.trainer,
        )

        # if stage == "test":
        #     for i, (logit1D, logit2D, logit) in enumerate(zip(batch_res["logit1D"],batch_res["logit2D"],batch_res["logit"])):
        #         new_logit = (batch_res["logit1D"][i] + batch_res["logit2D"][i]) / 2
        #         if torch.sign(logit) != torch.sign(logit1D) and torch.sign(logit) != torch.sign(logit2D):
        #             if torch.abs(max(logit1D, logit2D)) > torch.abs(logit):
        #                 # batch_res["logit"][i] = max(logit1D, logit2D)
        #                 batch_res["logit"][i] = (logit1D+logit2D) /2

        
        batch_res["pred"] = (torch.sigmoid(batch_res["logit"]) + 0.5).int()

        if stage == "test":
            with open("test-mutliview.txt", "a") as f:
                for i, (pred, label) in enumerate(
                    zip(batch_res["pred"], batch["label"])
                ):
                    # if torch.sign(batch_res["logit"][i]) != torch.sign(batch_res["logit1D"][i]) and torch.sign(batch_res["logit"][i]) != torch.sign(batch_res["logit2D"][i]):
                    if pred != label:
                        f.write(
                            " ".join(
                                str(a)
                                for a in [
                                    batch_idx,
                                    label.item(),
                                    batch_res["logit1D"][i].item(),
                                    batch_res["logit2D"][i].item(),
                                    batch_res["logit"][i].item(),
                                    batch['source'][i] if 'source' in batch.keys() else "NoSource"
                                ]
                            )
                            + "\n"
                        )

        return batch_res
