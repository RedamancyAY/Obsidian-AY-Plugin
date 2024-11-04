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
from tqdm.auto import tqdm
from ay2.torchaudio.transforms.self_operation import (
    AudioToTensor,
    CentralAudioClip,
    RandomAudioClip,
    RandomPitchShift,
    RandomSpeed,
)

# -

from ay2.tools import (
    freeze_modules,
    rich_bar,
    unfreeze_modules,
    find_unsame_name_for_file,
)

# +
from ay2.torchaudio.transforms import SpecAugmentBatchTransform
from ay2.torchaudio.transforms.self_operation import RandomSpeed

random_speed = RandomSpeed(min_speed=0.5, max_speed=2.0, p=0.5)

# + editable=true slideshow={"slide_type": ""}
try:
    from .model import AudioModel
    from .utils import OrthogonalProjectionLoss
    from .mi_estimator import CLUBSample_group
    from ._cs_models import AudioCSModule
except ImportError:
    from model import AudioModel
    from utils import OrthogonalProjectionLoss
    from mi_estimator import CLUBSample_group
    from _cs_models import AudioCSModule

# +
def set_grad(var):
    def hook(grad):
        var.grad = grad

    return hook


def reduce_similarity(feature1, feature2):
    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(feature1, feature2)

    # 对余弦相似度进行处理以减小相似性
    reduced_similarity = torch.mean(1 - cosine_similarity)

    return reduced_similarity


def mixup(data, targets, alpha=0.8):
    indices = torch.randperm(data.size(0))
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)
    new_data = data * lam + shuffled_data * (1 - lam)
    new_targets = [targets, shuffled_targets, lam]
    return new_data, new_targets


# + editable=true slideshow={"slide_type": ""}
class AudioModel_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        self.model = AudioModel(
            feature_extractor=cfg.feature_extractor,
            dims=cfg.dims,
            n_blocks=cfg.n_blocks,
            vocoder_classes=cfg.method_classes,
            cfg=cfg,
            args=args,
        )
        self.cfg = cfg
        self.beta1, self.beta2, self.beta3 = cfg.beta

        self.one_stem = cfg.one_stem

        # self.transform = AddGaussianSNR(snr_max_db=20)
        self.audio_transform = SpecAugmentBatchTransform.from_policy(cfg.aug_policy)
        self.ttt_transform = [
            RandomSpeed(min_speed=0.5, max_speed=2.0, p=1),
            CentralAudioClip(length=48000),
        ]
        self.ttt_transform = AddGaussianSNR(snr_max_db=5)

        self.automatic_optimization = False

        self.mixup = False
        # freeze_modules(self.model.feature_model.get_main_stem())


        # self.cs_model = AudioCSModule(1600, 0.25)
        
        self.configure_loss_fn()
        self.save_hyperparameters()

    def configure_loss_fn(
        self,
    ):
        self.bce_loss = LabelSmoothingBCE(label_smoothing=0.1)
        self.contrast_loss2 = BinaryTokenContrastLoss(alpha=0.1)
        self.contrast_lossN = MultiClass_ContrastLoss(alpha=2.5, distance="l2")
        # self.contrast_lossN = MultiClass_ContrastLoss(alpha=0.1)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        from torchvision.ops import sigmoid_focal_loss

        self.focal_loss = partial(sigmoid_focal_loss, reduction="mean")
        self.op_loss = OrthogonalProjectionLoss(gamma=0.5)
        self.feat_mi_esti = CLUBSample_group(512, 512, 512)
    
    def calcuate_loss_one_stem(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]

        batch_res["logit"] = batch_res["content_logit"]
        label = batch["label"]
        label_32 = label.type(torch.float32)
        losses = {}

        losses["cls_loss"] = self.bce_loss(batch_res["logit"].squeeze(), label_32)
        losses["content_contrast_loss"] = self.contrast_loss2(
            batch_res["content_feature"], label_32
        )

        # loss = sum(losses.values()) / (len(losses))
        losses["loss"] = (losses["cls_loss"] + losses["content_contrast_loss"]) / 2
        return losses

    # Copied and edited from https://www.kaggle.com/code/riadalmadani/fastai-effb0-base-model-birdclef2023
    def binary_classification_loss(self, logit, label, mixup=False):
        logit = logit.squeeze()
        if not mixup:
            return self.bce_loss(logit, label.type(torch.float32))
        else:
            targets1, targets2, lam = label[0:3]
            return lam * self.bce_loss(logit, targets1.type(torch.float32)) + (
                1 - lam
            ) * self.bce_loss(logit, targets2.type(torch.float32))

    def get_vocoder_stem_loss(self, losses, batch_res, batch, stage="train"):
        if not "vocoder_label" in batch.keys():
            return 0.0

        losses["voc_cls_loss"] = self.ce_loss(
            batch_res["vocoder_logit"], batch["vocoder_label"]
        )

        losses["voc_contrast_loss"] = self.contrast_lossN(
            batch_res["vocoder_feature"], batch["vocoder_label"].type(torch.float32)
        )
        vocoder_stem_loss = losses["voc_cls_loss"] + 0.5 * losses["voc_contrast_loss"]

        return vocoder_stem_loss

    def get_content_stem_loss(self, losses, batch_res, batch, stage="train"):
        label_32 = batch["label"].type(torch.float32)
        batch_size = len(label_32)

        losses["content_cls_loss"] = self.binary_classification_loss(
            batch_res["content_logit"], label_32
        )
        losses["content_contrast_loss"] = self.contrast_loss2(
            batch_res["content_feature"], label_32
        )
        content_stem_loss = losses["content_cls_loss"] + losses["content_contrast_loss"]
        return content_stem_loss

    def get_content_adv_loss(self, batch_res, batch, stage="train"):
        vocoder_label = torch.ones_like(batch_res["content_voc_logit"]) * (
            1 / batch_res["content_voc_logit"].shape[-1]
        )
        loss = self.ce_loss(
            batch_res["content_voc_logit"],
            vocoder_label,
        )
        return loss
    def Entropy_loss(self, logit):
        def entropy(predict_prob):
            l = -predict_prob * torch.log(predict_prob)
            return l.sum(-1)
        if logit.ndim == 1:
            prob = torch.sigmoid(logit)
            prob = torch.stack([prob, 1-prob], dim=0)
        else:
            prob = torch.softmax(logit, dim=-1)
        return torch.mean(entropy(prob))

    
    def calcuate_loss(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]
        label = batch["label"]
        losses = {}

        losses["cls_loss"] = self.binary_classification_loss(batch_res["logit"], label)
        if stage == "train":
            losses["cls_loss"] += self.focal_loss(
                batch_res["shuffle_logit"], batch["shuffle_label"].type(torch.float32)
            )

        losses["vocoder_stem_loss"] = self.get_vocoder_stem_loss(
            losses, batch_res, batch, stage
        )
        # content_stem_loss = self.get_content_stem_loss(losses, batch_res, batch, stage)

        losses["speed_loss"] = self.ce_loss(
            batch_res["speed_logit"],
            batch["speed_label"].long(),
        ) 
        losses["compression_loss"] = self.ce_loss(
            batch_res["compression_logit"],
            batch["compression_label"].long(),
        )
        # print(batch["speed_label"], batch["compression_label"])
        # + 0.5 * self.contrast_lossN(
            # batch_res["content_feature"], batch["speed"].type(torch.float32)
        # )

        losses["op_loss"] = 0.0
        if self.cfg.use_op_loss:
            # losses["op_loss"] = self.op_loss(batch_res["feature"], label)
            losses["op_loss"] = self.contrast_loss2(
                batch_res["feature"], label
            )
        # losses['mi_loss'] = self.feat_mi_esti.mi_est(batch_res["content_feature"], batch_res["vocoder_feature"][:, None, :])
        losses['entropy_loss'] = self.Entropy_loss(batch_res["logit"])
        
        
        
        if self.trainer.current_epoch < 0:
            losses["loss"] = (
                losses["vocoder_stem_loss"]
                + 1.0 * (losses["speed_loss"] + losses["compression_loss"])
            )
        else:
            losses["loss"] = (
                losses["cls_loss"]
                # + 0.5 * content_stem_loss
                + 0.5 * losses["vocoder_stem_loss"]
                + 0.5 * (losses["speed_loss"] + losses["compression_loss"])
                + 0.5 * losses["op_loss"]
                # + 0.5 * losses['entropy_loss']
            )

        return losses

    def loss_adjustment(self, auxiliary_loss, main_loss, sigma=0.5):
        while auxiliary_loss > main_loss * sigma:
            auxiliary_loss = auxiliary_loss * 0.9
        return auxiliary_loss

    def configure_optimizers(self):
        optim = Adam_GC  # or torch.optim.Adam
        # optim = torch.optim.Adam
        # optimizer = optim(self.model.parameters(), lr=0.00001, weight_decay=0.0001)

        optimizer = Optimizers_with_selective_weight_decay_for_modulelist(
            [self.model, self.feat_mi_esti], optimizer="Adam", lr=0.0001, weight_decay=0.01
        )
        # optimizer = Optimizers_with_selective_weight_decay(
        #     self.model, optimizer="SGD", lr=0.001, weight_decay=0.01
        # )
        return [optimizer]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        """common predict step for train/val/test

        Note that the data augmenation is done in the self.model.feature_extractor.

        """
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        # audio_re, _, _ = self.cs_model(audio)
        audio = self.model.feature_model.preprocess(audio, stage=stage)
        if stage == "train" and self.cfg.feature_extractor == 'ResNet':
            audio = self.audio_transform.batch_apply(audio)
            # print('hello')
        batch_res = self.model(
            audio, stage=stage, batch=batch if stage == "train" else None
        )

        # batch_res["pred"] = (torch.sigmoid(batch_res["logit"]) + 0.5).int()
        batch_res["pred"] = batch_res["logit"]

        return batch_res

    def _shared_eval_step(
        self,
        batch,
        batch_idx,
        stage="train",
        loss=True,
        dataloader_idx=0,
        *args,
        **kwargs,
    ):
        """common evaluation step for train/val/test

        In contrast to the predict step, this evaluation step calculates the losses and logs
        them to logger.

        """
        batch_res = self._shared_pred(batch, batch_idx, stage=stage)
        label = batch["label"]

        if not loss:
            return batch_res

        if not self.one_stem:
            loss = self.calcuate_loss(batch_res, batch, stage=stage)
        else:
            loss = self.calcuate_loss_one_stem(batch_res, batch, stage=stage)

        suffix = "" if dataloader_idx == 0 else f"-dl{dataloader_idx}"
        self.log_dict(
            {f"{stage}-{key}{suffix}": loss[key] for key in loss},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=batch["label"].shape[0],
        )
        batch_res.update(loss)
        return batch_res

    def training_step(self, batch, batch_idx):
        """custom training step for twp-step parameter updating."""

        opt1 = self.optimizers()
        with torch.autograd.set_detect_anomaly(True):
            batch_res = self._shared_eval_step(batch, batch_idx, stage="train")
    
            opt1.zero_grad()
            self.manual_backward(batch_res["loss"], retain_graph=True)
            # opt1.step()
    
            freeze_modules([self.model.cls_voc, self.model.feature_model.get_main_stem()])
            batch_res["content_adv_loss"] = 0.5 * self.get_content_adv_loss(
                batch_res, batch
            )
            self.manual_backward(batch_res["content_adv_loss"])
            unfreeze_modules([self.model.cls_voc, self.model.feature_model.get_main_stem()])
    
            opt1.step()

        return batch_res

# +
# x = torch.randn(3, 10)
# torch.std(x, dim=1)
