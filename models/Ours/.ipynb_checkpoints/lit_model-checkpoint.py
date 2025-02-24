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
        # self.bce_loss = LabelSmoothingBCE(label_smoothing=0.1)
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.contrast_loss2 = BinaryTokenContrastLoss(alpha=0.1)
        self.contrast_lossN = MultiClass_ContrastLoss(alpha=2.5, distance="l2")
        # self.contrast_lossN = MultiClass_ContrastLoss(alpha=0.1)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        from torchvision.ops import sigmoid_focal_loss

        self.focal_loss = partial(sigmoid_focal_loss, reduction="mean")
        self.op_loss = OrthogonalProjectionLoss(gamma=0.5)
        self.feat_mi_esti = CLUBSample_group(512, 512, 512)


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

        if self.cfg.voc_con_loss:
            vocoder_stem_loss = losses["voc_cls_loss"] + 0.5 * losses["voc_contrast_loss"]
        else:
            vocoder_stem_loss = losses["voc_cls_loss"]
        
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

    # def get_vocoder_adv_loss(self, batch_res, batch, stage="train"):
    #     speed_label = torch.ones_like(batch_res["voc_based_speed_logit"]) * (
    #         1 / batch_res["speed_logit"].shape[-1]
    #     )
    #     loss = self.ce_loss(
    #         batch_res["voc_based_speed_logit"],
    #         speed_label,
    #     )
    #     return loss
    
    def Entropy_loss(self, logit):
        def entropy(predict_prob):
            l = -predict_prob * torch.log(predict_prob)
            return l.sum(-1)

        if logit.ndim == 1:
            prob = torch.sigmoid(logit)
            prob = torch.stack([prob, 1 - prob], dim=0)
        else:
            prob = torch.softmax(logit, dim=-1)
        return torch.mean(entropy(prob))

    def calcuate_loss(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]
        label = batch["label"]
        losses = {}

        losses["cls_loss"] = self.binary_classification_loss(batch_res["logit"], label)
        ## shuffle loss that based on randomly combine content features and vocoder features
        if stage == "train" and self.cfg.feat_shuffle:
            losses["feat_shuffle_loss"] = self.focal_loss(
                batch_res["shuffle_logit"], batch["shuffle_label"].type(torch.float32)
            )
            losses["cls_loss"] += self.cfg.betas[0] *  losses["feat_shuffle_loss"]
        losses["vocoder_stem_loss"] = self.get_vocoder_stem_loss(
            losses, batch_res, batch, stage
        )
        # content_stem_loss = self.get_content_stem_loss(losses, batch_res, batch, stage)

        losses["compression_loss"], losses["speed_loss"] = 0, 0
        if self.cfg.use_speed_loss:
            losses["speed_loss"] = self.ce_loss(
                batch_res["speed_logit"],
                batch["speed_label"].long(),
            )
        if self.cfg.use_compression_loss:
            losses["compression_loss"] = self.ce_loss(
                batch_res["compression_logit"],
                batch["compression_label"].long(),
            )
        losses["f0_loss"] = 0
        if self.cfg.use_f0_loss:
            from .modules.f0_loss import get_f0_loss
            losses["f0_loss"] = get_f0_loss( batch['audio'],
                batch_res["f0_logit"]
            )
            

        losses["feat_contrast_loss"] = 0.0
        if self.cfg.feat_con_loss:
            losses["feat_contrast_loss"] = self.contrast_loss2(batch_res["feature"], label)

        # losses["entropy_loss"] = self.Entropy_loss(batch_res["logit"])

        losses["dis_loss"] = reduce_similarity(
                batch_res["content_feature"], batch_res["vocoder_feature"]
            )


        # losses["content_adv_loss"] = 0.5 * self.get_content_adv_loss(
        #         batch_res, batch
        #     )
        # losses["vocoder_adv_loss"] = 0.5 * self.get_vocoder_adv_loss(
        #         batch_res, batch
        #     )

        losses["loss"] = (
            losses["cls_loss"]
            + self.cfg.betas[1] * losses["vocoder_stem_loss"]
            + self.cfg.betas[2] * (losses["speed_loss"] + losses["compression_loss"] + losses["f0_loss"])
            + self.cfg.betas[3] * losses["feat_contrast_loss"]
        )

        return losses

    
    def calcuate_loss_one_stem(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]

        label = batch["label"]
        label_32 = label.type(torch.float32)
        losses = {}

        if self.cfg.use_only_vocoder_stem:
            batch_res["logit"] = batch_res["vocoder_based_cls_logit"]
            losses["cls_loss"] = self.bce_loss(batch_res["logit"].squeeze(), label_32)
            losses["vocoder_stem_loss"] = self.get_vocoder_stem_loss(
                losses, batch_res, batch, stage
            )
            losses["vocoder_contrast_loss"] = self.contrast_loss2(
                batch_res["vocoder_feature"], label_32
            )
            losses["loss"] = losses["cls_loss"] + 0.5 * (losses["vocoder_contrast_loss"] + losses["vocoder_stem_loss"])
        else:
            batch_res["logit"] = batch_res["content_based_cls_logit"]
            losses["cls_loss"] = self.bce_loss(batch_res["logit"].squeeze(), label_32)
            losses["content_contrast_loss"] = self.contrast_loss2(
                batch_res["content_feature"], label_32
            )
            losses["speed_compression"] = 0
            if  self.cfg.use_speed_loss_in_one_stem and  self.cfg.use_compression_loss_in_one_stem:
                losses["speed_compression"] = self.ce_loss(
                    batch_res["speed_logit"],
                    batch["speed_label"].long(),
                ) + self.ce_loss(
                    batch_res["compression_logit"],
                    batch["compression_label"].long(),
                )

                
            losses["loss"] = losses["cls_loss"] + 0.5 * losses["content_contrast_loss"] + 0.5 * losses["speed_compression"]
        return losses


    def configure_optimizers(self):
        # optim = Adam_GC  # or torch.optim.Adam
        # optim = torch.optim.AdamW
        # optimizer = optim(self.model.parameters(), lr=0.0001, weight_decay=0.0001)

        optimizer = Optimizers_with_selective_weight_decay_for_modulelist(
            [self.model],
            optimizer="Adam",
            lr=0.0001,
            weight_decay=0.0001,
        )
        
        optimizer_adv = Optimizers_with_selective_weight_decay_for_modulelist(
            self.model.get_content_stream_modules(),
            optimizer="Adam",
            lr=0.0001,
            weight_decay=0.01,
        )
        
        # optimizer = Optimizers_with_selective_weight_decay(
        #     self.model, optimizer="SGD", lr=0.001, weight_decay=0.0001
        # )
        return [optimizer, optimizer_adv]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        """common predict step for train/val/test

        Note that the data augmenation is done in the self.model.feature_extractor.

        """
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        # audio_re, _, _ = self.cs_model(audio)
        audio = self.model.feature_model.preprocess(audio, stage=stage)
        if stage == "train" and self.cfg.feature_extractor == "ResNet":
            audio = self.audio_transform.batch_apply(audio)
        batch_res = self.model(
            audio, stage=stage, batch=batch if stage == "train" else None,one_stem=self.one_stem
        )

        batch_res["pred"] = (torch.sigmoid(batch_res["logit"]) + 0.5).int()
        # batch_res["pred"] = batch_res["logit"]

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


    def get_content_adv_loss2(self, content_voc_logit):
        vocoder_label = torch.ones_like(content_voc_logit) * (
            1 / content_voc_logit.shape[-1]
        )
        loss = self.ce_loss(
            content_voc_logit,
            vocoder_label,
        )
        return loss

    
    def training_step(self, batch, batch_idx):
        """custom training step for twp-step parameter updating."""

        # if not self.automatic_optimization:
            # return self.training_step_with_sam(batch, batch_idx)

        opt1, opt_adv = self.optimizers()
        with torch.autograd.set_detect_anomaly(True):
            opt1.zero_grad()
            
            batch_res = self._shared_eval_step(batch, batch_idx, stage="train")

            
            self.manual_backward(batch_res["loss"], retain_graph=False)
            # opt1.step()
            
            if ((not self.one_stem) and self.cfg.use_adversarial_loss) or (self.one_stem and (not self.cfg.use_only_vocoder_stem)):
                    # opt_adv.zero_grad()
                    # print('use adv loss')
                    try:
                        content_feature, _ = self.model.feature_model.get_final_feature(
                            batch_res["hidden_states"].detach()
                        )
                    except ValueError:
                        content_feature = self.model.feature_model.get_final_feature(
                            batch_res["hidden_states"].detach()
                        )
                
                    content_voc_logit = self.model.cls_voc(self.model.dropout(content_feature))
                    
                    batch_res["content_adv_loss"] = self.cfg.betas[2] * self.get_content_adv_loss2(
                        content_voc_logit
                    )
                    freeze_modules(
                        [self.model.cls_voc, self.model.feature_model.get_main_stem()]
                    )
                    self.manual_backward(batch_res["content_adv_loss"], retain_graph=False)
                    unfreeze_modules(
                        [self.model.cls_voc, self.model.feature_model.get_main_stem()]
                    )       
            opt1.step()
        

        return batch_res


    # def training_step_with_sam(self, batch, batch_idx):

    #     opt, opt_adv = self.optimizers()

    #     for i in range(2):
    #         self.sam_optimizer.zero_grad()
    #         batch_res = self._shared_eval_step(batch, batch_idx, stage="train")

    #         self.manual_backward(batch_res["loss"], retain_graph=False)
    #         if not self.one_stem:
    #             if self.cfg.use_adversarial_loss:
    #                 content_feature, _ = self.model.feature_model.get_final_feature(
    #                     batch_res["hidden_states"].detach()
    #                 )
    #                 content_voc_logit = self.model.cls_voc(self.model.dropout(content_feature))
    #                 batch_res["content_adv_loss"] = self.cfg.betas[2] * self.get_content_adv_loss2(
    #                     content_voc_logit
    #                 )
    #                 freeze_modules(
    #                     [self.model.cls_voc, self.model.feature_model.get_main_stem()]
    #                 )
    #                 self.manual_backward(batch_res["content_adv_loss"], retain_graph=False)
    #                 unfreeze_modules(
    #                     [self.model.cls_voc, self.model.feature_model.get_main_stem()]
    #                 )
    #         if i == 0:
    #             final_res = batch_res
    #         if i == 0:
    #             self.sam_optimizer.first_step(zero_grad=True)
    #         else:
    #             self.sam_optimizer.second_step(zero_grad=True)

    #     opt.zero_grad()
    #     opt.step()

    #     return batch_res