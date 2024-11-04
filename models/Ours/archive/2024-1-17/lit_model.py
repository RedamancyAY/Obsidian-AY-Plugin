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
import torch.nn.functional as F
from ay2.torch.deepfake_detection import DeepfakeAudioClassification
from ay2.torch.losses import (
    BinaryTokenContrastLoss,
    LabelSmoothingBCE,
    MultiClass_ContrastLoss,
)
from ay2.torch.optim import Adam_GC
from ay2.torch.optim.selective_weight_decay import (
    Optimizers_with_selective_weight_decay,
)
from ay2.torchaudio.transforms import AddGaussianSNR

from copy import deepcopy

# + editable=true slideshow={"slide_type": ""}
try:
    from .model import AudioModel
except ImportError:
    from model import AudioModel


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
            args=args
        )
        
        self.beta1, self.beta2, self.beta3 = cfg.beta

        self.one_stem = cfg.one_stem

        self.bce_loss = LabelSmoothingBCE(label_smoothing=0.1)
        self.contrast_loss2 = BinaryTokenContrastLoss(alpha=0.1)
        self.contrast_lossN = MultiClass_ContrastLoss(alpha=2.5, distance="l2")
        # self.contrast_lossN = MultiClass_ContrastLoss(alpha=0.1)
        self.cross_entropy_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

        # self.transform = AddGaussianSNR(snr_max_db=20)

        self.save_hyperparameters()

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

    def set_grad(self, prop):
        def hook(grad):
            prop = grad

        return hook

    def get_vocoder_stem_loss(self, losses, batch_res, batch, stage="train"):

        if not "vocoder_label" in batch.keys():
            return 0.0
        
        losses["voc_cls_loss"] = self.cross_entropy_loss(
            batch_res["vocoder_logit"], batch["vocoder_label"]
        )

        # if stage == 'val':
        # print(torch.argmax(batch_res["vocoder_logit"], dim=1), batch["vocoder_label"])
        # if stage == 'train':
        #     losses["voc_cls_loss"].register_hook(self.set_grad())

        # + self.cross_entropy_loss(batch_res["vocoder_logit"], batch["emotion_label"])
        losses["voc_contrast_loss"] = self.contrast_lossN(
            batch_res["vocoder_feature"], batch["vocoder_label"].type(torch.float32)
        )
        vocoder_stem_loss = losses["voc_cls_loss"] + losses["voc_contrast_loss"]

        # if stage == 'train':
        #     self.optimizers().zero_grad()
        #     self.vocoder_grad = None
        #     batch_res["hidden_states"].register_hook(self.set_grad(self.vocoder_grad))
        #     losses["voc_cls_loss"].backward(retain_graph=True)

        return vocoder_stem_loss

    def get_content_stem_loss(self, losses, batch_res, batch, stage="train"):
        label_32 = batch["label"].type(torch.float32)
        batch_size = len(label_32)
        losses["content_cls_loss"] = self.bce_loss(
            batch_res["content_logit"].squeeze(), label_32
        )
        losses["content_contrast_loss"] = self.contrast_loss2(
            batch_res["content_feature"], label_32
        )

        vocoder_label = torch.ones_like(batch_res["content_voc_logit"]) * (
            1 / batch_res["content_voc_logit"].shape[-1]
        )
        losses["content_adv_loss"] = self.cross_entropy_loss(
            batch_res["content_voc_logit"],
            vocoder_label,
        )
        content_stem_loss = (
            losses["content_cls_loss"]
            + losses["content_contrast_loss"]
            + losses["content_adv_loss"]
        )
        return content_stem_loss

    def calcuate_loss(self, batch_res, batch, stage="train"):
        B = batch_res["logit"].shape[0]
        label = batch["label"]
        label_32 = label.type(torch.float32)
        losses = {}

        losses["cls_loss"] = self.bce_loss(batch_res["logit"].squeeze(), label_32)
        vocoder_stem_loss = self.get_vocoder_stem_loss(losses, batch_res, batch, stage)
        content_stem_loss = self.get_content_stem_loss(losses, batch_res, batch, stage)

        losses["feature_similar_loss"] = reduce_similarity(
            batch_res["content_feature"], batch_res["vocoder_feature"]
        )
        similar_loss = losses["feature_similar_loss"]


        losses["loss"] = content_stem_loss + 0.5 * vocoder_stem_loss + 0.5 * similar_loss

        return losses

    def loss_adjustment(self, auxiliary_loss, main_loss, sigma=0.5):
        while auxiliary_loss > main_loss * sigma:
            auxiliary_loss = auxiliary_loss * 0.5
        return auxiliary_loss

    def configure_optimizers(self):
        optim = Adam_GC  # or torch.optim.Adam
        # optim = torch.optim.Adam
        # optimizer = optim(self.model.parameters(), lr=0.00001, weight_decay=0.0001)

        # optimizer = Optimizers_with_selective_weight_decay(
        #     self.model, optimizer="Adam", lr=0.0001, weight_decay=0.01
        # )
        optimizer = Optimizers_with_selective_weight_decay(
            self.model, optimizer="SGD", lr=0.001, weight_decay=0.01
        )
        return [optimizer]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        # if stage == 'train':
        # audio = self.transform(audio)

        batch_res = self.model(
            audio, stage=stage, batch=batch if stage == "train" else None
        )

        # batch_res = self.model(audio, stage=stage, batch=batch)
        batch_res["pred"] = (torch.sigmoid(batch_res["logit"]) + 0.5).int()

        return batch_res

    def _shared_eval_step(
        self, batch, batch_idx, stage="train", dataloader_idx=0, *args, **kwargs
    ):
        batch_res = self._shared_pred(batch, batch_idx, stage=stage)

        label = batch["label"]

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
