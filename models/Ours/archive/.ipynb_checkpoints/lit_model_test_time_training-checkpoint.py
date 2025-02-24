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
        self.audio_transform = SpecAugmentBatchTransform.from_policy("ss")
        self.ttt_transform = [
            RandomSpeed(min_speed=0.5, max_speed=2.0, p=1),
            CentralAudioClip(length=48000),
        ]
        self.ttt_transform = AddGaussianSNR(snr_max_db=5)

        self.automatic_optimization = False

        self.mixup = False
        # freeze_modules(self.model.feature_model.get_main_stem())

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
        vocoder_stem_loss = losses["voc_cls_loss"] + losses["voc_contrast_loss"]

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
        content_stem_loss = self.get_content_stem_loss(losses, batch_res, batch, stage)

        losses["speed_loss"] = self.ce_loss(
            batch_res["speed_logit"],
            batch["speed"].long(),
        )

        losses["loss"] = (
            # losses["cls_loss"]
            # + 0.5 * content_stem_loss
            content_stem_loss
            + 0.5 * losses["cls_loss"]
            + +0.5 * losses["vocoder_stem_loss"]
            + 1.0 * losses["speed_loss"]
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

        optimizer = Optimizers_with_selective_weight_decay(
            self.model, optimizer="Adam", lr=0.0001, weight_decay=0.01
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
        audio = self.model.feature_model.preprocess(audio, stage=stage)
        # if stage == "train":
        # audio = self.audio_transform(audio)
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

        batch_res = self._shared_eval_step(batch, batch_idx, stage="train")

        opt1.zero_grad()
        self.manual_backward(batch_res["loss"], retain_graph=True)
        # opt1.step()

        freeze_modules([self.model.cls_voc, self.model.feature_model.get_main_stem()])
        batch_res["content_adv_loss"] = 0.5 * self.get_content_adv_loss(
            batch_res, batch
        )
        # batch_res["content_adv_loss"] = self.loss_adjustment(
        #     batch_res["content_adv_loss"], batch_res["cls_loss"]
        # )
        self.manual_backward(batch_res["content_adv_loss"])
        unfreeze_modules([self.model.cls_voc, self.model.feature_model.get_main_stem()])

        opt1.step()

        return batch_res

    def get_ttt_optimizer(self, *args, **kwargs):
        """get optimizer for test-time training"""
        if hasattr(self, "ttt_optimizer"):
            return self.ttt_optimizer
        else:
            from ay2.tools import color_print

            print("build optimizer for TTT")
            parameters = nn.ModuleList(
                self.model.feature_model.get_main_stem()
                + self.model.feature_model.get_content_stem()
                + self.model.feature_model.get_vocoder_stem()
            ).parameters()
            parameters = self.model.parameters()
            self.ttt_optimizer = torch.optim.SGD(parameters, lr=0.001, momentum=0.9)
            # self.ttt_optimizer = torch.optim.SGD(parameters, lr=0.001)
            # self.ttt_optimizer = torch.optim.Adam(parameters, lr=0.001)
            return self.ttt_optimizer

    def backup_state_dict(self):
        if not hasattr(self, "org_state_dict"):
            self.org_state_dict = deepcopy(self.model.state_dict())

    def restore_state_dict(self):
        self.model.load_state_dict(self.org_state_dict)

    def hash_module(self, m):
        _hashes = []
        for k, v in dict(m.state_dict()).items():
            _h = hash(tuple(v.reshape(-1).tolist()))
            _hashes.append(_h)
        return sum(_hashes)

    # def on_test_start(self):
    #     if hasattr(self, "ttt_optimizer"):
    #         del self.ttt_optimizer

    #     if hasattr(self, "cov_on_train"):
    #         return

    #     log_dir = self.trainer.log_dir
    #     ckpt_path = os.path.join(log_dir, "hello.ckpt")
    #     if os.path.exists(ckpt_path):
    #         (
    #             self.cov_on_train,
    #             self.mu_on_train,
    #             self.coral_on_train,
    #             self.std_on_train,
    #             self.vocoder_feats,
    #         ) = torch.load(ckpt_path, map_location=self.device)
    #         return

    #     feat_all = {"feat": [], "org_feat": [], "voc_feat": [], "content_feat": []}
    #     self.cov_on_train, self.mu_on_train, self.coral_on_train = {}, {}, {}
    #     self.std_on_train = {}
    #     self.vocoder_feats = {"real": [], "fake": []}

    #     with torch.no_grad():
    #         for i, batch in tqdm(enumerate(self.trainer.trainset_wo_transform)):
    #             x = batch["audio"].cuda()
    #             x = self.model.feature_model.preprocess(x)
    #             feat = self.model.feature_model.get_hidden_state(x)
    #             content_feat = self.model.feature_model.get_final_feature(feat)
    #             vocoder_feat = self.model.feature_model.get_final_feature_copyed(feat)

    #             # fmt: off
    #             self.vocoder_feats["fake"].append(vocoder_feat[batch["label"] == 0].cpu())
    #             self.vocoder_feats["real"].append(vocoder_feat[batch["label"] == 1].cpu())
    #             feat_all["feat"].append(F.adaptive_avg_pool2d(feat, 1).squeeze([-1, -2]))
    #             # feat_all["feat"].append(F.adaptive_avg_pool1d(feat, 1).squeeze([-1]))
    #             # feat_all["org_feat"].append(feat)
    #             feat_all["voc_feat"].append(vocoder_feat)
    #             # feat_all["content_feat"].append(content_feat)
    #             # fmt: on

    #     # fmt: off
    #     # print(torch.concat(self.vocoder_feats['real']).shape, torch.concat(self.vocoder_feats['fake']).shape)
    #     n_samples = torch.concat(self.vocoder_feats['real']).shape[0]
    #     ids = torch.randperm(n_samples)
    #     self.vocoder_feats['real'] = torch.concat(self.vocoder_feats['real'])[ids]
    #     self.vocoder_feats['fake'] = torch.concat(self.vocoder_feats['fake'])[ids]
    #     self.vocoder_feats['real'] = self.vocoder_feats['real'].mean(dim=0, keepdims=True).cuda()
    #     self.vocoder_feats['fake'] = self.vocoder_feats['fake'].mean(dim=0, keepdims=True).cuda()
    #     # fmt: on

    #     for key in [
    #         "feat",
    #         # "org_feat",
    #         "voc_feat",
    #         # "content_feat",
    #     ]:
    #         losses = []
    #         _feat_all = feat_all[key]
    #         _cov, _mu = self.get_cov(_feat_all[0]), _feat_all[0].mean(dim=0)
    #         for i in range(1, len(_feat_all)):
    #             loss = self.coral(self.get_cov(_feat_all[i]), _cov)
    #             losses.append(loss.item())

    #         _feat_all = torch.cat(_feat_all)
    #         self.cov_on_train[key] = self.get_cov(_feat_all)
    #         self.mu_on_train[key] = _feat_all.mean(dim=0)
    #         self.std_on_train[key] = _feat_all.std(dim=0)
    #         self.coral_on_train[key] = statistics.mean(losses)

    #     torch.save(
    #         [
    #             self.cov_on_train,
    #             self.mu_on_train,
    #             self.coral_on_train,
    #             self.std_on_train,
    #             self.vocoder_feats,
    #         ],
    #         ckpt_path,
    #     )

    def get_cov(self, feat):
        if feat.ndim == 2:
            return torch.cov(feat.T)
        elif feat.ndim == 3:
            feat = F.adaptive_avg_pool1d(feat, 1).squeeze([-1])
            return torch.cov(feat.T)
        else:
            feat = F.adaptive_avg_pool2d(feat, 1).squeeze([-1, -2])
            return torch.cov(feat.T)

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

    def skewness(self, tensor):
        mean = torch.mean(tensor)
        diff = tensor - mean
        numerator = torch.mean(diff**3)
        denominator = torch.mean(diff**2).pow(1.5)
        return numerator / denominator

    def coral(self, cs, ct):
        d = cs.shape[0]
        loss = (cs - ct).pow(2).sum() / (4.0 * d**2)
        return loss

    def Euclidean_distance(self, feat1, feat2):
        return torch.mean((feat1 - feat2) ** 2)

    def feat_loss(self, key, feat, cls_loss=None, **kwargs):
        if key == "feat":
            if feat.ndim == 4:
                feat = F.adaptive_avg_pool2d(feat, 1).squeeze([-1, -2])
            elif feat.ndim == 3:
                feat = F.adaptive_avg_pool1d(feat, 1).squeeze([-1])

        mu_feat = feat.mean(dim=0)

        cosine_dis = torch.nn.functional.cosine_similarity(
            feat, self.mu_on_train[key][None]
        )
        dis = self.Euclidean_distance(mu_feat, self.mu_on_train[key]).item()
        if dis > 0.03:
            mu_weight = 10
        else:
            mu_weight = 10

        cov_feat = self.get_cov(feat)
        cov_loss = self.coral(self.cov_on_train[key], cov_feat) * (
            0.05 / self.coral_on_train[key]
        )
        mu_loss = (mu_feat - self.mu_on_train[key]).pow(2).mean() * 10
        std_loss = (feat.std(dim=0) - self.std_on_train[key]).pow(2).mean() * 10
        skewness_loss = torch.abs(
            self.skewness(mu_feat) - self.skewness(self.mu_on_train[key])
        )

        debug = 1
        if debug:
            _dict = {
                "cls_loss": "{:.4f}".format(cls_loss.item()),
                "l2-dis": "{:.4f}".format(
                    self.Euclidean_distance(mu_feat, self.mu_on_train[key]).item()
                ),
                "cos-dis": "{:.4f}".format(torch.mean(cosine_dis).item()),
                "mu_loss": "{:.4f}".format(mu_loss.item()),
                "cov_loss": "{:.4f}".format(cov_loss.item()),
                "skewness_loss": "{:.4f}".format(skewness_loss.item()),
            }
            if not hasattr(self, "xxxx"):
                self.xxxx = 0
            if not hasattr(self, "data"):
                self.data = pd.DataFrame(columns=list(_dict.keys()))
            self.data.loc[len(self.data)] = _dict
            self.xxxx += 1
            if self.xxxx % 1000 == 0:
                print(key, " ".join([f"{k}:{v}" for k, v in _dict.items()]))

        return mu_loss
        # return skewness_loss + mu_loss
        # return cov_loss + mu_loss

    def on_test_end(
        self,
    ):
        if hasattr(self, "data"):
            save_path = find_unsame_name_for_file(self.trainer.log_dir + "/data.csv")
            # self.data.to_csv(save_path, index=False)
            del self.data

    def ttt_step(self, batch, batch_idx):
        batch["audio"] = batch["audio"].clone().requires_grad_()
        batch_size = batch["audio"].shape[0]
        # audio_aug = self.ttt_transform(batch["audio"].clone()).requires_grad_()

        base_label = torch.ones(batch_size, dtype=batch["vocoder_label"].dtype).to(
            batch["audio"].device
        )
        # batch['audio'], speed_labels = random_speed.batch_apply(batch['audio'])

        _input = batch["audio"]
        # _input = torch.concat([batch["audio"], audio_aug], dim=0)
        x = self.model.feature_model.preprocess(_input)
        _h = self.model.feature_model.get_hidden_state(x)
        _c = self.model.feature_model.get_final_feature(_h)
        # _content_logit = self.model.cls_content(_c).squeeze(-1)
        _v = self.model.feature_model.get_final_feature_copyed(_h)
        _logit = self.model.cls_final(
            torch.concat(
                [
                    _c,
                    self.vocoder_feats["fake"].repeat(batch_size, 1),
                ],
                dim=-1,
            )
        ).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(_logit, base_label * 0.0)

        # loss += self.Euclidean_distance(_c[:batch_size, ...], _c[batch_size:,...]) * 10
        # loss += self.loss_adjustment(
        #     main_loss=loss, auxiliary_loss=self.Entropy_loss(_logit), sigma=0.5
        # )

        _h = _h[:batch_size, ...]
        ### feature loss
        _feat_loss = self.feat_loss(key="feat", feat=_h, cls_loss=loss)
        _feat_loss = self.loss_adjustment(
            main_loss=loss, auxiliary_loss=_feat_loss, sigma=0.5
        )
        loss += _feat_loss

        # _content_feat_loss = self.feat_loss(key='content_feat', feat=_v)
        # _content_feat_loss = self.loss_adjustment(main_loss=loss, auxiliary_loss=_content_feat_loss)
        # loss += _content_feat_loss

        # _voc_feat_loss = self.feat_loss(key='voc_feat', feat=_v, cls_loss=loss)
        # _voc_feat_loss = self.loss_adjustment(main_loss=loss, auxiliary_loss=_voc_feat_loss)
        # loss += _voc_feat_loss

        # speed loss
        # base = torch.ones(
        #     batch["audio"].shape[0], dtype=batch["vocoder_label"].dtype
        # ).to(batch["audio"].device) * 5
        # loss = self.ce_loss(batch_res["speed_logit"], base)

        return loss

    # @torch.enable_grad()
    # @torch.inference_mode(False)
    # def test_step(self, batch, batch_idx):
    #     # h1 = self.hash_module(self.model)
    #     torch.backends.cudnn.enabled = False

    #     loss = self.ttt_step(batch, batch_idx)
    #     self.backup_state_dict()

    #     opt = self.get_ttt_optimizer()
    #     opt.zero_grad()
    #     self.manual_backward(loss, retain_graph=False)
    #     opt.step()

    #     # h2 = self.hash_module(self.model)
    #     # print(h1, h2)

    #     with torch.no_grad():
    #         batch_res = self._shared_eval_step(
    #             batch, batch_idx, stage="test", loss=False
    #         )
    #     # print(batch_res['logit'])
    #     self.restore_state_dict()
    #     return batch_res


# +
# x = torch.randn(3, 10)
# torch.std(x, dim=1)
