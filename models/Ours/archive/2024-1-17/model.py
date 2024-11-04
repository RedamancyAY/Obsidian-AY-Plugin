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
import math
import random
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from ay2.torch.nn import LambdaFunctionModule
# -

from torchvision.transforms import v2

# + editable=true slideshow={"slide_type": ""}
from .feature_extractor import LCNN, MSFM, RawNet2, ResNet
# from .gradient_reversal import GradientReversal
# from .modules.classifier import Classifier
# from .modules.feature_extractor import FeatureExtractor, FeatureExtractor2D
# from .modules.model_RawNet2 import LayerNorm, RawNet_FeatureExtractor, SincConv_fast
from .utils import weight_init


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# from gradient_reversal import GradientReversal
# from modules.classifier import Classifier
# from modules.feature_extractor import FeatureExtractor, FeatureExtractor2D
# from modules.model_RawNet2 import LayerNorm, RawNet_FeatureExtractor, SincConv_fast
# from utils import weight_init

# + editable=true slideshow={"slide_type": ""}
class AudioModel(nn.Module):
    def __init__(
        self,
        feature_extractor: str,
        dims=[32, 64, 64, 64, 128],
        n_blocks=[1, 1, 1, 2, 1],
        n_heads=[1, 2, 2, 4, 1, 1],
        samples_per_frame=640,
        gru_node=128,
        gru_layers=3,
        fc_node=128,
        num_classes=1,
        vocoder_classes=8,
        adv_vocoder=False,
        cfg=None,
        args=None
    ):
        super().__init__()

        self.cfg = cfg

        # self.norm = LayerNorm(48000)
        self.dims = dims
        # self.feature_model = FeatureExtractor2D(
        #     dims=dims,
        #     n_blocks=n_blocks,
        #     n_heads=n_heads,
        #     samples_per_frame=samples_per_frame,
        #     use_gru_head=False,
        #     gru_node=gru_node,
        #     gru_layers=gru_layers,
        # )

        if feature_extractor == "LCNN":
            self.feature_model = LCNN()
            final_dim = 64
        elif feature_extractor == "RawNet":
            self.feature_model = RawNet2()
            final_dim = 1024
        elif feature_extractor == "ResNet":
            self.feature_model = ResNet()
            final_dim = 512
        elif feature_extractor == "MSFM":
            self.feature_model = MSFM(
                dims=dims,
                n_blocks=n_blocks,
                n_heads=n_heads,
                args=args
            )
            final_dim = dims[-1]

        self.feature_model.copy_final_stage()

        self.dropout = nn.Dropout(0.1)
        self.cls_content = nn.Linear(final_dim, 1, bias=False)
        self.cls_voc = nn.Linear(final_dim, vocoder_classes + 1, bias=False)
        self.cls_final = nn.Sequential(
            nn.Linear(final_dim * 2, final_dim * 2, bias=False),
            nn.BatchNorm1d(final_dim * 2),
            nn.ReLU(),
            nn.Linear(final_dim * 2, 1, bias=False)
        )

        self.debug = 0

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.normal_(m.weight, mean=1, std=0.02)
        #         nn.init.constant_(m.bias, 0)

        # self.apply(weight_init)


        self.transform = v2.RandomErasing()


    def module_similaryity(self):
        loss = []
        for p1, p2 in zip(
                self.feature_model.get_final_block_parameters(),
                self.feature_model.get_copied_final_block_parameters(),
            ):
            _loss = 1 - F.cosine_similarity(p1.view(1, -1), p2.view(1, -1))[0]
            loss.append(_loss)
        loss = sum(loss) / len(loss)
        return loss

    def forward(self, x, stage="test", batch=None):
        batch_size = x.shape[0]
        res = {}

        x = self.feature_model.preprocess(x, stage=stage)

        # if stage == 'train':
        #     x = self.transform(x)
        
        res["hidden_states"] = self.feature_model.get_hidden_state(x)
        res["content_feature"] = self.feature_model.get_final_feature(
            res["hidden_states"]
        )
        res["content_logit"] = self.cls_content(
            self.dropout(res["content_feature"])
        ).squeeze()
        

        # learn a vocoder feature extractor and classifier
        res["vocoder_feature"] = self.feature_model.get_final_feature_copyed(
            res["hidden_states"].detach()
        )
        res["vocoder_logit"] = self.cls_voc(self.dropout(res["vocoder_feature"]))


        res["content_voc_logit"] = self.cls_voc(self.dropout(res["content_feature"]))


        # res["logit"] = self.cls_final(
        #     self.dropout(
        #         torch.concat([res["content_feature"], res["vocoder_feature"]], dim=-1)
        #     )
        # ).squeeze()
        # res["aug_logit"] = self.cls_final(
        #     self.dropout(
        #         torch.concat(
        #             [
        #                 res["content_feature"],
        #                 res["vocoder_feature"][torch.randperm(batch_size)],
        #             ],
        #             dim=-1,
        #         )
        #     )
        # )
        if batch is not None:
            vocoder_label = batch["vocoder_label"]
            w_tensor = self.cls_voc.weight
            logits_vocoder = res["vocoder_logit"]
            perturbation = 1
            epsilon = 1e-5
            grad_aug = -1 * w_tensor[vocoder_label] + torch.matmul(
                logits_vocoder.detach(), w_tensor
            )
            FGSM_attack = perturbation * (
                grad_aug.detach()
                / (grad_aug.detach().norm(2, dim=1, keepdim=True) + epsilon)
            )
            # print(grad_aug.shape, FGSM_attack.shape, res["vocoder_feature"].shape)
            ratio = random.random()
            # ratio = 1.0
            feature_aug = ratio * FGSM_attack
        else:
            ratio = 0
            feature_aug = 0.0

        # print(feature_aug, res["vocoder_feature"])
        res["logit"] = self.cls_final(
            self.dropout(
                torch.concat(
                    [
                        res["content_feature"],
                        res["vocoder_feature"] + feature_aug,
                    ],
                    dim=-1,
                )
            )
        ).squeeze()

        return res

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = AudioModel(vocoder_classes=7)
# x = torch.randn(32, 1, 48000)
# _ = model(x)

# + tags=["active-ipynb"]
# # ckpt = torch.load(
# #     "/home/ay/data/DATA/1-model_save/0-Audio/Ours/LibriSeVoc_cross_dataset/version_7/checkpoints/best-epoch=3-val-auc=0.99.ckpt"
# # )
#
# # state_dict = ckpt["state_dict"]
#
# # state_dict2 = {key.replace("model.", "", 1): state_dict[key] for key in state_dict}
#
# # model.load_state_dict(state_dict2)
