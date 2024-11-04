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
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from ay2.torch.nn import LambdaFunctionModule

# + editable=true slideshow={"slide_type": ""}
from .gradient_reversal import GradientReversal
from .modules.classifier import Classifier
from .modules.feature_extractor import FeatureExtractor, FeatureExtractor2D
from .modules.model_RawNet2 import LayerNorm, RawNet_FeatureExtractor, SincConv_fast
from .utils import weight_init

from .feature_extractor import LCNN


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

        self.feature_model = LCNN()
        self.feature_model.copy_final_stage()
        
        
        # self.voc_stage = deepcopy(self.feature_model.stages[-1])

        self.dropout = nn.Dropout(0.1)
        self.cls_content = nn.Linear(dims[-1], 1, bias=False)
        self.cls_voc = nn.Linear(dims[-1], vocoder_classes + 1, bias=False)
        self.cls_final = nn.Linear(dims[-1] * 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

        # self.apply(weight_init)

    
    def preprocess(self, x, stage='test'):
        x = self.spectrogram(x)
        x = torch.log(x + 1e-7)
     
        
        if self.debug:
            print('log-scale feature : ', x.shape)
        
        # if stage=='train':
        #     x = self.transform(x)

        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )
        return x

    
    
    def forward(self, x, stage="test"):
        batch_size = x.shape[0]
        res = {}

        x = self.preprocess(x)
        
        stage_id = len(self.dims) - 2
        hidden_state = self.feature_model.get_hidden_state(x, stage_id=stage_id, stage=stage)

        
        res["content_feature"] = self.feature_model.get_final_feature(
            hidden_state, stage_id=stage_id
        )
        res["vocoder_feature"] = self.feature_model.pool_reshape(
            self.voc_stage(hidden_state)
        )

        # print(res["content_feature"].shape)
        
        res["content_logit"] = self.cls_content(
            self.dropout(res["content_feature"])
        ).squeeze()
        res["content_voc_logit"] = self.cls_voc(self.dropout(res["content_feature"]))

        
        res["vocoder_logit"] = self.cls_voc(self.dropout(res["vocoder_feature"]))

        
        res["logit"] = self.cls_final(
            self.dropout(
                torch.concat([res["content_feature"], res["vocoder_feature"]], dim=-1)
            )
        ).squeeze()
        res["aug_logit"] = self.cls_final(
            self.dropout(
                torch.concat(
                    [
                        res["content_feature"],
                        res["vocoder_feature"][torch.randperm(batch_size)],
                    ],
                    dim=-1,
                )
            )
        )

        return res

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# model = AudioModel(vocoder_classes=7)
# x = torch.randn(32, 1, 48000)
# _ = model(x)

# + tags=["active-ipynb"]
# ckpt = torch.load(
#     "/home/ay/data/DATA/1-model_save/0-Audio/Ours/LibriSeVoc_cross_dataset/version_7/checkpoints/best-epoch=3-val-auc=0.99.ckpt"
# )
#
# state_dict = ckpt["state_dict"]
#
# state_dict2 = {key.replace("model.", "", 1): state_dict[key] for key in state_dict}
#
# model.load_state_dict(state_dict2)
