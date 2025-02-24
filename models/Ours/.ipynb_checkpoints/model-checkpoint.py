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

import math
import random
from copy import deepcopy
from functools import partial
from einops import rearrange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from ay2.torch.nn import LambdaFunctionModule

from torchvision.transforms import v2

# + editable=true slideshow={"slide_type": ""}
try:
    from .feature_extractor import LCNN, MSFM, RawNet2, ResNet
    from .utils import weight_init
except ImportError:
    from feature_extractor import LCNN, MSFM, RawNet2, ResNet
    from utils import weight_init

# from .gradient_reversal import GradientReversal
# from .modules.classifier import Classifier
# from .modules.feature_extractor import FeatureExtractor, FeatureExtractor2D
# from .modules.model_RawNet2 import LayerNorm, RawNet_FeatureExtractor, SincConv_fast


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# from gradient_reversal import GradientReversal
# from modules.classifier import Classifier
# from modules.feature_extractor import FeatureExtractor, FeatureExtractor2D
# from modules.model_RawNet2 import LayerNorm, RawNet_FeatureExtractor, SincConv_fast
# from utils import weight_init
# -

class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

    
    def forward(self, x, y):
        h, w = x.shape[2:4]
        short_cut = x
        x = rearrange(x, 'b c h w -> b (h w) c')
        y = rearrange(y, 'b c h w -> b (h w) c')
        x, _ = self.multihead_attn(y, x, x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h,w=w)
        return x + short_cut
        # return x


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
        args=None,
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

        self.feature_extractor = feature_extractor
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
                dims=dims, n_blocks=n_blocks, n_heads=n_heads, args=args, cfg=cfg
            )
            final_dim = dims[-1]

        self.feature_model.copy_final_stage()
        
        self.dropout = nn.Dropout(0.1)
        self.cls_content = nn.utils.weight_norm(nn.Linear(final_dim, 1, bias=False))
        if cfg.one_stem:
            self.content_based_cls = nn.utils.weight_norm(nn.Linear(final_dim, 1, bias=False))
        self.cls_voc = nn.utils.weight_norm(
            nn.Linear(final_dim, vocoder_classes + 1, bias=False)
        )

        
        self.cls_final = nn.Sequential(
            # nn.utils.weight_norm(nn.Linear(final_dim * 2, final_dim * 2, bias=False)),
            # nn.BatchNorm1d(final_dim * 2),
            # nn.ReLU(),
            nn.utils.weight_norm(nn.Linear(final_dim * 2, 1, bias=False)),
        )

        self.cls_speed = nn.utils.weight_norm(nn.Linear(final_dim, 16, bias=False))
        self.cls_compression = nn.utils.weight_norm(
            nn.Linear(final_dim, 10, bias=False)
        )

        self.debug = 0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.cross_attention1 = CrossAttention(512)
        self.cross_attention2 = CrossAttention(512)
        self.transform = v2.RandomErasing()

    def weight_init(self):
        """initialize all the weights

        If use this initialization, should call this funciton in the __init__ function.

        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
        self.apply(weight_init)

    def get_content_stream_modules(
        self,
    ):
        return self.feature_model.get_content_stem() + [self.cls_content]

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

    def ttt(self, x):
        res = {}
        res["hidden_states"] = self.feature_model.get_hidden_state(x)

    
    def feature_norm(self, code):
        code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
        code = torch.div(code, code_norm)
        return code
        
    def fuse_stem_featurs(self, feat1, feat2):
        ### feat1 and feat2 's dim is 2
        if feat1.ndim == 2:
            feat = torch.concat([feat1, feat2], dim=-1)
            return feat

        ### dim is 4
        feat1 = self.cross_attention1(feat1, feat2)
        feat2 = self.cross_attention2(feat2, feat1)
        feat = torch.concat([feat1, feat2], dim=1)
        feat = self.avgpool(feat)
        feat = feat.reshape(feat.size(0), -1)
        # feat = self.feature_norm(feat)
        return feat
    
    def forward(self, x, stage="test", batch=None, one_stem=False):
        batch_size = x.shape[0]
        res = {}

        res["hidden_states"] = self.feature_model.get_hidden_state(x)
        if self.feature_extractor == "ResNet":
            res["content_feature"], conv_feat1 = self.feature_model.get_final_feature(
                res["hidden_states"]
            )
        else:
            res["content_feature"] = self.feature_model.get_final_feature(
                res["hidden_states"]
            )

        if one_stem:
            res["content_based_cls_logit"] = self.content_based_cls(
                self.dropout(res["content_feature"])
            ).squeeze(-1)
        res["speed_logit"] = self.cls_speed(self.dropout(res["content_feature"]))
        res["compression_logit"] = self.cls_compression(
            self.dropout(res["content_feature"])
        )

        # learn a vocoder feature extractor and classifier

        hidden_states = res["hidden_states"]
        # if stage == 'train':
        #     B, C = res["hidden_states"].shape[0:2]
        #     L = 100
        #     feat_clone = res["hidden_states"].clone()
        #     shuffle_id = torch.randperm(B)
        #     s = np.random.randint(0, C - 100, 1)[0]
        #     feat_clone[:B//2, s:s+L] = 0.5 * feat_clone[:B//2, s:s+L] + 0.5 * res["hidden_states"][shuffle_id[:B//2], s:s+L]
        #     feat_clone[B//2:, s:s+L] = 0
        #     # print(res["hidden_states"].shape)
        #     hidden_states = feat_clone

        
        if self.feature_extractor == "ResNet":
            (
                res["vocoder_feature"],
                conv_feat2,
            ) = self.feature_model.get_final_feature_copyed(hidden_states)
        else:
            res["vocoder_feature"] = self.feature_model.get_final_feature_copyed(hidden_states)
            
        res["vocoder_logit"] = self.cls_voc(self.dropout(res["vocoder_feature"]))
        res["content_voc_logit"] = self.cls_voc(self.dropout(res["content_feature"]))

        
        # print(feature_aug, res["vocoder_feature"])
        voc_feat = res["vocoder_feature"]
        content_feat = res["content_feature"]
        if stage == 'train' and self.cfg.style_shuffle:
            shuffle_id = torch.randperm(batch_size)
            shuffle_id = get_permutationID_by_label(batch['label'])
            voc_feat =  exchange_mu_std(res["vocoder_feature"], res["vocoder_feature"][shuffle_id], dim=-1)
            content_feat =  exchange_mu_std(res["content_feature"], res["content_feature"][shuffle_id], dim=-1)
        res["feature"] = self.fuse_stem_featurs(res["content_feature"], res["vocoder_feature"])
        final_feat = self.fuse_stem_featurs(content_feat, voc_feat)
        res["logit"] = self.cls_final(self.dropout(final_feat)).squeeze(-1)

        
        
        if stage == "train" and self.cfg.feat_shuffle:
            shuffle_id = torch.randperm(batch_size)
            res["shuffle_logit"] = self.cls_final(
                self.dropout(
                    # self.fuse_stem_featurs(conv_feat1, conv_feat2[shuffle_id])
                    # self.fuse_stem_featurs(res["content_feature"], res["vocoder_feature"][shuffle_id])
                    self.fuse_stem_featurs(content_feat, voc_feat[shuffle_id])
                )
            ).squeeze(-1)
            batch["shuffle_label"] = deepcopy(batch["label"])
            for i in range(batch_size):
                if batch["label"][shuffle_id[i]] == 0 or batch["label"][i] == 0:
                    batch["shuffle_label"][i] = 0
                else:
                    batch["shuffle_label"][i] = 1

        if hasattr(self, "gradcam") and self.gradcam:
            logit = torch.sigmoid(res['logit'])[:, None] # (B, 1)
            logit = torch.concat([1-logit, logit], dim=-1) # (B, 2)
            return logit
        
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
# -

def get_permutationID_by_label(label):

    x = label.cpu()
    index0 = np.where(x == 0)[0]
    index1 = np.where(x == 1)[0]

    shuffle_index0 = np.random.permutation(index0)
    shuffle_index1 = np.random.permutation(index1)

    new_index = np.ones_like(x)
    for i in range(len(index0)):
        new_index[index0[i]] = shuffle_index0[i]
    for i in range(len(index1)):
        new_index[index1[i]] = shuffle_index1[i]  
    return new_index
    

def exchange_mu_std(x, y, dim=None):
    mu_x = torch.mean(x, dim=dim, keepdims=True)
    mu_y = torch.mean(y, dim=dim, keepdims=True)
    std_x = torch.std(x, dim=dim, keepdims=True)
    std_y = torch.std(y, dim=dim, keepdims=True)

    alpha = np.random.randint(50, 100) / 100
    target_mu = alpha * mu_x + (1-alpha) * mu_y 
    target_std = alpha * std_x + (1-alpha) * std_y
    z = target_std * ( (x - mu_x) / std_x) + target_mu

    noise_level = 20
    add_noise_level=np.random.randint(0, noise_level) / 100
    mult_noise_level=np.random.randint(0, noise_level) / 100
    z = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level)
    
    return z


def _noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    if add_noise_level > 0.0:
        add_noise = add_noise_level * np.random.beta(2, 5) * torch.FloatTensor(x.shape).normal_().to(x.device)
    if mult_noise_level > 0.0:
        mult_noise = mult_noise_level * np.random.beta(2, 5) * (2*torch.FloatTensor(x.shape).uniform_()-1).to(x.device) + 1 
    return mult_noise * x + add_noise
