# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: audio_df
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List, Optional, Tuple, Union

try:
    from .modules import Model_1D, Model_2D, Model_Text
except ImportError:
    from modules import Model_1D, Model_2D, Model_Text


class MultiView3DModel(nn.Module):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.args = args

        self.model_1d = Model_1D()
        self.model_2d = Model_2D()
        self.model_text = Model_Text()

        self.dropouts = nn.ModuleList([nn.Dropout(p=0.05) for _ in range(3)])

        self.cls_head1 = nn.Linear(768, 1)
        self.cls_head2 = nn.Linear(768, 1)
        self.cls_head_text = nn.Linear(768, 1)

        self.classifier = nn.Linear(768, 1)

    def feat_norm(self, x: torch.Tensor):
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-9)
        return x

    def random_dropout_features(
        self, view_features: list[torch.Tensor], p: float
    ) -> list[torch.Tensor]:
        """
        Randomly drops out features from a list of view features based on a 
        probability `p`. Dropped features are replaced with zero tensors of the same shape.

        Args:
            view_features (list[torch.Tensor]): A list of feature tensors, where
                                                each tensor represents a view.
            p (float): The probability of dropping out a feature. Must be in the
                        range [0, 1).

        Returns:
            list[torch.Tensor]: A list of feature tensors after applying random
                                dropout. At least one feature will remain.

        Example:
            >>> view_features = [torch.rand(3, 4), torch.rand(3, 4), torch.rand(3, 4)]
            >>> p = 0.5
            >>> dropped_features = random_dropout_features(view_features, p)
            >>> print(dropped_features)
            [tensor(...), tensor(...)]  # At least one feature remains
        """
        if not self.training:
            return view_features
        
        if not (0 <= p < 1):
            raise ValueError("Probability `p` must be in the range [0, 1).")

        n = len(view_features)
        if n == 0:
            return view_features

        # Generate a mask for dropout
        dropout_mask = torch.rand(n) < p
        # Ensure at least one feature remains
        if dropout_mask.sum() >= n:
            # If all features are marked for dropout, randomly keep one
            keep_idx = torch.randint(0, n, (1,))
            dropout_mask[keep_idx] = False

        # Apply the dropout mask and replace dropped features with zeros
        dropped_features = []
        for feature, mask in zip(view_features, dropout_mask):
            if mask:
                # Replace dropped feature with a zero tensor of the same shape
                dropped_features.append(torch.zeros_like(feature))
            else:
                # Keep the feature
                dropped_features.append(feature)

        return dropped_features

    def forward(self, x: torch.Tensor):

        x_1d = self.model_1d(x)
        x_2d = self.model_2d(x)
        x_2d = rearrange(x_2d, "b c h w -> b c (h w)")
        x_2d = F.adaptive_avg_pool1d(x_2d, x_1d.shape[-1])
        x_text = self.model_text(x)

        # print(x_1d.shape, x_2d.shape, x_text.shape)

        # x_1d = self.dropouts[0](self.feat_norm(x_1d))
        # x_2d = self.dropouts[1](self.feat_norm(x_2d))
        # x_text = self.dropouts[2](self.feat_norm(x_text))
        [x_1d,x_2d,x_text] = self.random_dropout_features([x_1d,x_2d,x_text], p=0.1)

        res = {}

        res["logit1"] = self.cls_head1(x_1d.mean(dim=-1)).squeeze(-1)
        res["logit2"] = self.cls_head2(x_2d.mean(dim=-1)).squeeze(-1)
        res["logit_text"] = self.cls_head_text(x_text.mean(dim=-1)).squeeze(-1)

        feat = (x_1d + x_2d + x_text) / 3
        feat = feat.mean(dim=-1)
        res["logit"] = self.classifier(feat).squeeze(-1)

        return res


# + tags=["active-ipynb"]
# model = MultiView3DModel()
# x = torch.randn(2, 16_000*4)
# model(x)
