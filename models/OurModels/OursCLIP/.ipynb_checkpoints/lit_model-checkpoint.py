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
import glob
import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision as tv
from ay2.torch.deepfake_detection import DeepfakeAudioClassification
from IPython.display import Audio, display
from PIL import Image
# -

from typing import List, Optional, Tuple, Union

# + editable=true slideshow={"slide_type": ""}
try:
    from .model import AudioCLIP
except ImportError:
    from model import AudioCLIP


# -

# ## 测试

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# aclp = AudioCLIP(pretrained="/home/ay/data/DATA/0-model_weights/AudioClip/AudioCLIP-Full-Training.pt")
#
# audio = torch.randn(2, 48000)
# image = torch.randn(2, 3, 224, 224)
# ((audio_features, image_features, _), _), _ = aclp(audio=audio, image=image)
# audio_features, audio_features.shape
# -

# ## Lit model

# + editable=true slideshow={"slide_type": ""}
class AudioClip(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = AudioCLIP(pretrained="/home/ay/data/DATA/0-model_weights/AudioClip/AudioCLIP-Full-Training.pt")
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
        self.resize_transform = tv.transforms.Resize(224, antialias=True)
        self.proj = nn.Linear(1024 * 2, 1)

        self.cls_situation1 = nn.Linear(1024, 309)
        self.cls_situation2 = nn.Linear(1024, 309)

    def forward(self, x):
        audio_features, image_features, logits_audio_image = self.extract_feature(x)
        y = self.make_prediction(audio_features, image_features)
        return y

    def extract_spec(self, x, stage="test"):
        x = self.spectrogram(x)
        x = torch.log(x + 1e-7)
        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9)
        return x

    def extract_feature(self, x):
        spec = self.extract_spec(x)
        spec = self.resize_transform(spec)
        spec = torch.concat([spec, spec, spec], dim=1)
        ((audio_features, image_features, _), logits), _ = self.model(audio=x, image=spec)
        logits_audio_image, _, _ = logits
        return audio_features, image_features, logits_audio_image

    def make_prediction(self, audio_features, image_features):
        feat = torch.concat([audio_features, image_features], dim=-1)
        y = self.proj(feat)
        return y


# -

model = AudioClip()
x = torch.randn(2, 1, 48000)
model(x)


# + editable=true slideshow={"slide_type": ""}
class OursCLIP_lit(DeepfakeAudioClassification):
    def __init__(self, backend="linear", cfg=None, args=None, **kwargs):
        super().__init__()
        self.args = args
        self.cfg = cfg

        self.model = AudioClip()

        sd = torch.load(
            "/home/ay/data/DATA/1-model_save/00-Deepfake/1-df-audio/OursCLIP/VGGSound/version_1/checkpoints/best-epoch=5-val-loss=35.758720.ckpt"
        )
        self.load_state_dict(sd["state_dict"])

        self.configure_loss_fn()
        self.save_hyperparameters()

    def configure_loss_fn(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def pretrain_loss(self, batch_res, batch):
        label = batch["label"]
        clip_loss = self.clip_loss(batch_res["logits_audio_spec"])

        cls_loss_audio = self.ce_loss(self.model.cls_situation1(batch_res["audio_feat"]), batch["label"])
        cls_loss_spec = self.ce_loss(self.model.cls_situation2(batch_res["spec_feat"]), batch["label"])
        loss = clip_loss + 0.5 * (cls_loss_audio + cls_loss_spec)
        losses = {
            "loss": loss,
            "clip_loss": clip_loss,
            "cls_loss_audio": cls_loss_audio,
            "cls_loss_spec": cls_loss_spec,
        }
        return losses

    def calcuate_loss(self, batch_res, batch):
        if "VGGSound" in self.args.cfg:
            return self.pretrain_loss(batch_res, batch)

        label = batch["label"]
        cls_loss = self.bce_loss(batch_res["logit"], label.type(torch.float32))
        clip_loss = self.clip_loss(batch_res["logits_audio_spec"])

        if self.trainer.current_epoch < -2:
            loss = clip_loss
        else:
            loss = cls_loss + 0.5 * clip_loss

        losses = {"loss": loss, "cls_loss": cls_loss, "clip_loss": clip_loss}
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        audio_features, image_features, logits_audio_image = self.model.extract_feature(audio)
        feat = torch.concat([audio_features, image_features], dim=-1)
        out = self.model.make_prediction(audio_features, image_features)

        out = out.squeeze(-1)
        batch_pred = (torch.sigmoid(out) + 0.5).int()
        return {
            "logit": out,
            "pred": batch_pred,
            "feature": feat,
            "audio_feat": audio_features,
            "spec_feat": image_features,
            "logits_audio_spec": logits_audio_image,
        }

    def clip_loss(self, logits, sample_weights=None) -> Optional[torch.Tensor]:
        batch_size = logits.shape[0]

        device = logits.device

        reference = torch.arange(batch_size, dtype=torch.int64, device=device)

        loss = torch.tensor(0.0, dtype=self.dtype, device=device)

        num_modalities: int = 1
        scale = torch.tensor(1.0, dtype=self.dtype, device=device)

        loss = F.cross_entropy(logits, reference, weight=sample_weights) + F.cross_entropy(
            logits.transpose(-1, -2), reference, weight=sample_weights
        )

        for idx in range(num_modalities):
            scale = scale * (idx + 1)

        return loss / scale
