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

# + editable=true slideshow={"slide_type": ""}
# %load_ext autoreload
# %autoreload 2

# +
import argparse
import os
import random
import sys

import numpy as np
import torch


# + editable=true slideshow={"slide_type": ""}
def make_model(cfg_file, cfg, args):
    """build models from cfg file name and model cfg

    Args:
        cfg_file: the file name of the model cfg, such as "LCNN/wavefake"
        cfg: the model config

    """
    if cfg_file.startswith("LCNN/"):
        from .LFCC_LCNN import LCNN_lit

        model = LCNN_lit()
    elif cfg_file.startswith("RawNet2/"):
        from .RawNet import RawNet2_lit

        model = RawNet2_lit()
    elif cfg_file.startswith("WaveLM/"):
        from .WaveLM import WaveLM_lit

        model = WaveLM_lit()
    elif cfg_file.startswith("Wave2Vec2"):
        from .Wave2Vec2 import Wav2Vec2_lit

        model = Wav2Vec2_lit()
    elif cfg_file.startswith("LibriSeVoc"):
        from .LibriSeVoc import LibriSeVoc_lit

        model = LibriSeVoc_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("Ours/"):
        from .Ours import AudioModel_lit

        model = AudioModel_lit(cfg=cfg.MODEL, args=args)
    elif cfg_file.startswith("Wav2Clip/"):
        from .Wav2Clip import Wav2Clip_lit

        model = Wav2Clip_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("AudioClip/"):
        from .AudioClip import AudioClip_lit

        model = AudioClip_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("AASIST/"):
        from .Aaasist import AASIST_lit

        model = AASIST_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("RawGAT/"):
        from .RawGAT_ST import RawGAT_lit

        model = RawGAT_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("ASDG/"):
        from .ASDG import ASDG_lit

        model = ASDG_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("ABCNet/"):
        from .Audio_models.ABCNet import ABCNet_lit

        model = ABCNet_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("MPE_LCNN/"):
        from .Audio_models.MPE_LCNN import MPE_LCNN_lit

        model = MPE_LCNN_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("SFATNet/"):
        from .Audio_models.SFAT_Net import SFATNet_lit

        model = SFATNet_lit(cfg=cfg.MODEL)
        
    elif cfg_file.startswith("MultiViewCombine/"):
        from .OurModels.MultiViewCombine import MultiViewCombine_lit

        model = MultiViewCombine_lit(cfg=cfg.MODEL.MultiViewCombine, args=args)
    elif cfg_file.startswith("OursMultiView/"):
        from .OursMultiView import MultiViewModel_lit

        model = MultiViewModel_lit(cfg=cfg.MODEL, args=args)
    elif cfg_file.startswith("OursMultiView2/"):
        from .OursMultiView2 import MultiViewModel_lit

        model = MultiViewModel_lit(cfg=cfg.MODEL, args=args)
    elif cfg_file.startswith("OursMultiView3/"):
        from .OurModels.OursMultiView3 import MultiViewModel_lit

        model = MultiViewModel_lit(cfg=cfg.MODEL, args=args)

    elif cfg_file.startswith("OursCLIP/"):
        from .OursCLIP import OursCLIP_lit

        model = OursCLIP_lit(cfg=cfg.MODEL, args=args)
    elif cfg_file.startswith("OursLCNN/"):
        from .OurModels.OursLCNN import OursLCNN_lit

        model = OursLCNN_lit(cfg=cfg.MODEL, args=args)
    elif cfg_file.startswith("OursPhonemeGAT/"):
        from .OurModels.phoneme_GAT import Phoneme_GAT_lit

        model = Phoneme_GAT_lit(cfg=cfg.MODEL, args=args)

    return model


# -

def make_attack_model(cfg_file, cfg, args):
    from .RawNet import RawNet2_lit

    path = (
        "/mnt/data/zky/DATA/1-model_save/00-Deepfake/1-df-audio-old/RawNet2/DECRO_chinese"
        "/version_0/checkpoints/best-epoch=12-val-auc=0.9745.ckpt"
    )
    cls_model = RawNet2_lit()
    sd = torch.load(path)["state_dict"]
    cls_model.load_state_dict(sd)

    if cfg_file.startswith("Attack/Ours"):
        from .attacks.Ours import AudioAttackModel

        model = AudioAttackModel(cfg=cfg.MODEL, args=args, audio_detection_model=cls_model)
    return model
