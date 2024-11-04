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
import argparse
import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import warnings
 
warnings.filterwarnings("ignore")
# -

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

from ay2.tools import color_print, to_list

from config import get_cfg_defaults
from data.make_dataset import make_data
from models import make_model
from utils import (
    clear_folder,
    backup_logger_file,
    build_logger,
    clear_old_test_file,
    get_ckpt_path,
    make_callbacks,
    write_model_summary,
)

ROOT_DIR = "/home/ay/data/DATA/1-model_save/00-Deepfake/1-df-audio"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--dims", type=str, default='[32, 64, 64, 128]')
    parser.add_argument("--nblocks", type=str, default='[1,1,3,1]')
    parser.add_argument("--ablation", type=str, default=None)
    
    ## multiview model
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.95)

    # parser.add_argument("--specaug", type=str, default='ss')
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--grad", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--earlystop", type=int, default=3)
    parser.add_argument("--min_epoch", type=int, default=1)
    parser.add_argument("--use_profiler", type=int, default=0)
    parser.add_argument("--use_lr_find", type=int, default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    parser.add_argument("-t", "--test", type=int, default=0)
    parser.add_argument("-l", "--log", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--theme", type=str, default='best')
    parser.add_argument("--collect", type=int, default=0)
    parser.add_argument("--clear_log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_as_val", type=int, default=999)
    parser.add_argument("--test_noise", type=int, default=0)
    parser.add_argument("--test_noise_level", type=int, default=30)
    parser.add_argument("--test_noise_type", type=str, default='bg')
    args = parser.parse_args()


    if args.seed != 42:
        pl.seed_everything(args.seed)
        
    
    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=args.ablation
    )
    # cfg.MODEL.dims = eval(args.dims)
    # cfg.MODEL.n_blocks = eval(args.nblocks)
    if args.batch_size > 0:
        cfg.DATASET.batch_size = args.batch_size
    ds, dl = make_data(cfg.DATASET, args=args)


    print(ds.train.data.groupby(['method']).count())
    print(ds.val.data.groupby(['method']).count())
    print(ds.test.data.groupby(['method']).count())
