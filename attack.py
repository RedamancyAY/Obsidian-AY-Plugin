# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# + tags=[]
import argparse
import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import torch
import warnings
 
warnings.filterwarnings("ignore")

# + tags=[]
pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

# + tags=[]
from ay2.tools import color_print, to_list

# + tags=[]
from config import get_cfg_defaults
from data.make_dataset import make_data
from models import make_attack_model
from utils import (
    build_logger,
    clear_old_test_file,
    get_ckpt_path,
    make_attack_callbacks,
    write_model_summary,
)
# -

ROOT_DIR = "/home/ay/data/DATA/1-model_save/00-Deepfake/1-df-audio"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--dims", type=str, default='[32, 64, 64, 128]')
    parser.add_argument("--nblocks", type=str, default='[1,1,3,1]')
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--specaug", type=str, default='ss')
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--earlystop", type=int, default=3)
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--pretrain_CS", type=int, default=0)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--collect", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--theme", type=str, default='best')
    args = parser.parse_args()

    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=args.ablation
    )
    cfg.MODEL.dims = eval(args.dims)
    cfg.MODEL.n_blocks = eval(args.nblocks)
    ds, dl = make_data(cfg.DATASET)

    model = make_attack_model(args.cfg, cfg, args)
    callbacks = make_attack_callbacks(args, cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.MODEL.epochs,
        # max_epochs=1,
        accelerator="gpu",
        devices=args.gpu,
        logger=build_logger(args, ROOT_DIR),
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        default_root_dir=ROOT_DIR,
        # profiler=pl.profilers.SimpleProfiler(dirpath='./', filename='test'),
        # limit_train_batches=200,
        # limit_val_batches=100,
        # limit_test_batches =10,
    )

    color_print(f"logger path : {trainer.logger.log_dir}")
    log_dir = trainer.logger.log_dir

    if not args.test:
        if args.resume:
            ckpt_path = get_ckpt_path(log_dir, theme="last")
            trainer.fit(model, dl.train, val_dataloaders=dl.val, ckpt_path=ckpt_path)
        else:
            # val_dl = dl.test[-1]
            val_dl = dl.val
            trainer.fit(model, dl.train, val_dataloaders=val_dl)
        write_model_summary(model, log_dir)

    else:
        # backup_logger_file(log_dir)
        clear_old_test_file(log_dir)
        ckpt_path = get_ckpt_path(log_dir, theme=args.theme)
        trainer.trainset_wo_transform = dl.train_wo_transform
        for test_dl in to_list(dl.test):
            trainer.test(model, test_dl, ckpt_path=ckpt_path)
