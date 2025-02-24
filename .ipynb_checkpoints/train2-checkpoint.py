# ---
# jupyter:
#   jupytext:
#     formats: py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
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
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.95)
    # parser.add_argument("--specaug", type=str, default='ss')
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("--profiler", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--earlystop", type=int, default=3)
    parser.add_argument("-v", "--version", type=int, default=None)
    parser.add_argument("-t", "--test", type=int, default=0)
    parser.add_argument("-l", "--log", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--theme", type=str, default='best')
    parser.add_argument("--collect", type=int, default=0)
    args = parser.parse_args()

    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=args.ablation
    )
    # cfg.MODEL.dims = eval(args.dims)
    # cfg.MODEL.n_blocks = eval(args.nblocks)
    if args.batch_size > 0:
        cfg.DATASET.batch_size = args.batch_size
    ds, dl = make_data(cfg.DATASET)

    # print(str(dict(cfg)))

    callbacks = make_callbacks(args, cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.MODEL.epochs if not args.profiler else 2,
        # max_epochs=3,
        accelerator="gpu",
        devices=args.gpu,
        logger=build_logger(args, ROOT_DIR),
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        default_root_dir=ROOT_DIR,
        strategy='ddp_find_unused_parameters_true' if len(args.gpu) > 1 else 'auto'
        # profiler=pl.profilers.SimpleProfiler(dirpath='./', filename='test'),
        # profiler=pl.profilers.AdvancedProfiler(dirpath='./', filename='test'),
        # limit_train_batches=1.0 if not args.profiler else 100,
        # limit_val_batches=100,
        # limit_test_batches =10,
    )


    profiler = pl.profilers.PassThroughProfiler() if not args.profiler else pl.profilers.SimpleProfiler(dirpath='./', filename='test')
    trainer.profiler = profiler
    model = make_model(args.cfg, cfg, args)
    
    color_print(f"logger path : {trainer.logger.log_dir}")
    log_dir = trainer.logger.log_dir

    
    if not args.test:
        ckpt_path = get_ckpt_path(log_dir, theme="last") if args.resume else None

        if 'Lib' in args.cfg:
            val_dl = to_list(dl.test)[2]
        else:
            val_dl = to_list(dl.test)[1]
        # val_dl = dl.val
        trainer.fit(model, dl.train, val_dataloaders=val_dl, ckpt_path=ckpt_path)
        
        write_model_summary(model, log_dir)

    else:
        # backup_logger_file(log_dir)
        clear_old_test_file(log_dir)
        ckpt_path = get_ckpt_path(log_dir, theme=args.theme)
        trainer.trainset_wo_transform = dl.train_wo_transform

        if not args.collect:
            for test_dl in to_list(dl.test):
                trainer.test(model, test_dl, ckpt_path=ckpt_path)
        else:
            for test_dl in to_list(dl.test) + [dl.val]:
                trainer.test(model, test_dl, ckpt_path=ckpt_path)
