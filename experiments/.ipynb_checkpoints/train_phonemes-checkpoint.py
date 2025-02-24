# %load_ext autoreload
# %autoreload 2

# +
import sys

sys.path.append("../")

# +
import os
import random
import pytorch_lightning as pl

import pandas as pd
import torch
from ay2.datasets.audio import Partial_CommonVoice_AudioDs
from ay2.tools.text import Phonemer_Tokenizer_Recombination
# -

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")
# torch.backends.cudnn.benchmark = True

# # Step 1: load dataset

# +
root_path = "/home/ay/data/0-原始数据集/common_voice_11_0"
languages = ["en", "es", "de"]

csv_path = os.path.join(root_path, "dataset_info.csv")
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path, low_memory=False)
    data["phoneme_id"] = data["phoneme_id"].apply(lambda x: eval(x))
else:
    ds = Partial_CommonVoice_AudioDs(root_path=root_path)
    data = ds.load_metadata_for_multiple_language(languages=languages, is_generate_phoneme=True)
    data.to_csv(csv_path, index=False)

data = data.query("audio_len < 16000*10").reset_index(drop=True)
data = data.query("phoneme_id_length < 120").reset_index(drop=True)
data = data.query("(audio_len // 320 - 1) > phoneme_id_length").reset_index(drop=True)
# data = data.query("locale == 'en'").reset_index(drop=True)

# +
# audio_length = sorted(data['audio_len'])
# audio_length[3000 * 16]

# +
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate, random_split

try:
    from .data.tools import WaveDataset
except ImportError:
    from data.tools import WaveDataset
# -

_ds = WaveDataset(
    data,
    sample_rate=16000,
    normalize=True,
    transform=None,
    dtype="tensor",
    read_features=["phoneme", "phoneme_id", "phoneme_id_length", "audio_path", "language"],
)

generator1 = torch.Generator().manual_seed(42)
train_size = int(0.9 * len(_ds))
val_size = len(_ds) - train_size
train_dataset, val_dataset = random_split(_ds, [train_size, val_size], generator=generator1)


# +
def collate_fn(batch):
    max_audio_length = max(item["audio"].size(1) for item in batch)  # audio shape : (1, L)
    max_phoneme_length = max(item["phoneme_id_length"] for item in batch)
    for item in batch:
        item["audio_length"] = item["audio"].size(1)
        item["audio"] = F.pad(item["audio"], (0, max_audio_length - item["audio"].size(1)))
        item["phoneme_id"] = item["phoneme_id"] + [0] * (max_phoneme_length - item["phoneme_id_length"])
        item["phoneme_id"] = torch.tensor(item["phoneme_id"])

    new_batch = default_collate(batch)
    return new_batch


class LengthSortedSampler(Sampler):
    def __init__(self, dataset):
        self.data_source = dataset
        self.sorted_indices = sorted(
            range(len(dataset.indices)), key=lambda idx: dataset.dataset.get_audio_length(dataset.indices[idx])
        )
        self.buckets = self.split_list(self.sorted_indices, segment_size=3000)

    def split_list(self, lst, segment_size):
        return [lst[i : i + segment_size] for i in range(0, len(lst), segment_size)]

    def __iter__(self):
        # Shuffle elements within each bucket
        for bucket in self.buckets:
            random.shuffle(bucket)

        # Flatten the list of buckets
        shuffled_indices = [idx for bucket in self.buckets for idx in bucket]
        return iter(shuffled_indices)

    def __len__(self):
        return len(self.data_source)


# -

dl_train = DataLoader(
    train_dataset,
    batch_size=16,
    collate_fn=collate_fn,
    shuffle=True,
    # sampler=LengthSortedSampler(train_dataset),
    num_workers=15,
)
dl_val = DataLoader(
    val_dataset,
    batch_size=8,
    collate_fn=collate_fn,
    shuffle=False,
    num_workers=15,
    sampler=LengthSortedSampler(val_dataset),
)

# +
# for x in dl_train:
#     print(x['audio_length'])
#     # break
# -

# ## Model

# +
## Tokenizer

vocab_path = "/home/ay/data/0-原始数据集/common_voice_11_0/vocab_phoneme"
tokenizer = Phonemer_Tokenizer_Recombination(
    vocab_files=[os.path.join(vocab_path, f"vocab-phoneme-{language}.json") for language in languages],
    languages=languages,
)
# -

from phoneme_model import BaseModule, network_param, optim_param

model = BaseModule(network_param, optim_param, tokenizer=tokenizer)

# +
# for i, x in enumerate(dataloader):
#     x["audio"] = x["audio"][:, 0, :]
#     model._get_outputs(x, 0)
#     print(i)
#     if i == 3:
#         break
# -

# # Trainer

from callback import LogMetricsCallback
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

version=None
ckpt_path = None

# +
# resume = 1
# version = 74
# ckpt_path = "/mnt/data/zky/DATA/1-model_save/01-phoneme/phoneme_recongition/version_0/checkpoints/last-v1.ckpt"
# model = BaseModule.load_from_checkpoint(
#     ckpt_path, network_param=network_param, optim_param=optim_param, tokenizer=tokenizer
# )

# +
ROOT_DIR = "/home/ay/data/DATA/1-model_save/01-phoneme"

ckpt_callback = ModelCheckpoint(
    dirpath=None,
    save_top_k=1,
    monitor="val-per",
    mode="min",
    save_last=False,
    filename="best-{epoch}-{val-per:.6f}",
    save_weights_only=True,
    verbose=True,
)
ckpt_callback_last = ModelCheckpoint(
    dirpath=None,
    save_last=True,
    filename="last",
    save_weights_only=False,
    verbose=False,
)
logger = pl.loggers.CSVLogger(
    ROOT_DIR,
    name="phoneme_recongition",
    version=version,
)

# +
# trainer = pl.Trainer(
#     logger=logger,
#     auto_lr_find=True,
#     accelerator="gpu",
#     # devices=[0],
#     devices=[0, 1, 2, 3],
#     strategy="ddp_find_unused_parameters_true",
#     accumulate_grad_batches=2,
# )
# trainer.tune(model, dl_train, val_dataloaders=dl_val)
# print(model.lr, '!!!!!!!!!!!!!')

# +
trainer = pl.Trainer(
    logger=logger,
    callbacks=[
        ckpt_callback,
        ckpt_callback_last,
        LogMetricsCallback(),
        LearningRateMonitor(logging_interval="step"),
    ],
    # callbacks=[ckpt_callback],
    accelerator="gpu",
    # devices=[0],
    devices=[0, 1, 2, 3],
    max_epochs=50,  # number of epochs
    # log_every_n_steps=1,
    accumulate_grad_batches=2,
    gradient_clip_val=1.0,
    strategy="ddp_find_unused_parameters_true",
    precision=16
    # limit_train_batches=300,
)


trainer.fit(model, dl_train, val_dataloaders=dl_val,ckpt_path=ckpt_path)

# +
# from lightning.pytorch.tuner import Tuner

# tuner = Tuner(trainer)


# lr_finder = tuner.lr_find(
#     model,
#     train_dataloaders=dl_train,  # the training data
#     min_lr=0.000001,  # minimum learning rate
#     max_lr=0.01,  # maximum learning rate)
#     num_training=3000,
# )
# # Results can be found in
# print(lr_finder.results)
