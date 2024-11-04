# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

import pytorch_lightning as pl
import torch
import copy

from .litmodel import GMM, GMMTest



def flatten_dataset(
    dataset,
    amount: int = None,
) -> torch.Tensor:
    """Flatten a dataset for GMM-EM training.

    Note, that we additionally switch time and freq dimensions.

    Args:
        dataset (TransformDataset): The data set to flatten.
        amount (int): If supplied, only convert this amount of files.

    Returns:
        The flattened data (torch.Tensor).
    """
    new_data = []
    for i, (wav, _, _) in enumerate(dataset):
        new_data.append(wav)
        if amount is not None and i >= amount - 1:
            break
    new_data = torch.cat(new_data)
    print(new_data.shape)
    return new_data


def flatten_dataloader(
    dataloader,
    amount: int = None,
) -> torch.Tensor:
    """Flatten a dataset for GMM-EM training.

    Note, that we additionally switch time and freq dimensions.

    Args:
        dataset (TransformDataset): The data set to flatten.
        amount (int): If supplied, only convert this amount of files.

    Returns:
        The flattened data (torch.Tensor).
    """
    new_data = []
    i = 0
    for batch in dataloader:
        _data = batch['audio']
        new_data.append(_data)
        i += _data.shape[0]
        if amount is not None and i >= amount - 1:
            break
    new_data = torch.cat(new_data)
    print(new_data.shape)
    return new_data


def split_real_fake_dataset(dataset):

    def get_index(_data):
        real_data = data.query("label == 1")
        fake_data = data.query("label == 0")
        return list(real_data.index), list(fake_data.index), real_data, fake_data
        
    if isinstance(dataset, torch.utils.data.Dataset):
        data = dataset.data
        real_index, fake_index, real_data, fake_data = get_index(data)
        real_dataset = torch.utils.data.Subset(dataset, real_index)
        fake_dataset = torch.utils.data.Subset(dataset, fake_index)
        return real_dataset, fake_dataset
    elif isinstance(dataset, torch.utils.data.DataLoader):
        dl = dataset
        data = dl.dataset.data
        real_index, fake_index, real_data, fake_data = get_index(data)
        real_dataset = torch.utils.data.Subset(dl.dataset, real_index)
        fake_dataset = torch.utils.data.Subset(dl.dataset, fake_index)
        real_dl = copy.deepcopy(dl)
        real_dl.dataset.data = real_dataset.data
        fake_dl = copy.deepcopy(dl)
        fake_dl.dataset = fake_dl
        return real_dl, fake_dl
    else:
        raise TypeError('Input should be dataset or dataloader')


def trainGMM(clusters, train_ds, val_ds, test_ds, test=0):
    train_ds_real, train_ds_fake = split_real_fake_dataset(train_ds)
    val_ds_real, val_ds_fake = split_real_fake_dataset(val_ds)
    # test_ds_real, test_ds_fake = split_real_fake_dataset(test_ds)

    print(len(train_ds_real), len(train_ds_fake), len(train_ds))

    real_model = GMM(clusters, inital_data=flatten_dataloader(train_ds_real, clusters+1))
    fake_model = GMM(clusters, inital_data=flatten_dataloader(train_ds_fake, clusters+1))
    model = GMMTest(real_model, fake_model)
    
    trainer = pl.Trainer(
        max_epochs=3,
        accelerator="gpu",
        devices=[0],
        check_val_every_n_epoch=1,
        logger=pl.loggers.CSVLogger(
            "/home/ay/data/Loggers/0-Audio", name="test", version=None
        ),
        default_root_dir="/home/ay/data/Loggers/0-Audio",
    )
    trainer.fit(
        model=real_model,
        train_dataloaders=train_ds_real,
        val_dataloaders=val_ds_real,
    )
