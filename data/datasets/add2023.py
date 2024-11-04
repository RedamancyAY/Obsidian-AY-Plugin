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

# %load_ext autoreload
# %autoreload 2

# +
import os
from typing import List, Union

import pandas as pd
import torch
import torchaudio
from ay2.common.audio import get_fps_len
from ay2.tools import check_dir, read_file_paths_from_folder
from pandarallel import pandarallel
from torch.utils.data import Dataset
from tqdm.auto import tqdm 


# -

# ```json
# track1.2
# ├── test
# │   ├── ADD2023_T1.2R1_E_00008971.wav
# │   └── ADD2023_T1.2R1_E_00018971.wav
# ├── test2
# │   ├── ADD2023_T1.2R2_E_00008971.wav
# │   └── ADD2023_T1.2R2_E_00018971.wav
# ├── train
# │   ├── label.txt
# │   └── wav
# │       ├── ADD2023_T1.2_T_00008971.wav
# │       └── ADD2023_T1.2_T_00018971.wav
# ├── dev
# │   ├── label.txt
# │   └── wav
# │       ├── ADD2023_T1.2_D_00008971.wav
# │       └── ADD2023_T1.2_D_00018971.wav
# ```

class ADD2023:
    """
    read metadatas for the train, dev, test sets for each track in ADD2023.
    """

    def __init__(self, root_path):
        super().__init__()
        self.root_path = root_path

    def read_test_metedata(self, track, test_round=1, generate_metadata=False):
        """
        Args:
            root_pah =
        """
        test_path = os.path.join(
            self.root_path,
            "track%s" % str(track),
            "test%s" % ("" if test_round == 1 else "2"),
        )
        print(test_path)
        data_path = os.path.join(test_path, "dataset_info.csv")
        if os.path.exists(data_path) and not generate_metadata:
            return pd.read_csv(data_path)
        else:
            data = self._read_wav_metadata_from_folder(test_path)
            data.to_csv(data_path, index=False)
            return data

    def _read_train_or_dev_metedata(self, track, item="train", generate_metadata=False):
        item_path = os.path.join(self.root_path, f"track{track}", item)

        wave_path = os.path.join(item_path, "wav")
        label_path = os.path.join(item_path, "label.txt")
        data_path = os.path.join(item_path, "dataset_info.csv")
        # print(data_path)
        if os.path.exists(data_path) and generate_metadata == False:
            return pd.read_csv(data_path)
        else:
            data1 = pd.read_csv(label_path, sep=" ", names=["name", "label"])
            data1["label"] = data1["label"].apply(lambda x: 0 if x == "fake" else 1)
            data2 = self._read_wav_metadata_from_folder(wave_path)

            data = pd.merge(data1, data2, on="name")
            data.to_csv(data_path, index=False)
            return data

    def _read_wav_metadata_from_folder(self, folder):
        wav_paths = read_file_paths_from_folder(folder, exts=["wav"])
        print(wav_paths[0:10])
        data = pd.DataFrame()
        data["path"] = wav_paths
        data["name"] = data["path"].apply(lambda x: os.path.split(x)[1])

        # pandarallel.initialize(progress_bar=True, nb_workers=10)
        tqdm.pandas(desc='Extract sample_rate and length:') 
        data[["fps", "length"]] = data.progress_apply(
            lambda x: tuple(get_fps_len(x["path"])), axis=1, result_type="expand"
        )
        return data


    def read_voice_conversion_metadata(self, track, generate_metadata=False):
        item_path = os.path.join(self.root_path, f"track{track}", 'voice_conversion')
        data_path = os.path.join(item_path, "dataset_info.csv")
        if os.path.exists(data_path) and generate_metadata == False:
            return pd.read_csv(data_path)
        else:
            data = self._read_wav_metadata_from_folder(item_path)
            data["label"] = 0
            data['name'] = data['path'].apply(lambda x: os.path.split(x)[-1])
            data.to_csv(data_path, index=False)
            return data
    
    def read_train_dev_metadata(
        self, track, train_or_dev="train", generate_metadata=False, over_sample=False
    ):
        if isinstance(train_or_dev, str):
            train_or_dev = [train_or_dev]
        data = [
            self._read_train_or_dev_metedata(
                track=track,
                item=item,
                generate_metadata=generate_metadata,
            )
            for item in train_or_dev
        ]
        data = pd.concat(data, ignore_index=True)
        if over_sample:
            data = over_sample_dataset(data)
        return data

# + tags=["active-ipynb"]
# add2023 = ADD2023(root_path="/home/ay/data/ADD2023")
# data = add2023.read_test_metedata(track="1.2", test_round=1, generate_metadata=False)
# print(len(data))

# + tags=["active-ipynb"]
# add2023 = ADD2023(root_path="/home/ay/data/ADD2023")
# data = add2023.read_test_metedata(track="1.2", test_round=2, generate_metadata=False)
# print(len(data))

# + tags=["active-ipynb"]
# data = add2023.read_train_dev_metadata(
#     track="1.2",
#     train_or_dev=["dev", "train"],
#     generate_metadata=False,
# )
# print(len(data))
# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# data = add2023.read_voice_conversion_metadata(
#     track="1.2",
#     generate_metadata=True,
# )
# print(len(data))
#
