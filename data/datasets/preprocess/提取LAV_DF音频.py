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

import pandas as pd
from ay2.tools import check_dir, read_file_paths_from_folder
from moviepy.editor import VideoFileClip
from pandarallel import pandarallel
from torchvision.io import read_video
from tqdm.auto import tqdm
import torchaudio
# -

root_path = "/home/ay/data/0-原始数据集/LAV-DF"
video_paths = read_file_paths_from_folder(root_path, exts="mp4")


def load_audio_from_video(_video_path):
    _audio_path = _video_path.replace("LAV-DF", "LAV-DF-Audio").replace(".mp4", ".wav")
    check_dir(_audio_path)
    # print(_audio_path)
    if os.path.exists(_audio_path):
        return 1
    video, audio, metadata = read_video(video_paths[0])
    torchaudio.save(_audio_path, audio, sample_rate=16_000)
    return 1


data = pd.DataFrame(video_paths, columns=["path"])
pandarallel.initialize(progress_bar=True, nb_workers=10)
_ = data["path"].parallel_apply(load_audio_from_video)
