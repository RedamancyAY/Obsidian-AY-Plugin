# %load_ext autoreload
# %autoreload 2

import torch
import torchaudio
from textless.data.speech_encoder import SpeechEncoder
from torch_kmeans import KMeans


def load_SpeechEncoder():
    dense_model_name = "hubert-base-ls960"
    quantizer_name, vocab_size = "kmeans", 500
    # input_file = "/home/ay/LibriSeVoc/diffwave/103_1241_000004_000002_gen.wav"

    # We can build a speech encoder module using names of pre-trained
    # dense and quantizer models.  The call below will download
    # appropriate checkpoints as needed behind the scenes. We can
    # also construct an encoder by directly passing model instances
    encoder = SpeechEncoder.by_name(
        dense_model_name=dense_model_name,
        quantizer_model_name=quantizer_name,
        vocab_size=vocab_size,
        deduplicate=True,
        need_f0=False
    )
    return encoder


def encode_speech(encoder, x):
    # res is a dict with keys ('dense', 'units', 'durations').
    # It can also contain 'f0' if SpeechEncoder was initialized
    # with need_f0=True flag.
    res = encoder(x)
    
    units = res["units"]  # tensor([71, 12, 57, ...], ...)
    res['original_units'] = torch.repeat_interleave(units, res["durations"])
    return res

# +
input_file = "/home/ay/DFDC/audio/aaaoqepxnf.wav"
waveform, sample_rate = torchaudio.load(input_file)
x = waveform.cuda()

encoder = load_SpeechEncoder().cuda()

# now convert it in a stream of deduplicated units (as in GSLM)
res = encode_speech(encoder, x[:, :48000])

res

# +
import torch
from torch_kmeans import KMeans

model = KMeans(n_clusters=4)

x = torch.randn((1, 1000, 30))   # (BS, N, D)
labels = model.fit_predict(x)
print(labels)
# -

model = KMeans(n_clusters=4).fit(x)

model

# +
from sklearn.cluster import MiniBatchKMeans
import numpy as np
# Assuming features is your initial dataset and new_features is new data arriving later
initial_data = np.random.randn(100, 30)   # load or generate initial data
new_data = np.random.randn(100, 30)     # load or generate new data

# Initialize MiniBatchKMeans
mb_kmeans = MiniBatchKMeans(n_init='auto', n_clusters=10, random_state=0, batch_size=100)

# Initial training with available data
mb_kmeans.fit(initial_data)

# As new data arrives
mb_kmeans.partial_fit(new_data)
# -

mb_kmeans.predict(new_data)

# # Load dataset

# +
import sys

sys.path.append("/home/ay/zky/Coding2/0-Deepfake/2-Audio")
# -

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler, default_collate, random_split
import pandas as pd
try:
    from .data.tools import WaveDataset
except ImportError:
    from data.tools import WaveDataset

from ay2.datasets.audio import MLAAD_AudioDs
from ay2.torchaudio.transforms.self_operation import CentralAudioClip
from tqdm import tqdm


def load_MLAAD_subset(root_path: str = "/home/ay/data/0-原始数据集/MLADD", language: str = "en", n_audios=1000):
    ds = MLAAD_AudioDs(root_path=root_path)
    if isinstance(language, str):
        data = ds.data.query(f"language == {language}")
    else:
        data = ds.data.query(f"language in {language}")

    sampled_data = [data.query(f"label == {i}").sample(n_audios//2, random_state=42) for i in [0, 1]]
    sampled_data = pd.concat(sampled_data, ignore_index=True)

    _datasets = []
    for label in [0, 1]:
        sampled_data = data.query(f"label == {label}").sample(n_audios, random_state=42)
        _ds = WaveDataset(
            sampled_data,
            sample_rate=16000,
            normalize=True,
            transform=[CentralAudioClip(48000)],
            dtype="tensor",
        )
        print(f"read {n_audios} {label} audios for the language {language} in {root_path}")

        _dl = DataLoader(_ds, num_workers=5, batch_size=32)
        _datasets.append(_dl)
    return _datasets


# _ds = load_MLAAD_subset(language=["en", "es", "de"], n_audios=3000)
fake_ds, real_ds = load_MLAAD_subset(language=["en"], n_audios=10240)

# +
mb_kmeans = MiniBatchKMeans(n_init='auto', n_clusters=5, random_state=0, batch_size=32)

for batch in tqdm(real_ds):

    batch_res = []
    
    for x in batch['audio']: ### torch.Size([1, 48000])
        with torch.no_grad():
            res = encode_speech(encoder=encoder, x=x.cuda())
            # batch_res.append(res['original_units'] / 500)
            batch_res.append(res['dense'].mean(-1))
    batch_res = torch.stack(batch_res).cpu().numpy()
    mb_kmeans.partial_fit(batch_res)
# -

mb_kmeans.predict(batch_res)

# +
from scipy.spatial.distance import cdist

def detect_anomalies(new_features, centroids, threshold):
    distances = cdist(new_features, centroids, 'euclidean')
    min_distances = np.min(distances, axis=1)
    print(min_distances)
    anomalies = min_distances > threshold
    return anomalies


# -

detection_res = []
for batch in tqdm(real_ds):
    batch_res = []

    
    for x in batch['audio']: ### torch.Size([1, 48000])
        with torch.no_grad():
            res = encode_speech(encoder=encoder, x=x.cuda())
            # res = (res['original_units']/500).cpu().numpy()
            res = (res['dense'].mean(-1)).cpu().numpy()
    
            anomalies = detect_anomalies(res[None,...], mb_kmeans.cluster_centers_, 0.6)
            detection_res += list(anomalies)
    print(anomalies)
    break

detection_res = []
for batch in tqdm(fake_ds):
    batch_res = []

    
    for x in batch['audio']: ### torch.Size([1, 48000])
        with torch.no_grad():
            res = encode_speech(encoder=encoder, x=x.cuda())
            # res = (res['original_units']/500).cpu().numpy()
            res = (res['dense'].mean(-1)).cpu().numpy()
    
            anomalies = detect_anomalies(res[None,...], mb_kmeans.cluster_centers_, 0.6)
            detection_res += list(anomalies)
    print(anomalies)
    break
