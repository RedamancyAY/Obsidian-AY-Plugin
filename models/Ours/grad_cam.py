# %load_ext autoreload
# %autoreload 2

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from ay2.tools.image import read_image
from model import AudioModel

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# +
from argparse import Namespace

cfg = Namespace(
    **{
        "epochs": 200,
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "lr": 0.0001,
        "lr_decay_factor": 0.5,
        "lr_scheduler": "linear",
        "warmup_epochs": 10,
        "label_smoothing": 0.1,
        "method_classes": 7,
        "pretrain": False,
        "nograd": False,
        "aug_policy": "ss",
        "use_op_loss": 1,
        "style_shuffle": 1,
        "feat_shuffle": 1,
        "voc_con_loss": 1,
        "feat_con_loss": 1,
        "use_adversarial_loss": 1,
        "feature_extractor": "ResNet",
        "dims": [32, 64, 64, 128],
        "n_blocks": [1, 1, 2, 1],
        "beta": [2.0, 0.5, 0.5],
        "one_stem": False,
    }
)
# -

ckpt = "/home/ay/data/DATA/1-model_save/00-Deepfake/1-df-audio/Ours/ResNet/LibriSeVoc_cross_method/version_0/checkpoints/best-epoch=3-val-auc=1.0000.ckpt"

sd = torch.load(ckpt)["state_dict"]
new_sd = {}

new_sd = {}
for key, value in sd.items():
    print(key, value.shape)
    if key.startswith("model."):
        key = ".".join(key.split(".")[1:])
        new_sd[key.replace("model.", "", 0)] = value

model = AudioModel(
    feature_extractor="ResNet",
    cfg=cfg,
    vocoder_classes=7,
)
model.gradcam = 1

model.load_state_dict(new_sd)

import os
path = "/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave"
audio_paths = []
for _path in os.listdir(path)[::100]:
    audio_path = os.path.join(path, _path)
    if not audio_path.endswith('wav'):
        continue
    x, fps = torchaudio.load(audio_path)
    if x.shape[-1] > 48000:
        audio_paths.append(audio_path)
    if len(audio_paths) > 10:
        break

audio_paths

0'/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/302_123516_000010_000000_gen.wav',
1 '/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/196_122159_000011_000001_gen.wav',
2 '/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/6415_111615_000003_000001_gen.wav',
3 '/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/6367_74004_000004_000010_gen.wav',
4 '/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/6415_100596_000060_000000_gen.wav',
5 '/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/1183_133256_000040_000000_gen.wav',
6 '/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/2691_156755_000017_000004_gen.wav',
7 '/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/8797_294123_000009_000002_gen.wav',
8 '/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/4195_186237_000010_000001_gen.wav',
9 '/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/587_54108_000054_000000_gen.wav',
10 '/home/ay/data/DATA/2-datasets/1-df-audio/LibriSeVoc/diffwave/4362_15663_000036_000006_gen.wav'

for i in range(10):
    x, fps = torchaudio.load(audio_paths[i])
    x = x[:, None, :48000]
    spec = model.feature_model.preprocess(x)[0, 0]
    plt.imsave(f"cams/{i}.jpg", spec)

    for target in [0, 1]:
        target_layers = [model.feature_model.layer4_copy[-1]]
        cam = get_cam(target_layers, x, target_class=target)
        cam = plot_cam(f"cams/{i}.jpg", cam)
        plt.imsave(f"cams/{i}-voc-{target}.jpg", cam)
    
        target_layers = [model.feature_model.model.layer4[-1]]
        cam = get_cam(target_layers, x, target_class=target)
        cam = plot_cam(f"cams/{i}.jpg", cam)
        plt.imsave(f"cams/{i}-content-{target}.jpg", cam)


def get_cam(target_layers, x, target_class=0):
    input_tensor = x
    input_tensor = model.feature_model.preprocess(input_tensor)
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    return grayscale_cam


def plot_cam(spec_path, grayscale_cam):
    
    mask = grayscale_cam
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    spec = read_image(spec_path) / 255
    # spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))
    image_weight = 0.5
    cam = (1 - image_weight) * heatmap + image_weight * spec
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    plt.imshow(cam)
    return cam


target_layers = [model.feature_model.layer4_copy[-1]]
cam = get_cam(target_layers, x)
cam = plot_cam('test.jpg', cam)

target_layers = [model.feature_model.model.layer4[-1]]
cam = get_cam(target_layers, x)
plot_cam('test.jpg', cam)

# !zip -r cams.zip cams
