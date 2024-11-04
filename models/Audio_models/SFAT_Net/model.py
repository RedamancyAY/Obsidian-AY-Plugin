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

# + editable=true slideshow={"slide_type": ""}
import math

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms
import torchyin
from einops import rearrange, repeat


# + [markdown] editable=true slideshow={"slide_type": ""}
# # Model Componenets
# -

# ## Preprocess　

# > Furthermore, each input audio file was processed by a standard pre-emphasis filter with coefficient 0.97 to emphasize the mid-high frequencies, and we imposed a window length of 32 ms and a hop size of 16 ms to extract the log spectrograms. The final input spectrograms were fixed to have size of $L=128$ frames by $M=256$ frequency bins, corresponding to 2.064 seconds of content. The patches were created by applying a $P_L=16 \times P_M=16$ grid, corresponding to a sequence with length $N=128$.

# There are three operations in preprocess:
# 1. use PreEmphasis to filter the input audio
# 2. use spectrogram to transform the audio into spectrogram
# 3. crop the spectrogram into patches.

# ### Pre-Emphasis

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97) -> None:
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            "flipped_filter",
            torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        assert x.ndim in [2, 3]
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # reflect padding to match lengths of in/out
        x = F.pad(x, (1, 0), "reflect")
        return F.conv1d(x, self.flipped_filter)


# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# x = torch.rand(1, 5)
# model = PreEmphasis()
# model(x), x
# -

# ### Spectrogram

# > We imposed a window length of 32 ms and a hop size of 16 ms to extract the log spectrograms. The final input spectrograms were fixed to have size of $L=128$ frames by $M=256$ frequency bins, corresponding to 2.064 seconds of content.

# For a audio with 16000 HZ:
# - 32ms windows length: win_length = 16 * 32 = 512
# - 16ms hop size: hop_length = 16 * 16 = 256
# - 256 frequency bins: n_fft = 511, but 0 < win_length <= n_fft, so, I set win_length to 511
# - For the frames, the author use 2.064s audio to generate 128 frames. About 64 frames per second. Therefore, I further adjust hop_length to 250, to generate 64 frames per second.
#
# The final setting is:
# ```python
# stft = torchaudio.transforms.Spectrogram(
#     n_fft=511,
#     win_length=511,
#     hop_length=250,
# )
# ```

class CustomSpectrogram(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=511,
            win_length=511,
            hop_length=250,
        )

    def forward(self, x):
        x = self.stft(x)
        x = torch.log(x + 1e-7)
        return x


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# x = torch.randn(2, 1, 16000 * 3)
# t = CustomSpectrogram()
# t(x).shape
# -

# ### Crop patches

# > Let us denote with $X \in \mathbb{R}^{L \times M}$ the log-spectrogram of an input recording $x(t)$, with $L$ denoting the number of frames and $M$ the number of frequency bins of $X$.
#
#
#
# > In a first step, the input log-spectrogram $X$ is split into a sequence of non-overlapping 2D patches $x_p \in \mathbb{R}^{P_L \times P_M}$. Each patch has a fixed amount of rows $\left(P_L\right)$ and of columns $\left(P_M\right)$, and the total amount of patches $N$ is fixed and equal to $(L \cdot M) /\left(P_L \cdot P_M\right)$
# > $$
# > x_p=\operatorname{reshape}\left(X, P_L, P_M\right)
# > $$
# > with $p \in 1 \ldots N$.
#
# > The patches were created by applying a $P_L=16 \times P_M=16$ grid, corresponding to a sequence with length $N=128$.

# The patch size is $16 \times 16$.

class CropPatch(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = rearrange(x, "B 1 (m pm) (l pl)-> B (m l) (pm pl)", pm=16, pl=16)
        return x


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# x = torch.randn([2, 1, 256, 192])
# t = CropPatch()
# print(t(x).shape)

# + [markdown] editable=true slideshow={"slide_type": ""}
# ### Combine above three steps

# + editable=true slideshow={"slide_type": ""}
class Preprocess(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pre_emphasis = PreEmphasis()
        self.spec_transform = CustomSpectrogram()
        self.crop_patch = CropPatch()

    def forward(self, x):
        """
        Args:
            x: (B, 1, L), L = 48000
        """
        x = self.pre_emphasis(x)  # (B, 1, L)
        spec = self.spec_transform(x)  # (B, 1, H, W)
        patch = self.crop_patch(spec)  # (B, HW/256, 16*16)
        return x, spec, patch


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# module = Preprocess()
# x = torch.randn(3, 1, 48000)
# x, spec, patch = module(x)
# print(patch.shape, spec.shape)
# -

# ## Encoder　& Decoders　

# | Component | Global Params |  | MLP Blocks |  | Self-Attention Blocks |  |
# | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
# |  | Depth | Embedding Size | Dimensions | Dropout | Number of Heads | Head Dimension |
# | $E$ - Spectrogram Encoder | 8 | 1024 | 2048 | 0 | 8 | 64 |
# | $D_{\text {dec }}^X$ - Spectrogram Decoder | 6 | 512 | 1024 | 0 | 8 | 64 |
# | $D_{\text {dec }}^{f_0}$ - F0 Decoder | 4 | 512 | 1024 | 0 | 8 | 64 |
# | $P-$ Synthesis Predictor | 4 | 512 | 1024 | 0.1 | 6 | 64 |

# + editable=true slideshow={"slide_type": ""}
try:
    from vit_pytorch import ViT
except ImportError:
    from .vit_pytorch import ViT


# -

# ### help functions

# +
def get_transformer(dim=1024, depth=8, heads=8, mlp_dim=2048, dropout=0.1):
    v = ViT(
        image_size=256,
        patch_size=16,
        num_classes=1000,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=0.1,
    )
    return v.transformer


def get_pos_embedding(num_patches=192, dim=1024):
    pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
    return pos_embedding


# -

# ### Encoder　

class Encoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim = 1024

        ### the original patch is 16x16, thus we need project its dim from 256 into dim
        self.mlp = nn.Linear(256, dim)

        self.pos_embedding = get_pos_embedding(dim=dim, num_patches=192)
        self.transformer = get_transformer(dim=1024, depth=8, heads=8, mlp_dim=2048, dropout=0)

    def forward(self, x):
        x = self.mlp(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n, :]  # (b, n, dim) + (1, n, dim)
        x = self.transformer(x)
        return x


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# encoder = Encoder()
# x = torch.randn(2, 192, 256)
# encoder(x).shape
# -

# ### SpectrogramDecoder

class SpectrogramDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim = 512

        ### the dim of encoder output is 1024, thus we need project its dim from 1024 into dim
        self.mlp = nn.Linear(1024, dim)

        self.pos_embedding = get_pos_embedding(dim=dim, num_patches=192)
        self.transformer = get_transformer(dim=dim, depth=6, heads=8, mlp_dim=1024, dropout=0)

        self.mlp2 = nn.Linear(dim, 16 * 16)

    def forward(self, x, num_bins, num_frames, patch_size=16):
        x = self.mlp(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n, :]  # (b, n, dim) + (1, n, dim)
        x = self.transformer(x)

        ## convert embedding into spectrogram
        x = self.mlp2(x)
        x = rearrange(
            x,
            "B (m l) (pm pl) -> B 1 (m pm) (l pl)",
            pm=patch_size,
            pl=patch_size,
            m=num_bins // patch_size,
            l=num_frames // patch_size,
        )
        return x


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# m = SpectrogramDecoder()
# x = torch.randn(2, 192, 1024)
# y = m(x, 256, 192)
# print(y.shape)
# -

# ### FundamentalFrequencyDecoder

# + [markdown] editable=true slideshow={"slide_type": ""}
# #### Extract F0　
# -

# Let us denote with $F_0 \in \mathbb{R}^{L \times M}$ the contour matrix of the $f_0$ trajectory, i.e. a matrix with the same dimension as the input spectrogram, in which
#
# $$
# F_0(l, m)= \begin{cases}1 & \text { if } f_0(m) \approx \frac{f_s}{2 M} \cdot m \\ 0 & \text { otherwise }\end{cases}
# $$
#

# + editable=true slideshow={"slide_type": ""}
def get_f0(x):
    """
    Assume that the input audio x is with shape (B, 1, 48000). If its length is not equal to 48000,
    you may have to change th frame stride (second).
    """
    pitch = torchyin.estimate(
        x[:, 0, :],
        sample_rate=16000,
        pitch_min=20,
        pitch_max=9000,
        frame_stride=0.01513,  # actually is 0.015625
    )
    return pitch


# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# x = torch.randn(8, 1, 48000)
# get_f0(x).shape

# +
def get_f0_matrix(pitch, spec, eps=1):
    res = torch.zeros_like(spec[:, 0, :, :])
    B, _, num_bins, num_frames = spec.shape
    for batch in range(B):
        for i in range(num_bins):
            for j in range(num_frames):
                base = 16000 / 512 * (i + 1)
                if pitch[batch, j] > base - eps and pitch[batch, j] < base + eps:
                    res[batch, i, j] = 1
                else:
                    res[batch, i, j] = 0
    return res


def get_f0_matrix2(pitch, spec, eps=1):
    base_frequencies = (16000 / 512) * (torch.arange(256, device=pitch.device) + 1)
    base_frequencies = base_frequencies.view(1, -1, 1)  # Shape: (1, 256, 1)
    pitch_expanded = pitch.unsqueeze(1)  # Shape: (8, 1, 192)
    condition = (pitch_expanded > base_frequencies - eps) & (pitch_expanded < base_frequencies + eps)
    res2 = condition.float()
    return res2


# +
# x, sr = torchaudio.load("/home/ay/LibriSeVoc/melgan/103_1241_000004_000002_gen.wav")
# x = x[:, None, 48000 : 48000 * 2]
# t = CustomSpectrogram()
# spec = t(x)
# pitch = get_f0(x)
# res1 = get_f0_matrix(pitch, spec)

# res2 = get_f0_matrix2(pitch, spec)
# torch.abs(res1 - res2).sum()
# -

class GaussianConv2d(nn.Module):
    def __init__(self):
        super(GaussianConv2d, self).__init__()

        # Define the Gaussian Kernel
        gaussian_kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        gaussian_kernel /= gaussian_kernel.sum()  # Normalize the kernel

        # Convert to 4D tensor required by nn.Conv2d (out_channels, in_channels, height, width)
        gaussian_kernel = gaussian_kernel.view(1, 1, 3, 3)  # (1, 1, 3, 3)

        # Initialize the Conv2d layer
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        # Set the kernel weight to the Gaussian kernel
        with torch.no_grad():
            self.conv.weight = nn.Parameter(gaussian_kernel, requires_grad=False)

        # Freeze the weights
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


# + editable=true slideshow={"slide_type": ""}
class F0ReconstructionLoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = GaussianConv2d()
        self.criterion = nn.MSELoss()

    def extract_f0(self, x):
        pitch = get_f0(x)
        res = get_f0_matrix2(pitch, spec=None)  # (B, num_bins, num_frames)
        res = res[:, None, :, :]
        res = self.conv(res)
        return res

    def forward(self, x, f0):
        return self.compute_loss(x, f0)

    def compute_loss(self, x, f0):
        f0_target = self.extract_f0(x)
        loss = self.criterion(f0, f0_target)
        return loss


# -

# #### build Decoder 

# + editable=true slideshow={"slide_type": ""}
class FundamentalFrequencyDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim = 512

        ### the dim of encoder output is 1024, thus we need project its dim from 1024 into dim
        self.mlp = nn.Linear(1024, dim)

        self.pos_embedding = get_pos_embedding(dim=dim, num_patches=192)
        self.transformer = get_transformer(dim=dim, depth=4, heads=8, mlp_dim=1024, dropout=0)

        self.mlp2 = nn.Linear(dim, 16 * 16)

    def forward(self, x, num_bins, num_frames, patch_size=16):
        x = self.mlp(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n, :]  # (b, n, dim) + (1, n, dim)
        x = self.transformer(x)

        ## convert embedding into spectrogram
        x = self.mlp2(x)
        x = rearrange(
            x,
            "B (m l) (pm pl) -> B 1 (m pm) (l pl)",
            pm=patch_size,
            pl=patch_size,
            m=num_bins // patch_size,
            l=num_frames // patch_size,
        )
        return x


# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# m = FundamentalFrequencyDecoder()
# x = torch.randn(2, 192, 1024)
# y = m(x, 256, 192)
# print(y.shape)
# -

# ### Cls Decoder

# + editable=true slideshow={"slide_type": ""}
class ClsDecoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        dim = 512

        ### the dim of encoder output is 1024, thus we need project its dim from 1024 into dim
        self.mlp = nn.Linear(1024, dim)

        self.pos_embedding = get_pos_embedding(dim=dim, num_patches=192)
        self.transformer = get_transformer(dim=dim, depth=4, heads=6, mlp_dim=1024, dropout=0.1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_head = nn.Linear(dim, 1)

    def forward(self, x):
        b, n, _ = x.shape

        x = self.mlp(x)
        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding[:, : n + 1, :]  # (b, n, dim) + (1, n, dim)
        x = self.transformer(x)

        ## classification
        feat = x[:, 0, :]
        x = self.cls_head(feat)
        return x, feat


# + tags=["style-solution", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# m = ClsDecoder()
# x = torch.randn(2, 192, 1024)
# y, feat = m(x)
# print(y.shape)

# + [markdown] editable=true slideshow={"slide_type": ""}
# # Build Model

# + editable=true slideshow={"slide_type": ""}
class SFATNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.preprocess = Preprocess()
        self.encoder = Encoder()
        self.spec_decoder = SpectrogramDecoder()
        self.f0_decoder = FundamentalFrequencyDecoder()
        self.cls_decoder = ClsDecoder()

    def forward(self, x):
        x_new, spec, patch = self.preprocess(x)

        feat = self.encoder(patch)

        num_bins, num_frames, patch_size = spec.shape[-2], spec.shape[-1], 16
        
        pred_spec = self.spec_decoder(feat, num_bins, num_frames, patch_size)
        pred_f0 = self.f0_decoder(feat, num_bins, num_frames, patch_size)
        pred_logit, feature = self.cls_decoder(feat)

        return {
            "emphasis_x": x,
            "spec": spec,
            "pred_spec": pred_spec,
            "pred_f0": pred_f0,
            "logit": pred_logit.squeeze(-1),
            "feature" : feature
        }

# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# x = torch.randn(3, 1, 48000)
# m = SFATNet()
# m(x)
