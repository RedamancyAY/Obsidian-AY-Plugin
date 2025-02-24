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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision
from numpy import prod

from torchaudio.transforms import MFCC

from torchvision.models import vgg19


# # Mel

class Melspec(nn.Module):
    def __init__(self):
        super().__init__()

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            hop_length=215,
            n_fft=512,
            n_mels=224,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:torch.Tensor: (B, 1, 48000)

        Returns:
            a tensor: (B, 3, 224, 224)
        """
        mel = self.mel_spectrogram(x)
        mel = torch.concat([mel, mel, mel], dim=1)
        return mel


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# x = torch.randn(2, 1, 16000 * 3)
# m = Melspec()
# m(x).shape
# -

# ---

# # VGG18

#
# 模型代码见：[torchvision.models.vgg — Torchvision main documentation](https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html#vgg19)
#
# 原始模型的forward：
# ```python
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
# ```

    def __init__(self):
        super().__init__()

        self.model = vgg19(weights=torchvision.models.VGG19_Weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:torch.Tensor: (B, 3, 224, 224)

        Returns:
            a tensor: (B, 512, 7, 7)
        """
        x = self.model.features(x)  # (B, 512, 7, 7)
        # x = self.model.avgpool(x) # (B, 512, 7, 7)
        # x = torch.flatten(x, 1) # (B, 512)
        return x

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# x = torch.randn(2, 3, 224, 224)
# model = VGG18()
# model(x).shape
# -

# # Attention

from einops import rearrange


class Attention_to_1D(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:torch.Tensor: (B, 512, 7, 7)

        Returns:
            a tensor: (B, 512, 7, 7)
        """
        x = rearrange(x, "b c h w -> b h w c")
        attn_weight = self.attn(x)  # (b, h, w, 1)

        x = x * attn_weight  # (b, h, w, c)
        x = x.sum((1, 2))  # (b, c)
        return x


class Attention_to_2D(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = nn.Linear(49, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
          x:torch.Tensor: (B, 512, 7, 7)

        Returns:
            a tensor: (B, 512, 7, 7)
        """
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b c (h w)")
        attn_weight = self.attn(x)  # (b, c, 1)

        x = x * attn_weight  # (b, c, (hw))
        x = x.sum((1), keepdims=True)  # (b, 1, hw)
        x = rearrange(x, "b 1 (h w) -> b 1 h w", h=h)
        return x


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# m = Attention_to_2D()
# x = torch.randn(2, 512, 7, 7)
# m(x).shape
# -

# # Capsule Network

# ## conv layer

# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# conv_layer = nn.Sequential(
#     nn.Conv2d(1, 512, 3, stride=2, bias=True, padding=0),
#     nn.ReLU(),
#     nn.Conv2d(512, 512, 3, stride=2, bias=True, padding=0),
# )
# x = torch.randn(2, 1, 7, 7)
# conv_layer(x).shape
# -

# ## PrimaryCaps

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32, kernel_size=9, stride=2):
        """Constructs a list of convolutional layers to be used in
        creating capsule output vectors.
        param num_capsules: number of capsules to create
        param in_channels: input depth of features, default value = 256
        param out_channels: output depth of the convolutional layers, default value = 32
        """
        super(PrimaryCaps, self).__init__()

        # creating a list of convolutional layers for each capsule I want to create
        # all capsules have a conv layer with the same parameters
        self.capsules = nn.ModuleList(
            [nn.Linear(in_features=in_channels, out_features=out_channels) for _ in range(num_capsules)]
        )

    def forward(self, x):
        """Defines the feedforward behavior.
        param x: the input; features from a convolutional layer
        return: a set of normalized, capsule output vectors
        """
        # get batch size of inputs
        batch_size = x.size(0)
        # reshape convolutional layer outputs to be (batch_size, vector_dim=1152, 1)
        # print(self.capsules[0](x).shape)

        u = [capsule(x).view(batch_size, -1, 1) for capsule in self.capsules]

        # stack up output vectors, u, one for each capsule
        u = torch.cat(u, dim=-1)
        # squashing the stack of vectors
        u_squash = self.squash(u)
        return u_squash

    def squash(self, input_tensor):
        """Squashes an input Tensor so it has a magnitude between 0-1.
        param input_tensor: a stack of capsule inputs, s_j
        return: a stack of normalized, capsule output vectors, v_j
        """
        squared_norm = (input_tensor**2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)  # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# x = torch.randn(2, 512)
# m = PrimaryCaps(kernel_size=3, in_channels=512, out_channels=512, stride=1, num_capsules=32)
# m(x).shape
# -





# +
def softmax(input_tensor, dim=1):
    # transpose input
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
    # calculate softmax
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    # un-transpose result
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input_tensor.size()) - 1)


# dynamic routing
def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    """Performs dynamic routing between two capsule layers.
    param b_ij: initial log probabilities that capsule i should be coupled to capsule j
    param u_hat: input, weighted capsule vectors, W u
    param squash: given, normalizing squash function
    param routing_iterations: number of times to update coupling coefficients
    return: v_j, output capsule vectors
    """
    # update b_ij, c_ij for number of routing iterations
    for iteration in range(routing_iterations):
        # softmax calculation of coupling coefficients, c_ij
        c_ij = softmax(b_ij, dim=2)

        # calculating total capsule inputs, s_j = sum(c_ij*u_hat)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)

        # squashing to get a normalized vector output, v_j
        v_j = squash(s_j)

        # if not on the last iteration, calculate agreement and new b_ij
        if iteration < routing_iterations - 1:
            # agreement
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)

            # new b_ij
            b_ij = b_ij + a_ij

    return v_j  # return latest v_j


class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, previous_layer_nodes=32 * 6 * 6, in_channels=8, out_channels=16):
        """Constructs an initial weight matrix, W, and sets class variables.
        param num_capsules: number of capsules to create
        param previous_layer_nodes: dimension of input capsule vector, default value = 1152
        param in_channels: number of capsules in previous layer, default value = 8
        param out_channels: dimensions of output capsule vector, default value = 16
        """
        super(DigitCaps, self).__init__()

        # setting class variables
        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes  # vector input (dim=1152)
        self.in_channels = in_channels  # previous layer's number of capsules

        # starting out with a randomly initialized weight matrix, W
        # these will be the weights connecting the PrimaryCaps and DigitCaps layers
        self.W = nn.Parameter(torch.randn(num_capsules, previous_layer_nodes, in_channels, out_channels))

    def forward(self, u):
        """Defines the feedforward behavior.
        param u: the input; vectors from the previous PrimaryCaps layer
        return: a set of normalized, capsule output vectors
        """

        # adding batch_size dims and stacking all u vectors
        u = u[None, :, :, None, :]
        # 4D weight matrix
        W = self.W[:, None, :, :, :]

        # calculating u_hat = W*u
        u_hat = torch.matmul(u, W)

        # getting the correct size of b_ij
        # setting them all to 0, initially
        b_ij = torch.zeros(*u_hat.size(), device=u.device)

        # moving b_ij to GPU, if available
        # if TRAIN_ON_GPU:
        # b_ij = b_ij.cuda()

        # update coupling coefficients and calculate v_j
        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

        return v_j.transpose(0, 1).squeeze((2, 3))  # return final vector outputs

    def squash(self, input_tensor):
        """Squashes an input Tensor so it has a magnitude between 0-1.
        param input_tensor: a stack of capsule inputs, s_j
        return: a stack of normalized, capsule output vectors, v_j
        """
        # same squash function as before
        squared_norm = (input_tensor**2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)  # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)
        return output_tensor


# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# d = DigitCaps(in_channels=32, num_capsules=32, out_channels=512, previous_layer_nodes=512).cuda()
# x = torch.randn(2, 512, 32).cuda()
# d(x).shape
# -

class CN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(1, 512, 3, stride=2, bias=True, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=2, bias=True, padding=0),
        )

        num_capsules = 32

        self.primary1 = PrimaryCaps(
            kernel_size=3, in_channels=512, out_channels=512, stride=1, num_capsules=num_capsules
        )
        self.digits1 = DigitCaps(
            in_channels=num_capsules, num_capsules=num_capsules, out_channels=512, previous_layer_nodes=512
        )

        self.primary2 = PrimaryCaps(
            kernel_size=3, in_channels=512, out_channels=512, stride=1, num_capsules=num_capsules
        )
        self.digits2 = DigitCaps(in_channels=num_capsules, num_capsules=2, out_channels=512, previous_layer_nodes=512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x:torch.Tensor: (B, 512, 7, 7)

        Returns:
            a tensor: (B, 512, 7, 7)
        """
        x = self.conv_layer(x).squeeze(-1).squeeze(-1)  # (B, 512)
        x = self.primary1(x)  # (B, 512, num_capsules)
        x = self.digits1(x)  # (B, num_capsules, 512)

        x = x.transpose(1, 2).mean(-1)  # (B, num_capsules, 512)-> (B, 512, num_capsules) -> (B, 512)
        x = self.primary2(x)  # (B, 512, num_capsules)
        x = self.digits2(x)  # (B, 2, 512)
        classes = (x**2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)  # (b, 2)
        return classes


# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# x = torch.randn(2, 1, 7, 7)
# m1 = CN()
# m1(x).shape
# -

# # Model

class ABCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = Melspec()
        self.model = VGG18()
        self.attention = Attention_to_2D()
        self.capsule = CN()

    def forward(self, x):
        x = self.mel(x)  # (B, 3, 224, 224)
        x = self.model(x)  # (B, 512, 7, 7)
        x = self.attention(x)  # (B, 1, 7, 7)
        x = self.capsule(x)  # (B, 2)
        return x


# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# x = torch.randn(2, 1, 48000)
# model = ABCNet()
# model(x)

# + editable=true slideshow={"slide_type": ""} tags=["style-solution", "active-ipynb"]
# x = torch.randn(2, 1, 48000).cuda()
# model = ABCNet().cuda()
# model(x)
# -

# # Lit Model

from ay2.torch.deepfake_detection import DeepfakeAudioClassification


# + editable=true slideshow={"slide_type": ""}
class ABCNet_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        self.model = ABCNet()
        self.configure_loss_fn()

        if args is not None and hasattr(args, "profiler"):
            self.profiler = args.profiler
        else:
            self.profiler = None

        self.lr = 1e-4
        self.save_hyperparameters()

    def configure_loss_fn(self):
        self.bce_loss = nn.BCEWithLogitsLoss()

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        batch_size = len(label)
        cls_loss = self.bce_loss(batch_res["logit"], label.type(torch.float32))
        loss = cls_loss
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.num_training_batches = self.trainer.num_training_batches
        return [optimizer]

    def _shared_pred(self, batch, batch_idx, stage="train"):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        # print(batch['language'])

        B = len(audio)
        logit = self.model(audio)

        return {
            "logit": logit,
        }
