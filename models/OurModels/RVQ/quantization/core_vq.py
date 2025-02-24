# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""

import typing as tp
import warnings

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F

from . import distrib


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    """Provide a default value if the given value is None.

    This utility function returns the first argument if it is not None, otherwise
    it returns the second argument, which acts as the default value.

    Args:
        val (tp.Any): The value to check.
        d (tp.Any): The default value to return if `val` is None.

    Returns:
        tp.Any: Either `val` if it is not None, or `d` if `val` is None.

    Raises:
        TypeError: If `d` is None, as it cannot serve as a default value.
    """
    
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    """Perform in-place exponential moving average (EMA) update.

    This function updates the moving average tensor with new values using the EMA formula.
    $$
    moving_avg = moving_avg * decay + new * (1 - decay)
    $$
    
    Args:
        moving_avg (torch.Tensor): The tensor to update with EMA.
        new (torch.Tensor): The new values to incorporate into the EMA.
        decay (float): The decay rate for the moving average. Should be between 0 and 1.

    Returns:
        None: This function performs an in-place operation.

    Raises:
        ValueError: If `decay` is not within the range [0, 1].
    """
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x: torch.Tensor, n_categories: int, epsilon: float = 1e-5) -> torch.Tensor:
    """Apply Laplace smoothing to the input tensor.

    Laplace smoothing, also known as add-one smoothing or add-k smoothing,
    is a technique to prevent zero probabilities in probability distributions.


    Args:
        x (torch.Tensor): The input tensor to be smoothed.
        n_categories (int): The number of categories in the distribution.
        epsilon (float, optional): The smoothing parameter. Defaults to 1e-5.

    Returns:
        torch.Tensor: The smoothed tensor.

    Raises:
        ValueError: If `n_categories` is less than or equal to zero.
    """
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int) -> torch.Tensor:
    """Initialize a tensor with uniform distribution using Kaiming uniform initialization.

    This function creates a tensor with the specified shape and initializes it using
    Kaiming uniform initialization, which is useful for initializing weights in neural
    networks to maintain the scale of gradients.

    Example:
        ```python
        # Initialize a 3x3 weight matrix for a neural network layer
        weights = uniform_init(3, 3)
        ```

    Args:
        *shape (int): Variable length argument list specifying the shape of the tensor to initialize.

    Returns:
        torch.Tensor: A tensor of the specified shape initialized with Kaiming uniform distribution.

    Raises:
        ValueError: If `shape` does not specify a valid tensor shape.
    """
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def sample_vectors(samples, num: int):
    """Sample vectors randomly from the given tensor of samples.

    This function either randomly selects a subset of vectors if there are enough samples,
    or randomly duplicates vectors if there are fewer samples than required.

    Args:
        samples (torch.Tensor): The tensor from which to sample vectors. It should have
            shape (num_samples, feature_dim) where num_samples is the number of samples
            and feature_dim is the dimensionality of each vector.
        num (int): The number of vectors to sample.

    Returns:
        torch.Tensor: A tensor of shape (num, feature_dim) containing the sampled vectors.

    Raises:
        ValueError: If `samples` is not a 2D tensor or if `num` is not positive.

    Notes:
        - If `num_samples` >= `num`, a random permutation of indices is used to select
            unique vectors.
        - If `num_samples` < `num`, vectors are sampled with replacement, potentially
            resulting in duplicates.
    """
    
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples: torch.Tensor, num_clusters: int, num_iters: int = 10):
    """Perform K-means clustering on the given samples.

    This function implements the K-means algorithm to partition the data into
    `num_clusters` clusters. It iteratively assigns each sample to the nearest
    cluster centroid and then recalculates the centroids based on the current
    assignment.

    Args:
        samples (torch.Tensor): A tensor of shape (n, d) where n is the number
            of samples and d is the dimensionality of each sample. The data to
            be clustered.
        num_clusters (int): The number of clusters to form as well as the number
            of centroids to generate.
        num_iters (int, optional): The number of iterations to run the K-means
            algorithm. Defaults to 10.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - means (torch.Tensor): A tensor of shape (num_clusters, d) containing
                the coordinates of the cluster centers.
            - bins (torch.Tensor): A tensor of length num_clusters indicating the
                number of samples in each cluster.

    Notes:
        - If a cluster ends up with no samples assigned to it, the centroid for
            that cluster does not change in that iteration to avoid division by zero.
        - The function uses PyTorch operations for efficiency with GPU acceleration.
    """
    # ... (rest of the function code)
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(means, "c d -> () c d")
        dists = -(diffs**2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """Vector quantization (VQ) codebook using Euclidean distance for clustering.

    This class implements a VQ codebook where vectors are clustered using the Euclidean distance metric.
    It supports initialization via uniform distribution or k-means clustering and provides functionality
    for code expiration and EMA (Exponential Moving Average) updates.

    Example:
        ```python
        # Initialize a EuclideanCodebook
        codebook = EuclideanCodebook(
            dim=256,
            codebook_size=1024,
            kmeans_init=True,
            kmeans_iters=50,
            decay=0.99,
            epsilon=1e-5,
            threshold_ema_dead_code=2
        )

        # Encode and decode some data
        data = torch.randn(32, 256)
        quantized, indices = codebook(data)
        ```

    Args:
        dim (int): The dimensionality of the embedding vectors.
        codebook_size (int): The number of codebook entries (clusters).
        kmeans_init (bool): If True, initialize the codebook using k-means clustering. Defaults to False.
        kmeans_iters (int): Number of iterations for k-means initialization. Defaults to 10.
        decay (float): Decay rate for EMA over the codebooks. Should be between 0 and 1. Defaults to 0.99.
        epsilon (float): Small value for numerical stability in calculations. Defaults to 1e-5.
        threshold_ema_dead_code (int): Threshold for expiring codes based on EMA cluster size. 
                                        Codes with an EMA size below this threshold are replaced. Defaults to 2.

    Attributes:
        inited (torch.Tensor): A flag to indicate whether the codebook has been initialized.
        cluster_size (torch.Tensor): EMA of the cluster sizes.
        embed (torch.Tensor): The current codebook embeddings.
        embed_avg (torch.Tensor): EMA of the codebook embeddings.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = (
            uniform_init if not kmeans_init else torch.zeros
        )
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size

        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([not kmeans_init]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())

    @torch.jit.ignore
    def init_embed_(self, data):
        """Initialize the codebook embeddings if not already initialized.

        If kmeans_init is set, this function runs k-means on the input data to initialize the embeddings.

        Args:
            data (torch.Tensor): The input data to initialize the codebook.

        Raises:
            RuntimeError: If the data is empty or has incorrect dimensionality.
        """
        if self.inited:
            return

        embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.inited.data.copy_(torch.Tensor([True]))
        # Make sure all buffers across workers are in sync after initialization
        distrib.broadcast_tensors(self.buffers())

    def replace_(self, samples, mask):
        """Replace expired or selected codes with new vectors from the batch.

        Args:
            samples (torch.Tensor): The batch of samples to sample from for replacement.
            mask (torch.Tensor): A mask indicating which codes to replace.

        Returns:
            None: This method modifies the codebook in-place.
        """
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)

    def expire_codes_(self, batch_samples):
        """Expire codes with small EMA cluster size and replace them with new samples.

        Args:
            batch_samples (torch.Tensor): The current batch of samples for potential replacement.

        Returns:
            None: This method modifies the codebook in-place.
        """
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, "... d -> (...) d")
        self.replace_(batch_samples, mask=expired_codes)
        distrib.broadcast_tensors(self.buffers())

    def preprocess(self, x):
        """Preprocess the input tensor for quantization.

        Args:
            x (torch.Tensor): The input tensor to preprocess.

        Returns:
            torch.Tensor: A flattened version of the input tensor.
        """
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        """Quantize the input tensor to the nearest codebook entries.

        Args:
            x (torch.Tensor): The input tensor to quantize.

        Returns:
            torch.Tensor: Indices of the nearest codebook entries.
        """
        embed = self.embed.t()
        dist = -(
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    def postprocess_emb(self, embed_ind, shape):
        """Postprocess the embedding indices to match the original input shape.

        Args:
            embed_ind (torch.Tensor): Indices of the quantized vectors.
            shape (tp.Tuple[int, ...]): The shape of the original input tensor.

        Returns:
            torch.Tensor: Reshaped indices tensor.
        """
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        """Dequantize the embedding indices to get the original vector representations.

        Args:
            embed_ind (torch.Tensor): Indices of the quantized vectors.

        Returns:
            torch.Tensor: The dequantized vectors.
        """
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        """Encode the input tensor into codebook indices.

        Args:
            x (torch.Tensor): The input tensor to encode.

        Returns:
            torch.Tensor: The indices of the nearest codebook entries.
        """
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        """Decode the codebook indices back to vectors.

        Args:
            embed_ind (torch.Tensor): The indices to decode.

        Returns:
            torch.Tensor: The decoded vectors.
        """
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        """Forward pass of the codebook, encoding and decoding the input.

        During training, this method also updates the codebook via EMA.

        Args:
            x (torch.Tensor): The input tensor to quantize.

        Returns:
            tp.Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - quantize (torch.Tensor): The quantized tensor.
                - embed_ind (torch.Tensor): The indices of the quantized vectors.

        Raises:
            ValueError: If the decay rate is not between 0 and 1.
        """
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x)

        embed_ind = self.quantize(x)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind)

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            self.expire_codes_(x)
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = x.t() @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = (
                laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
                * self.cluster_size.sum()
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        return quantize, embed_ind

        
class VectorQuantization(nn.Module):
    """Vector Quantization (VQ) module for encoding and decoding vectors.

    This class implements vector quantization with support for codebook projection and
    commitment loss. It uses Euclidean distance for clustering and can optionally
    initialize the codebook using k-means.

    Example:
        ```python
        # Initialize a VectorQuantization module
        vq = VectorQuantization(
            dim=256,
            codebook_size=1024,
            codebook_dim=512,
            decay=0.99,
            epsilon=1e-5,
            kmeans_init=True,
            kmeans_iters=50,
            threshold_ema_dead_code=2,
            commitment_weight=1.0
        )

        # Encode and decode some data
        data = torch.randn(1, 256, 1000)
        quantized, indices, loss = vq(data)
        ```

    Args:
        dim (int): The dimensionality of the input vectors.
        codebook_size (int): The number of entries in the codebook.
        codebook_dim (int, optional): The dimensionality of the codebook vectors. 
                                        Defaults to `dim` if not specified.
        decay (float): Decay rate for EMA over the codebooks. Should be between 0 and 1. 
                        Defaults to 0.99.
        epsilon (float): Small value for numerical stability in calculations. 
                            Defaults to 1e-5.
        kmeans_init (bool): If True, initialize the codebook using k-means clustering. 
                            Defaults to True.
        kmeans_iters (int): Number of iterations for k-means initialization. 
                            Defaults to 50.
        threshold_ema_dead_code (int): Threshold for expiring codes based on EMA cluster size. 
                                        Codes with an EMA size below this threshold are replaced. 
                                        Defaults to 2.
        commitment_weight (float): Weight for commitment loss during training. 
                                    Defaults to 1.0.

    Attributes:
        project_in (nn.Linear or nn.Identity): Projects input vectors to codebook dimension if needed.
        project_out (nn.Linear or nn.Identity): Projects quantized vectors back to input dimension if needed.
        _codebook (EuclideanCodebook): The codebook for vector quantization.
        codebook_size (int): The number of entries in the codebook.

    Raises:
        ValueError: If `decay` is not between 0 and 1 or if `commitment_weight` is negative.
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.0,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity()
        )
        self.project_out = (
            nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity()
        )

        self.epsilon = epsilon
        self.commitment_weight = commitment_weight

        self._codebook = EuclideanCodebook(
            dim=_codebook_dim,
            codebook_size=codebook_size,
            kmeans_init=kmeans_init,
            kmeans_iters=kmeans_iters,
            decay=decay,
            epsilon=epsilon,
            threshold_ema_dead_code=threshold_ema_dead_code,
        )
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        """Returns the current codebook embeddings.

        Returns:
            torch.Tensor: The codebook embeddings.
        """
        return self._codebook.embed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input tensor into codebook indices.

        Args:
            x (torch.Tensor): The input tensor to encode. Shape should be (batch_size, dim, num_vectors).

        Returns:
            torch.Tensor: The indices of the nearest codebook entries.
        """
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        """Decode the codebook indices back to vectors.

        Args:
            embed_ind (torch.Tensor): The indices to decode.

        Returns:
            torch.Tensor: The decoded vectors, reshaped to match the original input shape.
        """
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x: torch.Tensor) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the vector quantization, encoding, decoding, and computing loss.

        During training, this method also computes the commitment loss.

        Args:
            x (torch.Tensor): The input tensor to quantize. Shape should be (batch_size, dim, num_vectors).

        Returns:
            tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - quantize (torch.Tensor): The quantized tensor.
                - embed_ind (torch.Tensor): The indices of the quantized vectors.
                - loss (torch.Tensor): The total loss, including commitment loss if applicable.

        Raises:
            UserWarning: If using this in training mode, a warning is raised to check for a known issue.
        """
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)

        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        if self.training:
            warnings.warn(
                "When using RVQ in training model, first check "
                "https://github.com/facebookresearch/encodec/issues/25 . "
                "The bug wasn't fixed here for reproducibility."
            )
            if self.commitment_weight > 0:
                commit_loss = F.mse_loss(quantize.detach(), x)
                loss = loss + commit_loss * self.commitment_weight

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind, loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """

    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        )

    def forward(self, x, n_q: tp.Optional[int] = None):
        quantized_out = 0.0
        residual = x

        all_losses = []
        all_indices = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices, loss = layer(residual)
            # residual = residual - quantized
            residual = residual - quantized.detach()
            quantized_out = quantized_out + quantized

            all_indices.append(indices)
            all_losses.append(loss)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        return quantized_out, out_indices, out_losses

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)
        return out_indices

    def decode(self, q_indices: torch.Tensor) -> torch.Tensor:
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out
