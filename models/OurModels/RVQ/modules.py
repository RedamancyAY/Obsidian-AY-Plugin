# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: torch
#     language: python
#     name: python3
# ---

# +
import os
import torch
import numpy as np
import torch.nn as nn
import time

from torch import Tensor
# -

from transformers import HubertModel, Wav2Vec2FeatureExtractor
try:
    from .quantization import ResidualVectorQuantizer, QuantizedResult
except ImportError:
    from quantization import ResidualVectorQuantizer, QuantizedResult


# When using RVQ in training model, first check https://github.com/facebookresearch/encodec/issues/25 
#
# <center><img src="https://cdn.jsdelivr.net/gh/RedamancyAY/CloudImage@main/img20241209231215699.png" width="400" alt="$fileName"/></center>
#
#

class AudioQuantization(torch.nn.Module):

    def __init__(self, sample_rate=16000, *args, **kwargs):
        super().__init__(*args, **kwargs)

        model_name = "facebook/hubert-base-ls960"
        self.model = HubertModel.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        self.dim = self.model.config.hidden_size
        self.RVQ = ResidualVectorQuantizer(
            dimension=self.dim, bins=100, n_q=8
        )
        self.frame_rate = sample_rate / np.prod(self.model.config.conv_stride)

    def forward(self, x) -> QuantizedResult:

        with torch.no_grad():
            y = self.model(x, output_hidden_states=True)
            hidden_states = y.hidden_states

        # feat is with shape of (B, T, C)
        feat = hidden_states[9]

        ## rvq require the input be shape of (B, C, T)
        vq_res = self.RVQ(feat.transpose(2, 1), frame_rate=self.frame_rate)


        ## vq_res.codes is with shape of (n_q, B, T), we only use the last quantizer
        ## thus, the output if with shape of (B, T)
        vq_res.codes = vq_res.codes[-1]

        ## the shape of quantized is (B, C, T)
        # quantized = vq_res.quantized
        
    
        return vq_res


# + tags=["active-ipynb"]
# quantization_model = AudioQuantization()
# x = torch.randn(16, 48000)
# res = quantization_model(x)

# +
def segment_means(x: torch.Tensor, segment_sizes: torch.Tensor) -> torch.Tensor:
    """
    Args:
      tensor: torch.Tensor: a 2D tensor with shape of `(L, C)`
      segment_sizes: torch.Tensor: a 1D tensor that its sum is equal to the length `L` of tensor

    Returns:
        torch.Tenosr: the tensor with reduce length `(L', C)`, where $L'=len(segment_sizes)$
    """
    assert x.size(0) == segment_sizes.sum(), "Sum of segment sizes must equal the tensor's first dimension size."

    # Create an indices tensor that maps each row in the tensor to its corresponding segment
    indices = torch.repeat_interleave(torch.arange(len(segment_sizes), device=x.device), segment_sizes)

    # Create a tensor to hold the sum of each segment
    segment_sums = torch.zeros(len(segment_sizes), x.size(1), device=x.device)

    # Scatter and sum the inputs into the segment_sums tensor
    segment_sums.scatter_add_(0, indices.unsqueeze(1).expand(-1, x.size(1)), x)

    # Calculate the mean of each segment
    _segment_means = segment_sums / segment_sizes.unsqueeze(1)

    return _segment_means


def reduce_feat_by_phonemes(
    hidden_states: Tensor, audio_lengths: Tensor, phoneme_ids: Tensor, debug: bool = False
) -> Tensor:
    """
    For each audio, combine continuous phonemes to reduce the temporal dimension of audio features.
    For example, the audio with 10 frames, which phoneme ids will change from
    ```python
    [0, 0, 0, 1, 1, 1, 2, 2, 0, 0] -> [0, 1, 2, 0]
    ```
    and the hidden_states will also changed by this way.


    Args:
      hidden_states:Tensor: a 3D tensor with shape of (B, T, C), where T is audio frames
      audio_lengths:Tensor: a 1D tensor with shape of (B,) that represent the legnth of each audio
      phoneme_ids:Tensor: a 2D tensor with shape of (B, T) that represents the phoneme ids in each audio frame
      debug:bool: determine whether to debug tensor info.

    Returns:
        the reduced hidden_states with shape (B*L', C), reduced audio lengths, reduced phoneme ids.
    """

    reduced_hidden_states = []
    reduced_audio_lengths = []
    reduced_phoneme_ids = []
    phoneme_counts = []

    if debug:
        s = time.time()
        print("reduce feat input:", hidden_states.shape, audio_lengths.shape, phoneme_ids.shape)

    for i in range(len(audio_lengths)):
        _phoneme_ids = phoneme_ids[i, : audio_lengths[i]]
        unique_ids, _phoneme_counts = _phoneme_ids.unique_consecutive(return_counts=True)
        phoneme_counts += _phoneme_counts.tolist()

        reduced_audio_lengths.append(len(unique_ids))
        reduced_phoneme_ids.append(unique_ids)

    reduced_audio_lengths = torch.tensor(reduced_audio_lengths)
    reduced_phoneme_ids = torch.nn.utils.rnn.pad_sequence(reduced_phoneme_ids, batch_first=True)
    h = torch.concat([hidden_states[i, :_len, :] for i, _len in enumerate(audio_lengths)], dim=0)
    reduced_hidden_states = segment_means(h, torch.tensor(phoneme_counts, device=hidden_states.device))

    if debug:
        e = time.time()
        print(
            "reduce feat output:", reduced_hidden_states.shape, reduced_audio_lengths.shape, reduced_phoneme_ids.shape
        )
        print("reduce feat time:", e - s)
    return reduced_hidden_states, reduced_audio_lengths, reduced_phoneme_ids

def get_id_based_frame_res(_dense_res, _full_unit_res):
    reduced_hidden_states, reduced_audio_lengths, reduced_phoneme_ids = reduce_feat_by_phonemes(
        hidden_states=torch.tensor(_dense_res),
        audio_lengths=torch.tensor([149] * len(_dense_res)),
        phoneme_ids=torch.tensor(_full_unit_res),
        debug=0,
    )
    print(reduced_audio_lengths)

    split_res = torch.split(reduced_hidden_states, tuple(reduced_audio_lengths.numpy()))

    id_based_frame_res = torch.stack([x.mean(0) for x in split_res])
    return id_based_frame_res


# +
# get_id_based_frame_res(res.quantized, res.codes)
# -

# https://github.com/huggingface/transformers/blob/d363e71d0e32f44d7a5b3571d4921371907bd0ee/src/transformers/models/hubert/modeling_hubert.py#L945
#
#
# ```python
# layers = quantization_model.model.encoder.layers
# hidden_states = res.quantized.transpose(1, 2) # (B, T, C) -> (B, C, T)
# for layer in layers[9:12]:
#     layer_outputs = layer(hidden_states)
#     hidden_states = layer_outputs[0]
# ```

class LinearClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Define a linear layer with input dimension `input_dim` and output dimension `1`
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, vq_res: QuantizedResult):

        codes = vq_res.codes
        quantized = vq_res.quantized

        # Pass the input through the linear layer
        feat = quantized.mean(-1)
        output = self.classifier(feat)
        return output


class AudioModel(nn.Module):
    
    def __init__(self, ):
        super().__init__()
        
        self.quantizer = AudioQuantization()
        self.classifier = LinearClassifier(input_dim=self.quantizer.dim)
        
    
    def forward(self, x: torch.Tensor, train_quantizer=True):
        
        if x.ndim == 3 and x.size(1) == 1:
            x = x[:, 0, :]
        
        
        if train_quantizer:
            vq_res = self.quantizer(x) # (B, C, T)
        else:
            with torch.no_grad():
                vq_res = self.quantizer(x) # (B, C, T)
        codes = vq_res.codes # (B, T)
        
        
        layers = self.quantizer.model.encoder.layers
        hidden_states = vq_res.quantized.transpose(1, 2) # (B, C, T) -> (B, T, C)
        
        
        ## use transformer layers to process the hidden states
        ## The final hidden states is with shape of (B, T, C)
        for layer in layers[9:12]:
            layer_outputs = layer(hidden_states)
            hidden_states = layer_outputs[0]
        
        
        
        # feat = hidden_states.mean(1)
        feat = get_id_based_frame_res(hidden_states, codes)
        
        logit = self.classifier.classifier(feat)
        
        # logit = self.classifier(vq_res)
        
        return {"logit" : logit, "vq_res":vq_res}



