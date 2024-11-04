# %load_ext autoreload
# %autoreload 2

# +
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from ay2.torch.deepfake_detection import DeepfakeAudioClassification
from einops import rearrange
from torchaudio.transforms import LFCC, Spectrogram
# -


try:
    from grl import GradientReversal
    from lcnn import LCNN
    from leaf_audio.frontend import Leaf
    from pooling import AttentiveStatisticsPooling1D
except ImportError:
    from .grl import GradientReversal
    from .lcnn import LCNN
    from .leaf_audio.frontend import Leaf
    from .pooling import AttentiveStatisticsPooling1D


# ## Sepctrogram Feat Extraction Module

# this setting can get => (B, 1, n_LFCC, 300)
# ```python
# LFCC(
#     n_lfcc=n_LFCC,
#     speckwargs={"n_fft": 400, "hop_length": 160, "center": True},
# )
# ```
# this setting can get => (B, 1, 80, 300)
# ```python
# Spectrogram(n_fft=160, hop_length=160)
# ```

class SpectrogramExtraction(nn.Module):
    def __init__(self, spec_type="log", n_LFCC=80):
        super().__init__()

        self.spec_type = spec_type

        if self.spec_type.lower() == "lfcc":
            self.lfcc = LFCC(
                n_lfcc=n_LFCC,
                speckwargs={"n_fft": 512, "hop_length": 187, "center": True},
            )
        elif self.spec_type.lower() == "log":
            self.spectrogram = Spectrogram(n_fft=512, hop_length=187)
        elif self.spec_type.lower() == "leaf":
            self.leaf_audio = Leaf(n_filters=n_LFCC)

    def extract_log_spec(self, x):
        x = self.spectrogram(x)
        x = torch.log(x + 1e-7)
        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9)
        return x

    def forward(self, x):
        if self.spec_type.lower() == "lfcc":
            spec = self.lfcc(x)  # (B, 1, 80, 301)
        elif self.spec_type.lower() == "log":
            spec = self.extract_log_spec(x)
        elif self.spec_type.lower() == "leaf":
            spec = self.leaf_audio(x)

        return spec


# ### 测试

# +
# x = torch.randn(2, 1, 48000)
# for _t in ["log", "lfcc", "leaf"]:
#     m = SpectrogramExtraction(spec_type=_t, n_LFCC=257)
#     y = m(x)
#     print(_t, y.shape)
# -

# ## LSTM

class BLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.drop = nn.Dropout2d(0.1)
        self.rnn = nn.LSTM(input_size, input_size // 2, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.drop(x)
        x = rearrange(x, "b c h w -> b h (c w)")
        output, _ = self.rnn(x)
        return output + x


# ## Pooling Layer

class PoolingLayer(nn.Module):
    """Apply attentive pooling or normal average pooling

    For attentive pooling, this module will calculate the mean and std values, concat them, and project the results;
    For non-attentive pooling, this module will calculate the average pooling, i.e., average the tensor at `dim=1`

    Attributes:
        is_use_attn_pool: whether use the attentive pooling
        pool: the attentive pool layer
        fc1: the final projection layer
    """

    def __init__(self, is_use_attn_pool, t_dim, final_dim=512):
        """Initializes the instance based on given parameters.

        Args:
            is_use_attn_pool: whether use the attentive statistics pooling, which both
                calculate the mean and std values. If `False`, use average pooling.
            t_dim: the final temporal dim of input (B, C, T)
            final_dim: the final dim of the returned output result (B, final_dim).
        """
        super().__init__()

        self.is_use_attn_pool = is_use_attn_pool

        if is_use_attn_pool:
            self.pool = AttentiveStatisticsPooling1D(t_dim, t_dim * 2, 1)
            self.fc1 = nn.Linear(t_dim * 2, final_dim)
        else:
            self.pool = None
            self.fc1 = nn.Linear(t_dim, final_dim)

    def forward(self, x):
        """
        the input is with size of (B, C, T)

        Args:
            x: the 3D input that is with size of (B, C, T)

        Returns:

            a 2D tensor with shape of (B, `self.final_dim`)

        """
        if self.is_use_attn_pool:
            x = self.pool(x)
        else:
            x = torch.mean(x, dim=1)

        x = self.fc1(x)
        return x


# +
# x = torch.randn(2, 128, 400)
# m = PoolingLayer(is_use_attn_pool=True, t_dim=400)
# m(x).shape
# -

# ## 模型

# ```mermaid
# flowchart LR
#
# A[Input X] -->|Transform| B(Spectrogram)
# B --> C(LCNN)
# C --> D(Bi-LSTM)
# D --> E(Pooling)
# E --> F(Classification)
# ```

# +
# lcnn = LCNN()

# x = torch.randn(2, 1, 257, 257)
# y = lcnn.extract_feature(x, debug=True)
# print(y.shape)
# lstm = BLSTM(32 * 16)
# lstm(y)
# -

class OursLCNN(nn.Module):
    def __init__(self, n_LFCC=80, is_use_attn_pool=False):
        super().__init__()


        self.spec_extractor = SpectrogramExtraction(spec_type='log')
        self.lcnn = LCNN()
        self.lstm = BLSTM(512)
        self.pooling_layer = PoolingLayer(is_use_attn_pool=is_use_attn_pool, t_dim=512, final_dim=512)

        self.fc2 = nn.utils.parametrizations.weight_norm(nn.Linear(512, 1))
        self.grl = GradientReversal()
        self.fc_vocoder = nn.utils.parametrizations.weight_norm(nn.Linear(512, 32))
        self.fc_domain = nn.utils.parametrizations.weight_norm(nn.Linear(512, 3))

    def extract_feature(self, x, debug=0):
        spec_feat = self.spec_extractor(x) # (B, 1, H, W) where H is the frequency dim

        x = self.lcnn.extract_feature(spec_feat, debug=debug)  # (B, 32, H//16, W//16)
        x = self.lstm(x) 
        feat = self.pooling_layer(x) # (B, 512)
        feat = feat / (1e-9 + torch.norm(feat, p=2, dim=-1, keepdim=True))
        return feat

    def make_prediction(self, feat):
        logit = self.fc2(feat)
        return logit

    def gene_grl_alpha(self, current_step, total_step):
        a = 1 - 2 / (1 + math.exp(-10 * current_step / total_step))
        return torch.tensor(a)

    def predict_domain_label(self, feat, current_step=0, total_step=1, use_grl=0):
        if use_grl:
            alpha = self.gene_grl_alpha(current_step, total_step)
            feat_rev = self.grl(feat, alpha=alpha)
            logit = self.fc_domain(feat_rev)
        else:
            logit = self.fc_domain(feat)
        return logit

    def predict_vocoder_label(self, feat):
        logit = self.fc_vocoder(feat)
        return logit

    def forward(self, x, debug=0):
        feat = self.extract_feature(x, debug=debug)
        logit = self.fc2(feat)
        return logit

model = OursLCNN(is_use_attn_pool=1)
x = torch.randn(2, 1, 16000 * 3)
model(x)


class OursLCNN_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = OursLCNN(n_LFCC=80)

        self.configure_loss_fn()

        self.save_hyperparameters()

    def configure_loss_fn(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=0.1, p=2, eps=1e-7)

    def get_domain_label(self, batch):
        domains = ["en", "de", "es"]
        batch["ASGD_domain_label"] = []
        for i in range(len(batch["label"])):
            try:
                _label = batch["language"][i]
            except KeyError:
                _label = domains[0]
            if _label in domains:
                _label = domains.index(_label)
            else:
                _label = 0
            batch["ASGD_domain_label"].append(_label)
        batch["ASGD_domain_label"] = torch.tensor(batch["ASGD_domain_label"]).to(batch["label"].device)

    def generate_pairs(self, features, labels):
        """
        Generate positive and negative pairs based on class labels.

        Args:
        - features (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing the batch of features.
        - labels (torch.Tensor): A tensor of shape (batch_size,) representing the class labels for each feature.

        Returns:
        - pos_pairs (list of tuples): List of positive pairs (anchor_idx, pos_idx).
        - neg_pairs (list of tuples): List of negative pairs (anchor_idx, neg_idx).
        """
        pos_pairs = []
        neg_pairs = []
        batch_size = features.size(0)

        for anchor_idx in range(batch_size):
            anchor_label = labels[anchor_idx]

            find_pos = 0
            find_neg = 0

            for pair_idx in np.random.permutation(np.arange(batch_size)):
                if anchor_idx == pair_idx:
                    continue  # Skip the same element

                pair_label = labels[pair_idx]

                if anchor_label == pair_label:
                    if not find_pos:
                        pos_pairs.append(features[pair_idx])
                        find_pos = 1
                else:
                    if not find_neg:
                        neg_pairs.append(features[pair_idx])
                        find_neg = 1
                if find_pos and find_neg:
                    break
            if not find_pos:
                pos_pairs.append(features[anchor_idx])
            if not find_neg:
                neg_pairs.append(features[anchor_idx])

        pos_pairs = torch.stack(pos_pairs)
        neg_pairs = torch.stack(neg_pairs)

        # print(pos_pairs.shape, neg_pairs.shape)

        return pos_pairs, neg_pairs

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        batch_size = len(label)
        cls_loss = self.bce_loss(batch_res["logit"], label.type(torch.float32))

        # idx = torch.nonzero(label == 1).squeeze().cpu()
        # if len(idx) == 0:
        #     adv_loss = 0
        # else:
        #     adv_loss = self.ce_loss(batch_res["domain_logit"][idx], batch["vocoder_label"][idx])
        # adv_loss = 0

        vocoder_cls_loss = self.ce_loss(batch_res["vocoder_logit"], batch["vocoder_label"])
        domain_cls_loss = self.ce_loss(batch_res["domain_logit"], batch["ASGD_domain_label"])
        # idx = torch.nonzero(label == 1).squeeze().cpu()
        # if len(idx) == 0:
        #     adv_loss = 0
        # else:
        #     adv_loss = self.ce_loss(batch_res["domain_logit"][idx], batch["vocoder_label"][idx])
        # adv_loss = 0

        ### triplet loss for real and fake samples
        anchor = batch_res["feature"]
        triplet_label = []
        for i in range(batch_size):
            _label = 3 if label[i] == 0 else batch["ASGD_domain_label"][i]
            # _label = batch["ASGD_domain_label"][i]
            triplet_label.append(_label)
        pos_pairs, neg_pairs = self.generate_pairs(anchor, triplet_label)
        triplet_loss = self.triplet_loss(anchor, pos_pairs, neg_pairs)

        loss = cls_loss + 0.3 * (vocoder_cls_loss + domain_cls_loss + triplet_loss)
        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "vocoder_cls_loss": vocoder_cls_loss,
            "domain_cls_loss": domain_cls_loss,
            "triplet_loss": triplet_loss,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        self.num_training_batches = self.trainer.num_training_batches
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        self.get_domain_label(batch)

        # batch_out = self.model(audio).squeeze()
        feature = self.model.extract_feature(audio)
        batch_out = self.model.make_prediction(feature).squeeze(-1)
        batch_pred = (torch.sigmoid(batch_out) + 0.5).int()

        current_step = self.global_step
        try:
            total_step = self.num_training_batches * 64 * 50
        except AttributeError:
            total_step = 1000
        domain_logit = self.model.predict_domain_label(feature, current_step, total_step, use_grl=0)
        vocoder_logit = self.model.predict_vocoder_label(feature)

        return {
            "logit": batch_out,
            "domain_logit": domain_logit,
            "vocoder_logit": vocoder_logit,
            "pred": batch_pred,
            "feature": feature,
        }
