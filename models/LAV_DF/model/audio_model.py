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

# +
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
# -

from .audio_encoder import get_audio_encoder
from .boundary_module import BoundaryModule
from .frame_classifier import FrameLogisticRegression
from .loss import MaskedFrameLoss, MaskedBMLoss, MaskedContrastLoss


class Batfd(torch.nn.Module):
    def __init__(
        self,
        # basis
        temporal_dim=800,  # max frames of the audio
        max_duration=40,  # max continuous deepfake frames
        # encoder
        a_encoder: str = "cnn",  # encoder type
        n_features=(32, 64, 64),  # features dims in encoder
        a_cla_feature_in=256,  # feature dim of the encoder output
        # boundary
        boundary_features=(512, 128),  # features dims in boundary module
        boundary_samples=10,
        **kwargs
    ):
        super().__init__()
        self.temporal_dim = temporal_dim

        self.audio_encoder = get_audio_encoder(
            a_cla_feature_in, temporal_dim, a_encoder, n_features
        )
        self.audio_frame_classifier = FrameLogisticRegression(
            n_features=a_cla_feature_in
        )

        a_bm_in = a_cla_feature_in + 1
        self.audio_boundary_module = BoundaryModule(
            a_bm_in, boundary_features, boundary_samples, temporal_dim, max_duration
        )
        self.save_hyperparameters()
        

    def forward(
        self, audio: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # encoders
        a_features = self.audio_encoder(audio)
        a_frame_cla = self.audio_frame_classifier(a_features)
        a_bm_in = torch.column_stack([a_features, a_frame_cla])
        a_bm_map = self.audio_boundary_module(a_bm_in)
        return (
            a_bm_map,
            a_frame_cla,
            a_features,
        )


class Batfd_Audio_lit(LightningModule):
    def __init__(
        self,
        # basis
        temporal_dim=800,  # max frames of the audio
        max_duration=64,  # max continuous deepfake frames
        # encoder
        a_encoder: str = "cnn",  # encoder type
        n_features=(32, 64, 64),  # features dims in encoder
        a_cla_feature_in=256,  # feature dim of the encoder output
        # boundary
        boundary_features=(512, 128),  # features dims in boundary module
        boundary_samples=10,
        # training settings
        weight_frame_loss=2.0,
        weight_modal_bm_loss=1.0,
        weight_decay=0.0001,
        learning_rate=0.0002,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.temporal_dim = temporal_dim
        self.max_duration = max_duration

        self.audio_model = Batfd(
            temporal_dim=temporal_dim,
            max_duration=max_duration,
            a_encoder=a_encoder,
            n_features=n_features,
            a_cla_feature_in=a_cla_feature_in,
            boundary_features=boundary_features,
            boundary_samples=boundary_samples,
        )

        self.weight_frame_loss = weight_frame_loss
        self.weight_modal_bm_loss = weight_modal_bm_loss
        self.frame_loss = MaskedFrameLoss(BCEWithLogitsLoss())
        self.bm_loss = MaskedBMLoss(MSELoss())

        
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        optimizer = Adam(
            self.audio_model.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.9),
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-8
                ),
                "monitor": "val_loss",
            },
        }

    def loss_fn(
        self,
        a_bm_map: Tensor,
        a_frame_cla: Tensor,
        n_frames: Tensor,
        a_bm_label,
        a_frame_label,
    ) -> Dict[str, Tensor]:

        a_bm_loss = self.bm_loss(a_bm_map, a_bm_label, n_frames)
        a_frame_loss = self.frame_loss(a_frame_cla.squeeze(1), a_frame_label, n_frames)

        loss = (
            self.weight_modal_bm_loss * a_bm_loss
            + self.weight_frame_loss * a_frame_loss
        )

        return {
            "loss": loss,
            "a_bm_loss": a_bm_loss,
            "a_frame_loss": a_frame_loss,
        }

    def shared_evaluate_step(self, batch, batch_idx, prefix=""):
        a_bm_map, a_frame_cla, a_features = self.audio_model(batch["audio"])
        # print(batch['audio'].shape, batch['bm_label'].shape, a_bm_map.shape, a_frame_cla.shape)
        loss_dict = self.loss_fn(
            a_bm_map=a_bm_map,
            a_frame_cla=a_frame_cla,
            n_frames=batch['frames'],
            a_bm_label=batch["bm_label"],
            a_frame_label=batch["frame_label"],
        )

        self.log_dict(
            {f"{prefix}_{k}": v for k, v in loss_dict.items()},
            on_step=True if prefix == 'train' else False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        res = loss_dict
        res["bm_map"] = a_bm_map
        res["frame_label"] = a_frame_cla
        res['frames'] = batch['frames']
        return res

    def training_step(
        self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
    ) -> Tensor:
        res = self.shared_evaluate_step(batch, batch_idx, prefix="train")
        return res

    def validation_step(
        self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Tensor:
        res = self.shared_evaluate_step(batch, batch_idx, prefix="val")
        return res

    def test_step(
        self,
        batch: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None,
    ) -> Tensor:
        res = self.shared_evaluate_step(batch, batch_idx, prefix="test")
        return res

    def predict_step(
        self, batch: Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tensor:
        res = self.shared_evaluate_step(batch, batch_idx, prefix="predict")
        return res

# + tags=["active-ipynb"] editable=true slideshow={"slide_type": ""}
# audio = torch.randn(1, 64, 3200)
#
# model = Batfd()
# a_bm_map, a_frame_cla, a_features = model(audio)
#
# a_bm_map.shape, a_frame_cla.shape, a_features.shape
# -




