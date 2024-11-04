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
# # %load_ext autoreload
# # %autoreload 2

# +
import pytorch_lightning as pl
import torch

from .gaussian_mixture_model import GMMDescent
from .metric import calculate_eer


# -

class GMM(pl.LightningModule):
    def __init__(
        self, clusters, inital_data
    ):
        super().__init__()
        self.model = GMMDescent(clusters, inital_data, covariance_type="diag")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return [optimizer]


    def _shared_eval_step(self, batch, batch_idx):
        audio, sample_rate, label = batch
        
        pred = self.model(audio)
        loss = - pred.mean()
                
        self.log_dict(
            {'loss', loss},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)
    
    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)


class GMMTest(pl.LightningModule):
    def __init__(
        self, real_model, fake_model
    ):
        super().__init__()
        self.real_model = real_model
        self.fake_model = fake_model
        
        
    def _shared_eval_step(self, batch, batch_idx):
        audio, sample_rate, label = batch
        
        score = real_model(data).mean() - fake_model(data).mean()
        self.scores.append(score)
        self.labels.append(labels)

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx)
    
    def on_test_epoch_start(self):
        self.scores = []    
        self.labels = []
        
    def on_test_epoch_end(self):
        scores = torch.concat(self.scores)
        labels = torch.concat(self.labels)

        thresh, eer, fpr, tpr = calculate_eer(labels, scores)
        self.log_dict({'Thresh':thresh, 'EER': eer, 'FPR': fpr, 'TPR': tpr}, logger=True, prog_bar=True)
