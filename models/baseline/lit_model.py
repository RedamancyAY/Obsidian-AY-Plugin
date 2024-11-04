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

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy

# + tags=["active-ipynb", "style-commentate"]
# from wavlm import BaseLine
# -

from .wavlm import BaseLine

from ay2.torch.metrics.equal_error_rate import EER


class BaseLine_lit(pl.LightningModule):
    def __init__(self, pretrain_feat="last_hidden_state", backend='resnet'):
        super().__init__()
        self.model = BaseLine(pretrain_feat=pretrain_feat, backend=backend)

        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=None)
        self.acc_train, self.acc_val, self.acc_test = [
            BinaryAccuracy() for i in range(3)
        ]
        self.eer_train, self.eer_val, self.eer_test = [EER() for i in range(3)]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=0.0001
        )
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        # audio, sample_rate, label = batch
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        batch_out = self.model(audio).squeeze()
        batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
        return batch_out, batch_pred

    def _shared_eval_step(self, batch, batch_idx, metric_acc, metric_eer):
        batch_out, batch_pred = self._shared_pred(batch, batch_idx)

        label = batch["label"]
        loss = self.loss_fn(batch_out, label.type(torch.float32))
        metric_acc.update(batch_pred, label)
        metric_eer.update(batch_out, label)

        self.log_dict(
            {"loss": loss},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, self.acc_train, self.eer_train)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, self.acc_val, self.eer_val)

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, batch_idx, self.acc_test, self.eer_test)

    def on_train_epoch_end(self):
        res = {}
        res["train_acc"] = self.acc_train.compute()
        res["train_eer"] = self.eer_train.compute()
        self.log_dict(res, logger=True, prog_bar=True)

    def on_validation_epoch_end(self):
        res = {}
        res["val_acc"] = self.acc_val.compute()
        res["val_eer"] = self.eer_val.compute()
        self.log_dict(res, logger=True, prog_bar=True)

    def on_test_epoch_end(self):
        res = {}
        res["test_acc"] = self.acc_test.compute()
        res["test_eer"] = self.eer_test.compute()
        self.log_dict(res, logger=True, prog_bar=True)

    # predict scores
    #     obtain all batch results and write them into a txt file

    def predict_step(self, batch, batch_idx):
        batch_out, batch_pred = self._shared_pred(batch, batch_idx)
        names = batch["name"]
        self.predict_outputs.append((batch_out, names))

    def on_predict_start(self):
        self.predict_outputs = []

    def on_predict_end(self, *arg, **kwargs):
        scores, counts = {}, {}
        N = 0
        for y, filename in self.predict_outputs:
            for i in range(y.shape[0]):
                if filename[i] in scores.keys():
                    scores[filename[i]] += y[i]
                    counts[filename[i]] += 1
                else:
                    scores[filename[i]] = y[i]
                    counts[filename[i]] = 1
                N += 1

        print("Predict end: %d audio clips -> %d total audios" % (N, len(scores)))

        with open("scores.txt", "w") as f:
            for filename in scores.keys():
                f.write(
                    "%s %f\n" % (filename, scores[filename] / counts[filename])
                )

            # for y, filename in self.predict_outputs:
            #     for i in range(y.shape[0]):
            #         # f.write("%s %f\n" % (filename[i], y[i]))
            #         f.write(
            #             "%s %f\n"
            #             % (filename[i], scores[filename[i]] / counts[filename[i]])
            #         )
        del self.predict_outputs

