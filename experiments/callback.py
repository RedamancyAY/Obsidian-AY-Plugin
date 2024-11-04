from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_info
from torch import Tensor, tensor
from torchmetrics import Metric
from Levenshtein import distance


class PhonemeErrorRate(Metric):
    """
    https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/text/wer.py#L23-L93
    """

    def __init__(self):
        super().__init__()
        self.add_state("errors", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds, targets):
        """
        preds : list of sentence phoneme
        targets : list of sentence phoneme
        """
        errors, total = _per_update(preds, targets)

        self.errors += errors
        self.total += total

    def compute(self):
        return _per_compute(self.errors, self.total)


def _per_update(
    preds: Union[str, List[str]],
    target: Union[str, List[str]],
) -> Tuple[Tensor, Tensor]:
    """Update the wer score with the current set of references and predictions.
    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings
    Returns:
        Number of edit operations to get from the reference to the prediction, summed over all samples
        Number of words overall references
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    errors = tensor(0, dtype=torch.float)
    total = tensor(0, dtype=torch.float)
    for pred, tgt in zip(preds, target):
        if isinstance(pred, str):
            pred_tokens = pred.split()
            tgt_tokens = tgt.split()
        else:
            pred_tokens = pred
            tgt_tokens = tgt
            
        # errors += _edit_distance(pred_tokens, tgt_tokens)
        errors += distance(pred_tokens, tgt_tokens)
        total += len(tgt_tokens)
    return errors, total


def _per_compute(errors: Tensor, total: Tensor) -> Tensor:
    """Compute the word error rate.
    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        total: Number of words overall references
    Returns:
        Word error rate score
    """
    return errors / total

from numba import njit

# @njit
def _edit_distance(prediction_tokens: List[str], reference_tokens: List[str]) -> int:
    """Standard dynamic programming algorithm to compute the edit distance.
    Args:
        prediction_tokens: A tokenized predicted sentence
        reference_tokens: A tokenized reference sentence
    Returns:
        Edit distance between the predicted sentence and the reference sentence
    """
    # print(prediction_tokens, reference_tokens)
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(prediction_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


class MetricsModule:
    def __init__(self, set_name, device) -> None:
        """
        set_name: val/train/test
        """
        self.device = device
        dict_metrics = {}
        dict_metrics["per"] = PhonemeErrorRate().to(device)

        self.dict_metrics = dict_metrics

    def update_metrics(self, x, y):
        for _, m in self.dict_metrics.items():
            # metric on current batch
            m(x, y)  # update metrics (torchmetrics method)

    def log_metrics(self, name, pl_module):
        for k, m in self.dict_metrics.items():
            # metric on all batches using custom accumulation
            metric = m.compute()
            pl_module.log(name + k, metric)

            # Reseting internal state such that metric ready for new data
            m.reset()
            m.to(self.device)


class LogMetricsCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        device = pl_module.device

        self.metrics_module_train = MetricsModule("train", device)
        self.metrics_module_validation = MetricsModule("val", device)

    def on_test_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        device = pl_module.device

        self.metrics_module_test = MetricsModule("test", device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the train batch ends."""

        self.metrics_module_train.update_metrics(outputs["preds"], outputs["targets"])

    def on_train_epoch_start(self, trainer, pl_module):
        """Called when the train epoch ends."""
        device = pl_module.device
        self.metrics_module_train = MetricsModule("train", device)
        self.metrics_module_validation = MetricsModule("val", device)
        
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends."""

        self.metrics_module_train.log_metrics("train-", pl_module)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""

        self.metrics_module_validation.update_metrics(outputs["preds"], outputs["targets"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        self.metrics_module_validation.log_metrics("val-", pl_module)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Called when the validation batch ends."""

        self.metrics_module_test.update_metrics(outputs["preds"], outputs["targets"])

    def on_test_epoch_end(self, trainer, pl_module):
        """Called when the validation epoch ends."""

        self.metrics_module_test.log_metrics("test-", pl_module)
