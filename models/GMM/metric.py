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
from sklearn.metrics import auc, roc_curve
from typing import Callable, List, Optional, Tuple, Union
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np

def calculate_eer(y, y_score) -> Tuple[float, float, np.ndarray, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y, -y_score)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return thresh, eer, fpr, tpr

# + tags=["active-ipynb"]
# y = np.random.randint(0, 2, 50)
# y_score = np.random.rand(50)
# calculate_eer(y, y_score)
