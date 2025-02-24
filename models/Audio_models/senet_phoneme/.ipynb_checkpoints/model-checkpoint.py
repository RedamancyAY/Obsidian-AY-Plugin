# %load_ext autoreload
# %autoreload 2

import torch

from ay2.torchaudio.transforms import CQCC

try:
    from senet.se_resnet import se_resnet50
except ImportError:
    from .senet.se_resnet import se_resnet50

model = se_resnet50()

x = torch.randn(2, 3, 224, 224)
model(x)

cqcc = CQCC()

x = torch.randn(48000, 1).numpy()
feat = cqcc(x)

feat.shape




