# %load_ext autoreload
# %autoreload 2

import pytorch_lightning as pl
import torch
import torch.nn as nn
from ay2.torch.deepfake_detection import DeepfakeAudioClassification


from .modules import AudioModel




class RVQ_lit(DeepfakeAudioClassification):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self.model = AudioModel()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=None)
        self.save_hyperparameters()
    
    
    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        
        # print(batch_res['vq_res'].penalty)
        
        
        cls_loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        
        vq_loss = batch_res['vq_res'].penalty
        
        if self.current_epoch <= 3:
            loss = vq_loss
        else:
            loss = cls_loss
                    
        return {"loss" : loss, "cls_loss": cls_loss, "vq_loss" : vq_loss}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=0.00001
        )
        return [optimizer]


    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]


        train_quantizer=True if self.current_epoch <= 3 else False
        res = self.model(audio, train_quantizer=train_quantizer)
        
        res['logit'] = res['logit'].squeeze(-1)
        res['batch_pred'] = (torch.sigmoid(res['logit']) + 0.5).int()
        
        
        return res

