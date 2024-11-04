class WaveLM_lit(DeepfakeAudioClassification):
    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.model = WaveLM(num_classes=13)
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=0.0001
        )
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        out = self.model(audio)
        return {"logit": out}