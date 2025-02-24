# +
import os
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
)
# -

# # Utils

# +
from transformers import Wav2Vec2ForCTC, WavLMForCTC


class BaseModel(nn.Module):
    """
    BaseFeaturesExtractor class that will extract features according to the type of model
    https://huggingface.co/blog/fine-tune-wav2vec2-english
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        outputs = self.model(x)
        return outputs


class Wav2Vec2(BaseModel):
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC
    """

    def __init__(self, params):
        super().__init__(params)

        self.model = Wav2Vec2ForCTC.from_pretrained(params.pretrained_name)
        in_features = self.model.lm_head.in_features
        self.model.lm_head = nn.Linear(in_features=in_features, out_features=self.params.vocab_size)


class WavLM(BaseModel):
    """
    https://huggingface.co/docs/transformers/model_doc/wavlm#transformers.WavLMForCTC
    """

    def __init__(self, params):
        super().__init__(params)
        self.model = WavLMForCTC.from_pretrained(params.pretrained_name)
        in_features = self.model.lm_head.in_features
        self.model.lm_head = nn.Linear(in_features=in_features, out_features=self.params.vocab_size)
        print(self.model.lm_head.weight.shape)


network_param = Namespace(
    network_name="WavLM",
    # pretrained_name="microsoft/wavlm-base",
    pretrained_name="facebook/wav2vec2-base-960h",
    freeze=False,
    freeze_transformer=False,
    eos_token="</s>",
    bos_token="<s>",
    unk_token="<unk>",
    pad_token="<pad>",
    word_delimiter_token="|",
    vocab_size=200,
)
optim_param = Namespace(
    optimizer="AdamW",
    lr=2e-5,
    weight_decay=1e-08,
    accumulate_grad_batches=16,
    use_scheduler=1,
    max_epochs=10,
    warmup_epochs=1,
    warmup_start_lr=1e-5,
    eta_min=5e-06,
    step_size=2,
    gamma=0.1,
    milestones=[8, 10, 15],
    min_lr=5e-09,
    patience=10,
)


# +
# model = WavLM(network_param)
# -

# # Model

class BaseModule(LightningModule):
    def __init__(self, network_param, optim_param, tokenizer=None):
        """
        method used to define our model parameters
        """
        super().__init__()
        self.tokenizer = tokenizer
        network_param.vocab_size = tokenizer.total_num_phonemes
        # Optimizer
        self.optim_param = optim_param
        self.lr = optim_param.lr

        # network_param.vocab_size = self.phonemes_tokenizer.vocab_size

        # Loss function
        self.loss = nn.CTCLoss(blank=0)

        #  Model
        # self.model = WavLM(network_param)
        self.model = Wav2Vec2(network_param)
        if network_param.freeze:
            self.model.model.freeze_feature_encoder()

        if network_param.freeze_transformer:
            self.model.model.requires_grad_(False)
            self.model.model.lm_head.requires_grad_(True)

        self.save_hyperparameters()

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        loss, logits, preds, targets = self._get_outputs(batch, batch_idx)

        # print(self.optimizers().param_groups[0]['lr'])

        if loss != loss:
            print("loss is nan, model collapse, exiting")
            # exit(1)
            loss = torch.mean(logits)
            print(logits, targets, log_probs)
            exit(1)

        # Log loss
        self.log(
            "train/loss",
            loss,
            batch_size=len(preds),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        # if batch_idx %200 ==0:
        #     for pred, target in zip(preds, targets):
        #         print("pred", pred, "target", target)

        return {
            "loss": loss,
            "logits": logits.detach(),
            "preds": preds,
            "targets": targets,
        }

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits, preds, targets = self._get_outputs(batch, batch_idx)
        self.log("val/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        # for pred, target in zip(preds, targets):
        # print("pred", pred, "target", target)

        return {"loss": loss, "logits": logits, "preds": preds, "targets": targets}

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits, preds, targets = self._get_outputs(batch, batch_idx)
        self.log("test/loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return {"loss": loss, "logits": logits, "preds": preds, "targets": targets}

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim, self.optim_param.optimizer)
        optimizer = optimizer(self.parameters(), lr=self.lr, weight_decay=self.optim_param.weight_decay)

        if self.optim_param.use_scheduler:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.optim_param.warmup_epochs,
                max_epochs=self.optim_param.max_epochs,
                warmup_start_lr=self.optim_param.warmup_start_lr,
                eta_min=self.optim_param.eta_min,
            )
            return [[optimizer], [scheduler]]
        return optimizer

    def _get_outputs(self, batch, batch_idx):
        """convenience function since train/valid/test steps are similar"""
        x = batch

        # print(batch_idx, "lm head weight", self.model.model.lm_head.weight)
        # print(batch_idx,x["audio"].shape, x["audio_length"] ,x["phoneme_id_length"])

        # x['array'] gives the actual raw audio
        if x["audio"].ndim == 3:
            x["audio"] = x["audio"][:, 0, :]

        try:
            y = self.model.model(x["audio"], output_hidden_states=True)
            output = y.logits  # (B, T, C)
        except torch.cuda.OutOfMemoryError as e:
            print("torch.cuda.OutOfMemoryError, ", x["audio"].shape)
            raise e

        # process outputs
        log_probs = F.log_softmax(output, dim=-1)
        # input_lengths = torch.LongTensor([len(b) for b in log_probs])
        log_probs = log_probs.permute(1, 0, 2)  # (T, B, C)

        input_lengths = torch.LongTensor([min(b // 320, output.size(1)) for b in x["audio_length"]])

        # print(output.shape, log_probs.shape, input_lengths.shape, input_lengths, x["audio_length"])

        targets = x["phoneme_id"]
        target_lengths = x["phoneme_id_length"]

        loss = self.loss(log_probs, targets, input_lengths, target_lengths)

        # if loss < 0:
        # print(
        #     y.hidden_states[-1],
        #     "logits",
        #     output,
        #     "log_probs",
        #     log_probs,
        #     input_lengths,
        #     target_lengths,
        #     # x["audio_path"],
        #     x["audio_length"],
        #     x["phoneme"],
        # )
        # logger_path = self.logger.log_dir
        # torch.save(
        #     {
        #         "log_porbs": log_probs,
        #         "input_lengths": input_lengths,
        #         "target_length": target_lengths,
        #         "targets": targets,
        #         "audio_length" : x["audio_length"],
        #         'audio_path' : x['audio_path']
        #     },
        #     os.path.join(logger_path, f"{batch_idx}.pt")
        # )

        if loss != loss:
            print(batch_idx, "---" * 10)
            print(batch_idx, x["audio"], x["audio_path"], "logits", output, "log_probs", log_probs)

        # to compute metric and log samples
        phone_preds = self.tokenizer.batch_decode(x["language"], torch.argmax(output, dim=-1))
        # phone_preds = [_pred[: x["phoneme_id_length"][i]] for i, _pred in enumerate(phone_preds)]
        phone_preds = [combine_consecutive_identical(_pred) for i, _pred in enumerate(phone_preds)]
        labels = [_phoneme_id[: x["phoneme_id_length"][i]] for i, _phoneme_id in enumerate(x["phoneme_id"])]
        phone_targets = self.tokenizer.batch_decode(x["language"], labels)
        # phone_targets = x['phoneme']
        # return loss, output, output, targets, log_probs
        return loss, output, phone_preds, phone_targets


def combine_consecutive_identical(strings):
    if not strings:
        return []

    combined = [strings[0]]  # Start with the first string

    for i in range(1, len(strings)):
        # Only add the string if it's different from the last added string
        if strings[i] != combined[-1]:
            combined.append(strings[i])

    if combined[0] == "|":
        combined = combined[1:]

    return combined


# +
# model = BaseModule(network_param, optim_param)
# -

# ## load model

def load_phoneme_model(pretrained_path=None):
    ## Tokenizer
    from ay2.tools.text import Phonemer_Tokenizer_Recombination
    # from phoneme_model import BaseModule, network_param, optim_param

    vocab_path = "/home/ay/data/0-原始数据集/common_voice_11_0/vocab_phoneme"
    languages = ["en", "es", "de"]
    tokenizer = Phonemer_Tokenizer_Recombination(
        vocab_files=[
            os.path.join(vocab_path, f"vocab-phoneme-{language}.json")
            for language in languages
        ],
        languages=languages,
    )
    if pretrained_path is None:
        model = BaseModule(network_param, optim_param, tokenizer=tokenizer)
    else:
        model = BaseModule.load_from_checkpoint(
            pretrained_path,
            network_param=network_param,
            optim_param=optim_param,
            tokenizer=tokenizer,
        ).cpu()
        # model = BaseModule(network_param, optim_param, tokenizer=tokenizer)
        # model  = model.load_state_dict(torch.load(pretrained_path))
    return model

# +
# model = load_phoneme_model()
