from ay2.torch.lightning.callbacks import (
    Color_progress_bar,
    EarlyStoppingWithLambdaMonitor,
    EER_Callback,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ay2.torch.lightning.callbacks.metrics import (
    AudioPESQ_Callback,
    AudioSNR_Callback,
    AudioSDR_Callback,
    AudioPSNR_Callback,
    AudioSISDR_Callback,
    AudioSISNR_Callback,
    BinaryACC_Callback,
    BinaryAUC_Callback,
    BinaryClsCount_Callback,
    MulticlassACC_Callback,
    MulticlassAUC_Callback,
)


def common_callbacks():
    callbacks = [
        Color_progress_bar(),
        # AudioPESQ_Callback(batch_key="audio", output_key="x_re"),
        AudioSNR_Callback(batch_key="audio", output_key="x_re"),
        AudioSISDR_Callback(batch_key="audio", output_key="x_re"),
        AudioSISNR_Callback(batch_key="audio", output_key="x_re"),
        AudioSDR_Callback(batch_key="audio", output_key="x_re"),
        AudioPSNR_Callback(batch_key="audio", output_key="x_re"),
        BinaryACC_Callback(batch_key="label", output_key="logit"),
        BinaryACC_Callback(batch_key="label", output_key="org_logit", theme="org"),
        BinaryAUC_Callback(batch_key="label", output_key="logit"),
        BinaryAUC_Callback(batch_key="label", output_key="org_logit", theme="org"),
    ]

    return callbacks


def custom_callbacks(args, cfg):
    callbacks = []
    if args.cfg.startswith("Ours"):
        n = cfg.MODEL.method_classes + 1
        callbacks = []
    return callbacks


def training_callbacks(args):

    monitor = "val-SI-SDR"
    es = EarlyStopping

    callbacks = [
        # save last ckpt
        ModelCheckpoint(
            dirpath=None,
            save_top_k=0,
            save_last=True,
            save_weights_only=False,
            filename="lates-{epoch}",
        ),
        # save best ckpt
        ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            monitor=monitor,
            mode="max",
            save_last=False,
            filename="best-{epoch}-{val-auc:.4f}",
            save_weights_only=True,
            verbose=True,
        ),
    ]

    if args.earlystop:
        callbacks.append(
            es(
                monitor=monitor,
                min_delta=0.0001,
                patience=args.earlystop if args.earlystop > 1 else 3,
                mode="max",
                verbose=True,
            )
        )
    return callbacks


def make_attack_callbacks(args, cfg):
    callbacks = common_callbacks()
    callbacks += custom_callbacks(args, cfg)
    if not args.test:
        callbacks += training_callbacks(args)

    return callbacks
