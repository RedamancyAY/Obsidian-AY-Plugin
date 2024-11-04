from ay2.torch.lightning.callbacks import (
    Color_progress_bar,
    EER_Callback,
    EarlyStoppingWithLambdaMonitor,
    EarlyStoppingWithMinimumEpochs,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from functools import partial

from ay2.torch.lightning.callbacks.metrics import (
    BinaryACC_Callback,
    BinaryAUC_Callback,
    MulticlassACC_Callback,
    MulticlassAUC_Callback,
    BinaryClsCount_Callback,
)
from ay2.torch.lightning.callbacks import Collect_Callback


def common_callbacks():
    callbacks = [
        Color_progress_bar(),
        BinaryACC_Callback(batch_key="label", output_key="logit"),
        BinaryAUC_Callback(batch_key="label", output_key="logit"),
        EER_Callback(batch_key="label", output_key="logit"),
        #         BinaryClsCount_Callback(
        #         batch_key="label",
        #         output_key="logit",
        #         num_classes=2,
        #         theme="accFake"
        #     )
    ]

    return callbacks


def custom_callbacks(args, cfg):
    callbacks = []

    if args.cfg.startswith("OursLCNN/"):
        callbacks = [
            BinaryClsCount_Callback(
                batch_key="label",
                output_key="logit",
                num_classes=2,
                theme="count",
            ),
            MulticlassACC_Callback(
                batch_key="vocoder_label",
                output_key="vocoder_logit",
                num_classes=32,
                theme="vocoder",
            ),
            MulticlassACC_Callback(
                batch_key="ASGD_domain_label",
                output_key="domain_logit",
                num_classes=3,
                theme="domain",
            ),
            MulticlassAUC_Callback(
                batch_key="vocoder_label",
                output_key="vocoder_logit",
                num_classes=32,
                theme="vocoder",
            ),
            MulticlassAUC_Callback(
                batch_key="ASGD_domain_label",
                output_key="domain_logit",
                num_classes=3,
                theme="domain",
            ),
        ]
    if args.cfg.startswith("OursPhonemeGAT"):
        callbacks = [
            BinaryClsCount_Callback(
                batch_key="label",
                output_key="logit",
                num_classes=2,
                theme="count",
            )
        ]

    if args.cfg.startswith("OursMultiView/") or args.cfg.startswith("OursMultiView3/"):
        callbacks = [
            BinaryACC_Callback(batch_key="label", output_key="logit1D", theme="1D"),
            BinaryACC_Callback(batch_key="label", output_key="logit2D", theme="2D"),
            BinaryAUC_Callback(batch_key="label", output_key="logit1D", theme="1D"),
            BinaryAUC_Callback(batch_key="label", output_key="logit2D", theme="2D"),
            EER_Callback(batch_key="label", output_key="logit1D", theme="1D"),
            EER_Callback(batch_key="label", output_key="logit2D", theme="2D"),
            BinaryClsCount_Callback(
                batch_key="label",
                output_key="logit1D",
                num_classes=2,
                theme="count1D",
            ),
            BinaryClsCount_Callback(
                batch_key="label",
                output_key="logit2D",
                num_classes=2,
                theme="count2D",
            ),
            BinaryClsCount_Callback(
                batch_key="label",
                output_key="logit",
                num_classes=2,
                theme="count",
            ),
        ]

    if args.cfg.startswith("Ours/"):
        n = cfg.MODEL.method_classes + 1
        callbacks = [
            MulticlassACC_Callback(
                batch_key="vocoder_label",
                output_key="vocoder_logit",
                num_classes=n,
                theme="vocoder",
            ),
            MulticlassAUC_Callback(
                batch_key="vocoder_label",
                output_key="vocoder_logit",
                num_classes=n,
                theme="vocoder",
            ),
            # BinaryACC_Callback(
            #     batch_key="label",
            #     output_key="content_based_cls_logit",
            #     num_classes=n,
            #     theme="content_based_cls",
            # ),
            # BinaryAUC_Callback(
            #     batch_key="label",
            #     output_key="content_based_cls_logit",
            #     num_classes=n,
            #     theme="content_based_cls",
            # ),
            # EER_Callback(batch_key="label", output_key="content_based_cls_logit", theme='content_based_cls'),
            # BinaryACC_Callback(
            #     batch_key="label",
            #     output_key="vocoder_based_cls_logit",
            #     num_classes=n,
            #     theme="vocoder_based_cls",
            # ),
            # BinaryAUC_Callback(
            #     batch_key="label",
            #     output_key="vocoder_based_cls_logit",
            #     num_classes=n,
            #     theme="vocoder_based_cls",
            # ),
            # EER_Callback(batch_key="label", output_key="vocoder_based_cls_logit", theme='vocoder_based_cls'),
            MulticlassACC_Callback(
                batch_key="vocoder_label",
                output_key="content_voc_logit",
                num_classes=n,
                theme="vocoder_adv",
            ),
            MulticlassAUC_Callback(
                batch_key="vocoder_label",
                output_key="content_voc_logit",
                num_classes=n,
                theme="vocoder_adv",
            ),
            MulticlassACC_Callback(
                batch_key="speed_label",
                output_key="speed_logit",
                num_classes=16,
                theme="speed",
            ),
            MulticlassAUC_Callback(
                batch_key="speed_label",
                output_key="speed_logit",
                num_classes=16,
                theme="speed",
            ),
            MulticlassACC_Callback(
                batch_key="compression_label",
                output_key="compression_logit",
                num_classes=10,
                theme="compression",
            ),
            MulticlassAUC_Callback(
                batch_key="compression_label",
                output_key="compression_logit",
                num_classes=10,
                theme="compression",
            ),
            BinaryClsCount_Callback(
                batch_key="label",
                output_key="logit",
                num_classes=2,
                theme="count",
            ),
            # EER_Callback(
            #     batch_key="label", output_key="content_logit", theme="content"
            # ),
        ]
    return callbacks


def training_callbacks(args):
    if args.cfg.startswith("Ours/"):
        monitor = "val-auc+++val-acc"
        es = EarlyStoppingWithLambdaMonitor
        # monitor = "val-auc"
        # es = EarlyStopping
    elif args.cfg.startswith("OursMultiView/"):
        monitor = "val-auc"
        # es = EarlyStopping
        es = EarlyStoppingWithMinimumEpochs
    elif args.cfg.startswith("SFATNet/"):
        monitor = "val-loss"
        es = EarlyStopping(
            monitor=monitor,
            min_delta=0.0001,
            patience=args.earlystop if args.earlystop > 1 else 3,
            mode="min",
            stopping_threshold=0.0001,
            verbose=True,
        )
    else:
        monitor = "val-auc"
        es = EarlyStopping

    if args.min_epoch > 1:
        # es = partial(EarlyStoppingWithMinimumEpochs, min_epochs=args.min_epoch)
        monitor = "val-auc"
        es = EarlyStoppingWithMinimumEpochs(
            min_epochs=args.min_epoch,
            monitor=monitor,
            min_delta=0.001,
            patience=args.earlystop if args.earlystop > 1 else 3,
            mode="max",
            stopping_threshold=0.998 if monitor == "val-auc+++val-acc" else 0.999,
            verbose=True,
        )

    if args.use_profiler:
        callbacks = []
        print("Since use profiler, no ModelCheckpoint is used!!!!!!")
    else:
        callbacks = [
            # save last ckpt
            # ModelCheckpoint(dirpath=None, save_top_k=0, save_last=True, save_weights_only=False),
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
        print("!!!!!!Earlystoping is ", es, monitor)
        callbacks.append(
            es(
                monitor=monitor,
                min_delta=0.001,
                patience=args.earlystop if args.earlystop > 1 else 3,
                mode="max",
                stopping_threshold=0.998 if monitor == "val-auc+++val-acc" else 0.999,
                verbose=True,
            )
            if type(es) == type
            else es
        )
    return callbacks


def make_collect_callbacks(args, cfg):
    name = args.cfg.replace("/", "-")
    callbacks = [
        Collect_Callback(
            ### cross method
            batch_keys=["label", "vocoder_label"], 
            # batch_keys=["label", "vocoder_label_org"],  # cross dataset　
            output_keys=["feature"],
            save_path=f"./0-实验结果/npz/{name}",
        )
    ]
    return callbacks


def make_CLIP_pretrain_callbacks(args, cfg):
    monitor = "val-loss"
    mode = "min"
    callbacks = [
        Color_progress_bar(),
        # save last ckpt
        ModelCheckpoint(dirpath=None, save_top_k=0, save_last=True, save_weights_only=False),
        # save best ckpt
        ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            monitor=monitor,
            mode=mode,
            save_last=False,
            filename="best-{epoch}-{val-loss:.6f}",
            save_weights_only=True,
            verbose=True,
        ),
        EarlyStopping(
            monitor=monitor,
            min_delta=0.001,
            patience=args.earlystop if args.earlystop > 1 else 3,
            mode=mode,
            verbose=True,
        ),
    ]
    return callbacks


def make_callbacks(args, cfg):
    if "Ours" in args.cfg and "VGGSound" in args.cfg:
        return make_CLIP_pretrain_callbacks(args, cfg)

    callbacks = common_callbacks()
    callbacks += custom_callbacks(args, cfg)
    if not args.test:
        callbacks += training_callbacks(args)

    if args.collect and args.test:
        callbacks += make_collect_callbacks(args, cfg)

    return callbacks
