# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import os

from yacs.config import CfgNode as ConfigurationNode
# -

from .datasets_config import ALL_DATASETS, get_dataset_cfg

# # 默认配置

__C = ConfigurationNode()
__C.DATASET = ConfigurationNode()
__C.DATASET.name = "WaveFake"
__C.DATASET.task = "inner_eval"

# + editable=true slideshow={"slide_type": ""}
__C.MODEL = ConfigurationNode()
__C.MODEL.epochs = 200
__C.MODEL.optimizer = "AdamW"
__C.MODEL.weight_decay = 0.01
__C.MODEL.lr = 0.0001
__C.MODEL.lr_decay_factor = 0.5
__C.MODEL.lr_scheduler = "linear"
__C.MODEL.warmup_epochs = 10
__C.MODEL.label_smoothing = 0.1
__C.MODEL.method_classes = 7
__C.MODEL.pretrain = False
__C.MODEL.nograd = False
__C.MODEL.aug_policy = 'ss' 
__C.MODEL.use_op_loss = 1

## ablation studies
__C.MODEL.style_shuffle = 1
__C.MODEL.feat_shuffle = 1
__C.MODEL.voc_con_loss = 1 # vocoder constrastive loss
__C.MODEL.feat_con_loss = 1 # final_feat constrastive loss
__C.MODEL.use_adversarial_loss = 1
__C.MODEL.use_speed_loss = 1
__C.MODEL.use_compression_loss = 1
__C.MODEL.use_f0_loss = 0
__C.MODEL.betas = [1.0, 0.5, 0.5, 0.5]


__C.MODEL.feature_extractor = 'MSFM'
__C.MODEL.dims = [32,64,64,128]
__C.MODEL.n_blocks = [1, 1, 2, 1]
__C.MODEL.beta = [2.0, 0.5, 0.5]
# __C.MODEL.beta = [1, 1, 1]
__C.MODEL.one_stem = False
__C.MODEL.use_only_vocoder_stem = False
__C.MODEL.use_speed_loss_in_one_stem = False
__C.MODEL.use_compression_loss_in_one_stem = False

## Ours MultiView
__C.MODEL.MultiViewCombine = ConfigurationNode()
__C.MODEL.MultiViewCombine.model1D = 'rawnet'
__C.MODEL.MultiViewCombine.model2D = 'resnet'

__C.MODEL.use_inner_CL = 1
__C.MODEL.use_inter_CL = 1
__C.MODEL.use_cls_loss_1_2 = 1
__C.MODEL.use_fusion = 1
__C.MODEL.use_fusion1D = 1
__C.MODEL.use_fusion2D = 1
__C.MODEL.use_mse_loss = 0
__C.MODEL.only_1D = 0
__C.MODEL.only_2D = 0
__C.MODEL.drop_layer = 0.
__C.MODEL.w_con = 1.0
__C.MODEL.w_cls = 1.0



__C.MODEL.PhonemeGAT = ConfigurationNode()
__C.MODEL.PhonemeGAT.use_aug=1
__C.MODEL.PhonemeGAT.use_clip=1
__C.MODEL.PhonemeGAT.use_GAT=1 
__C.MODEL.PhonemeGAT.use_pool=1
__C.MODEL.PhonemeGAT.backbone='wavlm'
__C.MODEL.PhonemeGAT.use_raw=0  # use pretrained WavLM, not pretrained phoneme recognition model
__C.MODEL.PhonemeGAT.n_edges=10


# -

def get_cfg_defaults(cfg_file=None, ablation=None):
    """
    Get a yacs CfgNode object with default values.

    The loading sequence is:
        __C -> cfg_file.yaml -> data_aug.yaml -> 0-model.yaml -> ablation.yaml

    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    res = __C.clone()

    if cfg_file is not None:
        res.merge_from_file(cfg_file)

        res.DATASET = get_dataset_cfg(
            name=res.DATASET.name, task=res.DATASET.task, __C=res.DATASET
        )

        aug_file_path = os.path.join(os.path.split(cfg_file)[0], "data_aug.yaml")
        if os.path.exists(aug_file_path):
            res.merge_from_file(aug_file_path)
            print("load aug yaml in ", aug_file_path)

        model_file_path = os.path.join(os.path.split(cfg_file)[0], "0-model.yaml")
        if os.path.exists(model_file_path):
            res.merge_from_file(model_file_path)
            print("load model yaml in ", model_file_path)

        if ablation is not None:
            ablation_file_path = os.path.join(
                os.path.split(cfg_file)[0], f"{ablation}.yaml"
            )
            res.merge_from_file(ablation_file_path)
            print("load ablation yaml in ", ablation_file_path)

    return res
