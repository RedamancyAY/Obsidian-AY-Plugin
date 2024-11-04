python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 11 --seed 11;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' -t 1 -v 11;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 12 --seed 12;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' -t 1 -v 12;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 11 --seed 11;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' -t 1 -v 11;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 12 --seed 12;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' -t 1 -v 12;\
