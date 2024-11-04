python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 5 --seed 5;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' -t 1 -v 5;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 6 --seed 6;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' -t 1 -v 6;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 5 --seed 5;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' -t 1 -v 5;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 6 --seed 6;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' -t 1 -v 6;\
