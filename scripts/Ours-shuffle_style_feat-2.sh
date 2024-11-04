python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 9 --seed 9;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' -t 1 -v 9;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 10 --seed 10;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/shuffle_style_feat' -t 1 -v 10;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 9 --seed 9;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' -t 1 -v 9;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' --clear_log 1 -v 10 --seed 10;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/shuffle_style_feat' -t 1 -v 10;\
