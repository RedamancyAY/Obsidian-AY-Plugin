python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/one_stem_content' --clear_log 1 -v 1 --seed 1;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/one_stem_content' -t 1 -v 1;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/one_stem_content' --clear_log 1 -v 1 --seed 1;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/one_stem_content' -t 1 -v 1;\
