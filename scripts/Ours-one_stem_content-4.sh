python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/one_stem_content' --clear_log 1 -v 4 --seed 4;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/one_stem_content' -t 1 -v 4;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/one_stem_content' --clear_log 1 -v 4 --seed 4;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/one_stem_content' -t 1 -v 4;\
