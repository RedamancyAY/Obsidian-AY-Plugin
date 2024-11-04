python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/no_speed_compression_loss' --clear_log 1 -v 5 --seed 5;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/no_speed_compression_loss' -t 1 -v 5;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/no_speed_compression_loss' --clear_log 1 -v 6 --seed 6;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/no_speed_compression_loss' -t 1 -v 6;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/no_speed_compression_loss' --clear_log 1 -v 5 --seed 5;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/no_speed_compression_loss' -t 1 -v 5;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/no_speed_compression_loss' --clear_log 1 -v 6 --seed 6;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/no_speed_compression_loss' -t 1 -v 6;\
