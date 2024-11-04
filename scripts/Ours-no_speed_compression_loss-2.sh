python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/no_speed_compression_loss' --clear_log 1 -v 9 --seed 9;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/no_speed_compression_loss' -t 1 -v 9;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/no_speed_compression_loss' --clear_log 1 -v 10 --seed 10;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/no_speed_compression_loss' -t 1 -v 10;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/no_speed_compression_loss' --clear_log 1 -v 9 --seed 9;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/no_speed_compression_loss' -t 1 -v 9;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/no_speed_compression_loss' --clear_log 1 -v 10 --seed 10;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/no_speed_compression_loss' -t 1 -v 10;\
