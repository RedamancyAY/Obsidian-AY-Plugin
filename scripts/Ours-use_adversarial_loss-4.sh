python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/use_adversarial_loss' --clear_log 1 -v 4 --seed 4;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/use_adversarial_loss' -t 1 -v 4;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/use_adversarial_loss' --clear_log 1 -v 4 --seed 4;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/use_adversarial_loss' -t 1 -v 4;\
