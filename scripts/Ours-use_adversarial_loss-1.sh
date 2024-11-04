python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/use_adversarial_loss' --clear_log 1 -v 7 --seed 7;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/use_adversarial_loss' -t 1 -v 7;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/use_adversarial_loss' --clear_log 1 -v 8 --seed 8;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/use_adversarial_loss' -t 1 -v 8;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/use_adversarial_loss' --clear_log 1 -v 7 --seed 7;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/use_adversarial_loss' -t 1 -v 7;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/use_adversarial_loss' --clear_log 1 -v 8 --seed 8;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/use_adversarial_loss' -t 1 -v 8;\
