python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/con_loss_feat_voc' --clear_log 1 -v 1 --seed 1;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_method' --ablation 'ablations/con_loss_feat_voc' -t 1 -v 1;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/con_loss_feat_voc' --clear_log 1 -v 1 --seed 1;\
python train.py --gpu $1 --cfg 'Ours/ResNet/LibriSeVoc_cross_dataset' --ablation 'ablations/con_loss_feat_voc' -t 1 -v 1;\
