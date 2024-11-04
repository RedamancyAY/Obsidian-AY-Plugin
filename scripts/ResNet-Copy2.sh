
python train.py --gpu $1 --cfg 'Ours/ResNet/DECRO_english' --clear_log 1 -v 17 --seed 17 --min_epoch 10;\
python train.py --gpu $1 --cfg 'Ours/ResNet/DECRO_english'  -t 1 -v 17;\
python train.py --gpu $1 --cfg 'Ours/ResNet/DECRO_english' --clear_log 1 -v 18 --seed 18 --min_epoch 10;\
python train.py --gpu $1 --cfg 'Ours/ResNet/DECRO_english'  -t 1 -v 18;\
python train.py --gpu $1 --cfg 'Ours/ResNet/DECRO_english' --clear_log 1 -v 19 --seed 19 --min_epoch 10;\
python train.py --gpu $1 --cfg 'Ours/ResNet/DECRO_english'  -t 1 -v 19;\
python train.py --gpu $1 --cfg 'Ours/ResNet/DECRO_english' --clear_log 1 -v 20 --seed 20 --min_epoch 10;\
python train.py --gpu $1 --cfg 'Ours/ResNet/DECRO_english'  -t 1 -v 20;\
