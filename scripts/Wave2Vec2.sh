python train.py --gpu $1 --cfg 'Wave2Vec2/ASV2021_inner' --clear_log 1 -v 0 --seed 0;\
python train.py --gpu $1 --cfg 'Wave2Vec2/ASV2021_inner'  -t 1 -v 0;\
python train.py --gpu $1 --cfg 'Wave2Vec2/ASV2021_inner' --clear_log 1 -v 1 --seed 1;\
python train.py --gpu $1 --cfg 'Wave2Vec2/ASV2021_inner'  -t 1 -v 1;\
python train.py --gpu $1 --cfg 'Wave2Vec2/ASV2021_inner' --clear_log 1 -v 2 --seed 2;\
python train.py --gpu $1 --cfg 'Wave2Vec2/ASV2021_inner'  -t 1 -v 2;\

