python train.py --gpu 0 --cfg 'RawGAT/wavefake_inner' --clear_log 1 -v 7 --seed 7;\
python train.py --gpu 0 --cfg 'RawGAT/wavefake_inner'  -t 1 -v 7;\

python train.py --gpu 1 --cfg 'RawGAT/wavefake_inner' --clear_log 1 -v 8 --seed 8;\
python train.py --gpu 1 --cfg 'RawGAT/wavefake_inner'  -t 1 -v 8;\

python train.py --gpu 2 --cfg 'RawGAT/wavefake_inner' --clear_log 1 -v 9 --seed 9;\
python train.py --gpu 2 --cfg 'RawGAT/wavefake_inner'  -t 1 -v 9;\

python train.py --gpu 0 --cfg 'RawGAT/wavefake_inner' --clear_log 1 -v 10 --seed 10;\
python train.py --gpu 0 --cfg 'RawGAT/wavefake_inner'  -t 1 -v 10;\
