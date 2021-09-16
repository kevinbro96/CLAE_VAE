python main.py --dataset cifar10 --batch-size 128 --gpu 0 --seed 1
python eval.py --dataset cifar10 --batch-size 128 --gpu 0 --seed 1

python main.py --dataset cifar10 --batch-size 128 --gpu 0 --seed 1 --adv --bn_adv_momentum 0.01 --eps 1.0 --dim 128
python eval.py --dataset cifar10 --batch-size 128 --gpu 0 --seed 1 --adv --bn_adv_momentum 0.01 --eps 1.0 --dim 128
