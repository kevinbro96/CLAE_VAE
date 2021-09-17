python main.py  --seed 1 --gpu 1  --dataset CIFAR10
python eval_lr.py  --seed 1  --gpu 1   --dataset CIFAR10
python eval_knn.py  --seed 1  --gpu 1   --dataset CIFAR10

python main.py --alpha 1.0 --seed 1 --gpu 4 --adv  --eps 1.0 --bn_adv_momentum 0.01 --dataset CIFAR10 --dim 512 --vae_path ../results/vae_dim512_kl0.1_simclr/model_epoch22.pth;
python eval_lr.py --alpha 1.0 --seed 1 --adv --gpu 0  --eps 1.0 --bn_adv_momentum 0.01 --dataset CIFAR10
python eval_knn.py --alpha 1.0 --seed 1 --adv --gpu 0  --eps 1.0 --bn_adv_momentum 0.01 --dataset CIFAR10

python main_clae.py --alpha 1.0 --seed 1 --gpu 4 --adv  --eps 0.03 --bn_adv_momentum 0.01 --dataset CIFAR10
