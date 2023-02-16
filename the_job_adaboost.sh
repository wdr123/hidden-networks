#!/bin/bash -lT


#SBATCH --mail-user=986739772@qq.com
#SBATCH --mail-type=ALL
#SBATCH -A vision -p tier3 -n 4
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH --gres=gpu:v100:1
#SBATCH -t 1-00:00:00


#prune_rate=0.05
#ensemble_subnet_init="None"
#ensemble_subnet_init="kaiming_normal kaiming_uniform"
#data_repo="CIFAR10 CIFAR100"
#arch_repo="cResNet18 cResNet50"


python main_adaboost.py --config configs/smallscale/others/resnet101-ukn-unsigned.yaml --multigpu 0 --seed $seed --name CIFAR100 --data DATA/ --set CIFAR100 --prune-rate 5 --num-classes 100 --batch-size 256
