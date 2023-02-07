#!/bin/bash -lT


#SBATCH --mail-user=986739772@qq.com
#SBATCH --mail-type=ALL
#SBATCH -A vision -p tier3 -n 4
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH --gres=gpu:v100:1
#SBATCH -t 0-12:00:00

prune_rate=0.03
#prune_rate=0.05
ensemble_subnet_init="None"
#ensemble_subnet_init="kaiming_normal kaiming_uniform"
data_repo="CIFAR10 CIFAR100"
arch_repo="cResNet18 cResNet50"


python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --log-dir runs1_KL --name CIFAR10 --data DATA/ --set CIFAR10 --arch cResNet18 --prune-rate 0.03 \
        --ensemble-subnet-init $ensemble_subnet_init --KL

python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --log-dir runs1_L2 --name CIFAR10 --data DATA/ --set CIFAR10 --arch cResNet18 --prune-rate 0.03 \
        --ensemble-subnet-init $ensemble_subnet_init --L2

python main.py --config configs/smallscale/resnet50/resnet50-ukn-unsigned.yaml --gpu 0 --log-dir runs1_KL --name CIFAR10 --data DATA/ --set CIFAR10 --arch cResNet50 --prune-rate 0.03 \
        --ensemble-subnet-init $ensemble_subnet_init --KL

python main.py --config configs/smallscale/resnet50/resnet50-ukn-unsigned.yaml --gpu 0 --log-dir runs1_L2 --name CIFAR10 --data DATA/ --set CIFAR10 --arch cResNet50 --prune-rate 0.03 \
        --ensemble-subnet-init $ensemble_subnet_init --L2
        
        
python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --log-dir runs1_KL --num-classes 100 --name CIFAR100 --data DATA/ --set CIFAR100 --arch cResNet18 --prune-rate 0.03 \
        --ensemble-subnet-init $ensemble_subnet_init --KL

python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --log-dir runs1_L2 --num-classes 100 --name CIFAR100 --data DATA/ --set CIFAR100 --arch cResNet18 --prune-rate 0.03 \
        --ensemble-subnet-init $ensemble_subnet_init --L2

python main.py --config configs/smallscale/resnet50/resnet50-ukn-unsigned.yaml --gpu 0 --log-dir runs1_KL --num-classes 100 --name CIFAR100 --data DATA/ --set CIFAR100 --arch cResNet50 --prune-rate 0.03 \
        --ensemble-subnet-init $ensemble_subnet_init --KL

python main.py --config configs/smallscale/resnet50/resnet50-ukn-unsigned.yaml --gpu 0 --log-dir runs1_L2 --num-classes 100 --name CIFAR100 --data DATA/ --set CIFAR100 --arch cResNet50 --prune-rate 0.03 \
        --ensemble-subnet-init $ensemble_subnet_init --L2