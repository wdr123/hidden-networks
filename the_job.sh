#!/bin/bash -lT


#SBATCH --mail-user=986739772@qq.com
#SBATCH --mail-type=ALL
#SBATCH -A vision -p tier3 -n 2
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH --gres=gpu:a100:1

conda activate LTH



if [ "$arch" = "resnet18" ] | [ "$dataset" = "cifar10" ]; then
  python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --name cifar10 --data DATA/ --set CIFAR10 --prune-rate $i

elif [ "$arch" = "resnet18" ] | [ "$dataset" = "cifar100" ]; then

  python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --name cifar100 --data DATA/ --set CIFAR100 --prune-rate $i


elif [ "$arch" = "resnet18" ] | [ "$dataset" = "tinyimagenet" ]; then

  python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --name tinyimagenet --data DATA/ --set TinyImageNet --prune-rate $i


elif [ "$arch" = "resnet50" ] | [ "$dataset" = "tinyimagenet" ]; then

  python main.py --config configs/largescale/subnetonly/resnet50-ukn-unsigned.yaml --multigpu 0 --name tinyimagenet --data DATA/ --set TinyImageNet --prune-rate $i

fi

