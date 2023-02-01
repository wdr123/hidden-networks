#!/bin/bash -lT


#SBATCH --mail-user=986739772@qq.com
#SBATCH --mail-type=ALL
#SBATCH -A vision -p tier3 -n 4
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH --gres=gpu:v100:1
#SBATCH -t 1-00:00:00

conda activate LTH


if [ "$dataset" = "CIFAR10" ]; then
  if [ "$arch" = "resnet18" ]; then
    python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --name $dataset --data DATA/ --set $dataset --arch c10ResNet18 --prune-rate $prune
  
#    elif [ "$arch" = "resnet18" ] && [ "$init" = "signed_constant" ]; then
#      python main.py --config configs/smallscale/resnet18/resnet18-sc-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c10ResNet18 --prune-rate $prune
#
#    elif [ "$arch" = "resnet18" ] && [ "$init" = "kaiming_normal" ]; then
#      python main.py --config configs/smallscale/resnet18/resnet18-kn-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c10ResNet18 --prune-rate $prune
#
#    elif [ "$arch" = "resnet18" ] && [ "$init" = "standard" ]; then
#      python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c10ResNet18 --prune-rate $prune
#
  elif [ "$arch" = "resnet50" ]; then
    python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --name $dataset --data DATA/ --set $dataset --arch c10ResNet50 --prune-rate $prune

#  elif [ "$arch" = "resnet50" ] && [ "$init" = "signed_constant" ]; then
#    python main.py --config configs/smallscale/resnet18/resnet18-sc-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c10ResNet50 --prune-rate $prune
#
#  elif [ "$arch" = "resnet50" ] && [ "$init" = "kaiming_normal" ]; then
#    python main.py --config configs/smallscale/resnet18/resnet18-kn-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c10ResNet50 --prune-rate $prune
#
#  elif [ "$arch" = "resnet50" ] && [ "$init" = "standard" ]; then
#    python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c10ResNet50 --prune-rate $prune

  fi
fi


if [ "$dataset" = "CIFAR100" ]; then
  if [ "$arch" = "resnet18" ]; then
    python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --name $dataset --data DATA/ --set $dataset --arch c100ResNet18 --prune-rate $prune

#  elif [ "$arch" = "resnet18" ] && [ "$init" = "signed_constant" ]; then
#    python main.py --config configs/smallscale/resnet18/resnet18-sc-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c100ResNet18 --prune-rate $prune
#
#  elif [ "$arch" = "resnet18" ] && [ "$init" = "kaiming_normal" ]; then
#    python main.py --config configs/smallscale/resnet18/resnet18-kn-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c100ResNet18 --prune-rate $prune
#
#  elif [ "$arch" = "resnet18" ] && [ "$init" = "standard" ]; then
#    python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c100ResNet18 --prune-rate $prune

  elif [ "$arch" = "resnet50" ]; then
    python main.py --config configs/smallscale/resnet18/resnet18-ukm-unsigned.yaml --multigpu 0 --name $dataset --data DATA/ --set $dataset --arch c100ResNet50 --prune-rate $prune

#  elif [ "$arch" = "resnet50" ] && [ "$init" = "signed_constant" ]; then
#    python main.py --config configs/smallscale/resnet18/resnet18-sc-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c100ResNet50 --prune-rate $prune
#
#  elif [ "$arch" = "resnet50" ] && [ "$init" = "kaiming_normal" ]; then
#    python main.py --config configs/smallscale/resnet18/resnet18-kn-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c100ResNet50 --prune-rate $prune
#
#  elif [ "$arch" = "resnet50" ] && [ "$init" = "standard" ]; then
#    python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch c100ResNet50 --prune-rate $prune

  fi
fi

