#!/bin/bash
echo "Bash version ${BASH_VERSION}..."

script="$0"
arch="$1"
dataset="$2"


array=(0 0.1 0.3 0.5 0.7 0.9)


if [ "$arch" = "resnet18" ] | [ "$dataset" = "cifar10" ]; then
  for i in "${array[@]}"
  do
    python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --name cifar10 --data DATA/ --set CIFAR10 --prune-rate $i
  done

elif [ "$arch" = "resnet18" ] | [ "$dataset" = "cifar100" ]; then
  for i in "${array[@]}"
  do
    python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --name cifar100 --data DATA/ --set CIFAR100 --prune-rate $i
  done

elif [ "$arch" = "resnet18" ] | [ "$dataset" = "tinyimagenet" ]; then
  for i in "${array[@]}"
  do
    python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --multigpu 0 --name tinyimagenet --data DATA/ --set TinyImageNet --prune-rate $i
  done

elif [ "$arch" = "resnet50" ] | [ "$dataset" = "tinyimagenet" ]; then
  for i in "${array[@]}"
  do
    python main.py --config configs/largescale/subnetonly/resnet50-ukn-unsigned.yaml --multigpu 0 --name tinyimagenet --data DATA/ --set TinyImageNet --prune-rate $i
  done

fi

