#!/bin/bash -lT


#SBATCH --mail-user=986739772@qq.com
#SBATCH --mail-type=ALL
#SBATCH -A vision -p tier3 -n 4
#SBATCH -c 1
#SBATCH --mem=4g
#SBATCH --gres=gpu:v100:1
#SBATCH -t 0-12:00:00

conda activate LTH

if [ "$ensemble" = "True" ]; then
  if [ "$dataset" = "CIFAR10" ]; then
    if [ "$arch" = "resnet18" ] && [ "$ensemble_subnet_init" = "None" ]; then
        python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --name $dataset --data DATA/ --set $dataset --evaluate --arch ecResNet18 --prune-rate $prune
    else
        python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --ensemble_subnet_init $ensemble_subnet_init --evaluate --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune
#    elif [ "$arch" = "resnet50" ] && [ "$ensemble_subnet_init" = "None" ]; then
#        python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --name $dataset --data DATA/ --set $dataset --evaluate --arch ecResNet50 --prune-rate $prune
#    elif [ "$arch" = "resnet50" ] && [ "$ensemble_subnet_init" != "None" ]; then
#        python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --ensemble_subnet_init $ensemble_subnet_init --evaluate --name $dataset --data DATA/ --set $dataset --arch ecResNet50 --prune-rate $prune
    fi
  elif [ "$dataset" = "CIFAR100" ]; then
    if [ "$ensemble_subnet_init" = "None" ]; then
        python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --evaluate --name $dataset --data DATA/ --set $dataset --arch ecResNet18 --prune-rate $prune
    else
        python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --evaluate --ensemble_subnet_init $ensemble_subnet_init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune
#    elif [ "$arch" = "resnet50" ] && [ "$ensemble_subnet_init" = "None" ]; then
#        python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --evaluate --name $dataset --data DATA/ --set $dataset --arch ecResNet50 --prune-rate $prune
#    elif [ "$arch" = "resnet50" ] && [ "$ensemble_subnet_init" != "None" ]; then
#        python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --evaluate --ensemble_subnet_init $ensemble_subnet_init --name $dataset --data DATA/ --set $dataset --arch ecResNet50 --prune-rate $prune
    fi
  fi

elif [ "$dense" = "True" ]; then
  if [ "$dataset" = "CIFAR10" ] ; then
    if [ "$arch" = "cResNet18" ] ; then
      python main.py --config configs/smallscale/resnet18/resnet18-dense.yaml --gpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "cResNet50" ] ; then
      python main.py --config configs/smallscale/others/resnet50-dense.yaml --gpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "cResNet101" ] ; then
      python main.py --config configs/smallscale/others/resnet101-dense.yaml --gpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "vit" ] ; then
      python main.py --config configs/smallscale/others/vit-dense.yaml --gpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "swin" ] ; then
      python main.py --config configs/smallscale/others/swin-dense.yaml --gpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    fi

  fi


  if [ "$dataset" = "CIFAR100" ]; then
    if [ "$arch" = "cResNet18" ] ; then
      python main.py --config configs/smallscale/resnet18/resnet18-dense.yaml --gpu 0 --num-classes 100 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "cResNet50" ] ; then
      python main.py --config configs/smallscale/others/resnet50-dense.yaml --gpu 0 --num-classes 100 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "cResNet101" ] ; then
      python main.py --config configs/smallscale/others/resnet101-dense.yaml --gpu 0 --num-classes 100 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "vit" ] ; then
      python main.py --config configs/smallscale/others/vit-dense.yaml --gpu 0 --num-classes 100 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "swin" ] ; then
      python main.py --config configs/smallscale/others/swin-dense.yaml --gpu 0 --num-classes 100 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    fi

  fi

  if [ "$dataset" = "TinyImageNet" ]; then
    if [ "$arch" = "cResNet18" ] ; then
      python main.py --config configs/smallscale/resnet18/resnet18-dense.yaml --gpu 0 --num-classes 200 --batch-size 64 --epochs 200 --subnet-init $init --name $dataset --data DATA/imagenet --set $dataset --prune-rate $prune

    elif [ "$arch" = "cResNet50" ] ; then
      python main.py --config configs/smallscale/others/resnet50-dense.yaml --gpu 0 --num-classes 200 --batch-size 64 --epochs 200 --subnet-init $init --name $dataset --data DATA/imagenet --set $dataset --prune-rate $prune

    elif [ "$arch" = "cResNet101" ] ; then
      python main.py --config configs/smallscale/others/resnet101-dense.yaml --gpu 0 --num-classes 200 --batch-size 64 --epochs 200 --subnet-init $init --name $dataset --data DATA/imagenet --set $dataset --prune-rate $prune

    elif [ "$arch" = "vit" ] ; then
      python main.py --config configs/smallscale/others/vit-dense.yaml --gpu 0 --num-classes 200 --batch-size 64 --epochs 400 --subnet-init $init --name $dataset --data DATA/imagenet --set $dataset --prune-rate $prune

    elif [ "$arch" = "swin" ] ; then
      python main.py --config configs/smallscale/others/swin-dense.yaml --gpu 0 --num-classes 200 --batch-size 64 --epochs 400 --subnet-init $init --name $dataset --data DATA/imagenet --set $dataset --prune-rate $prune

    fi
  fi
else
  if [ "$dataset" = "CIFAR10" ] ; then
    if [ "$arch" = "cResNet18" ] ; then
      python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "cResNet50" ] ; then
      python main.py --config configs/smallscale/others/resnet50-ukn-unsigned.yaml --gpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "cResNet101" ] ; then
      python main.py --config configs/smallscale/others/resnet101-ukn-unsigned.yaml --gpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "vit" ] ; then
      python main.py --config configs/smallscale/others/vit-ukn-unsigned.yaml --gpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "swin" ] ; then
      python main.py --config configs/smallscale/others/swin-ukn-unsigned.yaml --gpu 0 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    fi

  fi


  if [ "$dataset" = "CIFAR100" ]; then
    if [ "$arch" = "cResNet18" ] ; then
      python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --num-classes 100 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "cResNet50" ] ; then
      python main.py --config configs/smallscale/others/resnet50-ukn-unsigned.yaml --gpu 0 --num-classes 100 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "cResNet101" ] ; then
      python main.py --config configs/smallscale/others/resnet101-ukn-unsigned.yaml --gpu 0 --num-classes 100 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "vit" ] ; then
      python main.py --config configs/smallscale/others/vit-ukn-unsigned.yaml --gpu 0 --num-classes 100 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    elif [ "$arch" = "swin" ] ; then
      python main.py --config configs/smallscale/others/swin-ukn-unsigned.yaml --gpu 0 --num-classes 100 --subnet-init $init --name $dataset --data DATA/ --set $dataset --arch $arch --prune-rate $prune

    fi

  fi

  if [ "$dataset" = "TinyImageNet" ]; then
    if [ "$arch" = "cResNet18" ] ; then
      python main.py --config configs/smallscale/resnet18/resnet18-ukn-unsigned.yaml --gpu 0 --num-classes 200 --batch-size 64 --epochs 200 --subnet-init $init --name $dataset --data DATA/imagenet --set $dataset --prune-rate $prune

    elif [ "$arch" = "cResNet50" ] ; then
      python main.py --config configs/smallscale/others/resnet50-ukn-unsigned.yaml --gpu 0 --num-classes 200 --batch-size 64 --epochs 200 --subnet-init $init --name $dataset --data DATA/imagenet --set $dataset --prune-rate $prune

    elif [ "$arch" = "cResNet101" ] ; then
      python main.py --config configs/smallscale/others/resnet101-ukn-unsigned.yaml --gpu 0 --num-classes 200 --batch-size 64 --epochs 200 --subnet-init $init --name $dataset --data DATA/imagenet --set $dataset --prune-rate $prune

    elif [ "$arch" = "vit" ] ; then
      python main.py --config configs/smallscale/others/vit-ukn-unsigned.yaml --gpu 0 --num-classes 200 --batch-size 64 --epochs 400 --subnet-init $init --name $dataset --data DATA/imagenet --set $dataset --prune-rate $prune

    elif [ "$arch" = "swin" ] ; then
      python main.py --config configs/smallscale/others/swin-ukn-unsigned.yaml --gpu 0 --num-classes 200 --batch-size 64 --epochs 400 --subnet-init $init --name $dataset --data DATA/imagenet --set $dataset --prune-rate $prune

    fi
  fi

fi

