import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args import args
import numpy as np 
from torch.utils.data import TensorDataset
from torch import Tensor
from torchvision.io import read_image 

class CIFAR10C:
    def __init__(self, args):
        super(CIFAR10C, self).__init__()

        data_root = os.path.join(args.data, "CIFAR-10-C")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = torchvision.datasets.CIFAR10(
            root=data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
    
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
        MEAN = 255* np.array([0.491, 0.482, 0.447])
        STD = 255*np.array([0.247, 0.243, 0.262])
        val_feats = np.load(os.path.join(data_root, args.noise_type+".npy"))
        pro_val_feats = []
        for a in val_feats:
            temp = a.transpose(-1, 0, 1)
            temp = (temp-MEAN[:, None, None])/STD[:, None, None]
            pro_val_feats.append(temp)
        pro_val_feats = np.array(pro_val_feats)
        val_labels = np.load(os.path.join(data_root, "labels"+".npy"))
        val_feats = torch.Tensor(pro_val_feats)
        val_labels = torch.Tensor(val_labels).long()
        val_dataset = TensorDataset(val_feats, val_labels)

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )
