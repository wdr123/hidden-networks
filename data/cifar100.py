import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import WeightedRandomSampler
from args import args
import numpy as np 
from PIL import Image

class CustomDataCIFAR100:
    def __init__(self, data_root, seed, transform= None, train = True):
        super(CustomDataCIFAR100, self).__init__()
        if train:
            self.data = np.load(os.path.join(data_root, 'X_train.npy'), allow_pickle = True)
            self.labels = np.load(os.path.join(data_root, 'y_train.npy'), allow_pickle = True).flatten()
        else:
            self.data = np.load(os.path.join(data_root, 'X_test.npy'), allow_pickle = True)
            self.labels = np.load(os.path.join(data_root, 'y_test.npy'), allow_pickle = True).flatten()
        self.transform = transform
       
        
    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        if self.transform:
            X = Image.fromarray(X.astype(np.uint8))
            X = self.transform(X)
        return X, y 
    def __len__(self):
        return len(self.data)

        
class CIFAR100:
    def __init__(self, args):
        super(CIFAR100, self).__init__()

        data_root = os.path.join(args.data, "cifar100")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.5074,0.4867,0.4411], std=[0.2011,0.1987,0.2025]
        )


        transform=transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])


        train_dataset = CustomDataCIFAR100(data_root,  args.seed, transform = transform, train = True)
        if args.seed==1:
            sampler = None 
            shuffle = True
        else:
            weights = np.load('runs/global/sample_weights/'+args.arch+'_'+args.set+'_'+str(args.prune_rate)+'_'+str(args.seed)+'.npy')
            sampler = WeightedRandomSampler(weights, len(train_dataset.data), replacement = True)
            shuffle = False
        transform=transforms.Compose([transforms.ToTensor(), normalize])
        test_dataset = CustomDataCIFAR100(data_root,  args.seed, transform = transform, train = False)
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler = sampler, shuffle=shuffle, **kwargs
        )
        train_infer_dataset = CustomDataCIFAR100(data_root,  args.seed, transform = transform, train = True)


        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )
        self.train_infer_loader = torch.utils.data.DataLoader(train_infer_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)



