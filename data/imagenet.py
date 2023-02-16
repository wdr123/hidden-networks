import os

import torch
from torchvision import datasets, transforms

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy("file_system")

class ImageNet:
    def __init__(self, args):
        super(ImageNet, self).__init__()

        data_root = os.path.join(args.data, "imagenet")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        if args.seed==1:
            sampler = None 
            shuffle = True
        else:
            weights = np.load('runs/global/sample_weights/'+args.arch+'_'+args.set+'_'+str(args.prune_rate)+'_'+str(args.seed)+'.npy')
            sampler = WeightedRandomSampler(weights, len(train_dataset.data), replacement = True)
            shuffle = False

        transform=transforms.Compose([transforms.ReSize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
        
       
        train_infer_dataset = datasets.ImageFolder(traindir, transform = transform)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=shuffle, sampler = sampler, **kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(
                valdir,
                transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=args.batch_size,
            shuffle=False,
            **kwargs
        )
        self.train_infer_loader = torch.utils.data.DataLoader(train_infer_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
