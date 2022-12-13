import os
import torch
import torchvision
from torchvision import transforms
import random
from torch.utils.data.sampler import SubsetRandomSampler
from args import args
import numpy as np
import pickle
import sys
from PIL import Image
from torchvision.datasets.vision import VisionDataset

class dataset_CIFAR10C(VisionDataset):
    """`CIFAR10C` Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.

    """
    base_folder = 'CIFAR-10-C'

    corruption_list = [
        'brightness.npy',
        'contrast.npy',
        'gaussian_blur.npy',
        'gaussian_noise.npy',
        'defocus_blur.npy',
    ]

    label_list = [
        'labels.npy',
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(dataset_CIFAR10C, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        self.data = []
        self.targets = []

        # now load the image and label numpy arrays stored in .npy files
        if self.train:
            for file_name in self.label_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)

                entry_original = np.array((np.load(file_path)),dtype=int)

                for i in range(len(self.corruption_list)):
                    entry = entry_original[i*10000:(i+1)*10000]
                    self.targets.extend(entry[:int(len(entry)*0.8)])

            for file_name in self.corruption_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)

                entry = np.load(file_path)
                entry = entry[:10000]
                self.data.append(entry[:int(len(entry) * 0.8)])
        else:
            for file_name in self.label_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)

                entry_original = np.array(np.load(file_path), dtype=int)
                for i in range(len(self.corruption_list)):
                    entry = entry_original[i * 10000:(i + 1) * 10000]
                    self.targets.extend(entry[int(len(entry) * 0.8):])

            for file_name in self.corruption_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)

                entry = np.load(file_path)
                entry = entry[:10000]
                self.data.append(entry[int(len(entry) * 0.8):])

        self.data = np.vstack(self.data)
        # self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        # self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)




class CIFAR10C:
    def __init__(self, args):
        super(CIFAR10C, self).__init__()

        data_root = os.path.join(args.data, "cifar10")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        normalize = transforms.Normalize(
            mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262]
        )

        train_dataset = dataset_CIFAR10C(
            root=data_root,
            train=True,
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

        test_dataset = dataset_CIFAR10C(
            root=data_root,
            train=False,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
        self.val_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )
