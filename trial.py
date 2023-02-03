import os
import numpy as np
import pickle
import sys
from PIL import Image
import torch

# data_root = 'DATA/cifar10/CIFAR-10-C'
#
#
# corruption_list = [
#     # 'labels.npy',
#     'brightness.npy',
#     'contrast.npy',
#     'gaussian_blur.npy',
#     'gaussian_noise.npy',
#     'defocus_blur.npy',
# ]
#
#
# for file_name in corruption_list:
#     file_path = os.path.join(data_root, file_name)
#     entry = np.load(file_path)
#
#     for idx, img_numpy in enumerate(entry[:10]):
#         img = Image.fromarray(img_numpy, "RGB")
#         img.save(os.path.join(data_root, 'display', file_name[:-4]+f'_{idx}.jpg'))


# data_root = 'DATA/cifar10/cifar-10-batches-py'
#
#
# data_list = [
#     'data_batch_1',
#     'data_batch_2',
#     'data_batch_3',
#     'data_batch_4',
#     'data_batch_5',
# ]
#
# for file_name in data_list:
#     file_path = os.path.join(data_root, file_name)
#
#     with open(file_path, 'rb') as f:
#         if sys.version_info[0] == 2:
#             entry = pickle.load(f)
#         else:
#             entry = pickle.load(f, encoding='latin1')
#
#     print(type(entry['labels']))
#     break


# print(torch.max(torch.tensor([[1,2,3,4],[1,2,3,4]]), dim=-1))