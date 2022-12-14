import os
import numpy as np
import pickle
import sys

# data_root = 'DATA/cifar10/CIFAR-10-C'
#
#
# corruption_list = [
#     'labels.npy',
#     # 'brightness.npy',
#     # 'contrast.npy',
#     # 'gaussian_blur.npy',
#     # 'gaussian_noise.npy',
#     # 'defocus_blur.npy',
# ]
#
#
# for file_name in corruption_list:
#     file_path = os.path.join(data_root, file_name)
#     entry = np.load(file_path)
#
#     print(len(entry))
#     break


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
