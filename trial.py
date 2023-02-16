import os
import pickle
import sys
import torch
import numpy as np
# hahaha
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

# with open('gts_cResNet50_CIFAR10_0.03_1.npy', 'rb') as f:
#     a = np.load(f)

# with open('target.npy', 'rb') as f:
#     b = np.load(f)

# print(a.shape, )

data_src_root = "runs/global/sample_weights"

for file_name in os.listdir(data_src_root):
    file_path = os.path.join(data_src_root, file_name)
    if file_name.startswith("c100"):
        dest_file_path = os.path.join(data_src_root, 'c' + file_name[4:])
        os.rename(file_path, dest_file_path)