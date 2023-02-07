import shutil
import os
import pathlib

# src_prefix = pathlib.Path('../DATA/imagenet/tinyimagenetC/val/images')
# dest_prefix = pathlib.Path('../DATA/imagenet/tinyimagenetC/val')
# record = {}
# with open('../DATA/imagenet/tinyimagenetC/val/val_annotations.txt') as f:
#     cur_line = f.readline()
#
#     while True:
#         name = cur_line.split('\t')[0]
#
#         category = cur_line.split('\t')[1].split('\t')[0]
#
#         if category not in record:
#             record[category] = [name]
#         else:
#             record[category].append(name)
#
#         cur_line = f.readline()
#         if not cur_line:
#             break
#
# print(record)
#
# for category, value in record.items():
#     dest = dest_prefix / category / "images"
#     if not dest.exists():
#         os.makedirs(dest)
#     for name in value:
#         shutil.move(src_prefix / name, dest / name)


dest_prefix = pathlib.Path('../DATA/imagenet/tinyimagenet/val')
for dir_path, subdirs, filenames in os.walk(dest_prefix):
    if dir_path.split('/')[-1].startswith('n'):
        for subdir in subdirs:
            if subdir == "image":
                sub_dir_path = os.path.join(dir_path, subdir)
                os.rename(sub_dir_path, os.path.join(dir_path, "images"))
