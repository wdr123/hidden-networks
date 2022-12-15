
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.


Traceback (most recent call last):
  File "main.py", line 439, in <module>
    main()
  File "main.py", line 44, in main
    main_worker(args)
  File "main.py", line 62, in main_worker
    data = get_dataset(args)
  File "main.py", line 295, in get_dataset
    dataset = getattr(data, args.set)(args)
  File "/home/dw7445/Projects/LTH_project/hidden-networks/data/cifar.py", line 34, in __init__
    normalize,
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 64, in __init__
    self.download()
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 148, in download
    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 248, in download_and_extract_archive
    download_url(url, download_root, filename, md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 74, in download_url
    makedir_exist_ok(root)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 50, in makedir_exist_ok
    os.makedirs(dirpath)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/os.py", line 221, in makedirs
    mkdir(name, mode)
FileNotFoundError: [Errno 2] No such file or directory: 'DATA/cifar10'
Traceback (most recent call last):
  File "main.py", line 439, in <module>
    main()
  File "main.py", line 44, in main
    main_worker(args)
  File "main.py", line 62, in main_worker
    data = get_dataset(args)
  File "main.py", line 295, in get_dataset
    dataset = getattr(data, args.set)(args)
  File "/home/dw7445/Projects/LTH_project/hidden-networks/data/cifar.py", line 34, in __init__
    normalize,
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 64, in __init__
    self.download()
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 148, in download
    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 248, in download_and_extract_archive
    download_url(url, download_root, filename, md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 74, in download_url
    makedir_exist_ok(root)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 50, in makedir_exist_ok
    os.makedirs(dirpath)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/os.py", line 221, in makedirs
    mkdir(name, mode)
FileNotFoundError: [Errno 2] No such file or directory: 'DATA/cifar10'
Traceback (most recent call last):
  File "main.py", line 439, in <module>
    main()
  File "main.py", line 44, in main
    main_worker(args)
  File "main.py", line 62, in main_worker
    data = get_dataset(args)
  File "main.py", line 295, in get_dataset
    dataset = getattr(data, args.set)(args)
  File "/home/dw7445/Projects/LTH_project/hidden-networks/data/cifar.py", line 34, in __init__
    normalize,
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 64, in __init__
    self.download()
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 148, in download
    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 248, in download_and_extract_archive
    download_url(url, download_root, filename, md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 74, in download_url
    makedir_exist_ok(root)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 50, in makedir_exist_ok
    os.makedirs(dirpath)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/os.py", line 221, in makedirs
    mkdir(name, mode)
FileNotFoundError: [Errno 2] No such file or directory: 'DATA/cifar10'
Traceback (most recent call last):
  File "main.py", line 439, in <module>
    main()
  File "main.py", line 44, in main
    main_worker(args)
  File "main.py", line 62, in main_worker
    data = get_dataset(args)
  File "main.py", line 295, in get_dataset
    dataset = getattr(data, args.set)(args)
  File "/home/dw7445/Projects/LTH_project/hidden-networks/data/cifar.py", line 34, in __init__
    normalize,
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 64, in __init__
    self.download()
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 148, in download
    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 248, in download_and_extract_archive
    download_url(url, download_root, filename, md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 74, in download_url
    makedir_exist_ok(root)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 50, in makedir_exist_ok
    os.makedirs(dirpath)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/os.py", line 221, in makedirs
    mkdir(name, mode)
FileNotFoundError: [Errno 2] No such file or directory: 'DATA/cifar10'
Traceback (most recent call last):
  File "main.py", line 439, in <module>
    main()
  File "main.py", line 44, in main
    main_worker(args)
  File "main.py", line 62, in main_worker
    data = get_dataset(args)
  File "main.py", line 295, in get_dataset
    dataset = getattr(data, args.set)(args)
  File "/home/dw7445/Projects/LTH_project/hidden-networks/data/cifar.py", line 34, in __init__
    normalize,
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 64, in __init__
    self.download()
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 148, in download
    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 248, in download_and_extract_archive
    download_url(url, download_root, filename, md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 74, in download_url
    makedir_exist_ok(root)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 50, in makedir_exist_ok
    os.makedirs(dirpath)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/os.py", line 221, in makedirs
    mkdir(name, mode)
FileNotFoundError: [Errno 2] No such file or directory: 'DATA/cifar10'
Traceback (most recent call last):
  File "main.py", line 439, in <module>
    main()
  File "main.py", line 44, in main
    main_worker(args)
  File "main.py", line 62, in main_worker
    data = get_dataset(args)
  File "main.py", line 295, in get_dataset
    dataset = getattr(data, args.set)(args)
  File "/home/dw7445/Projects/LTH_project/hidden-networks/data/cifar.py", line 34, in __init__
    normalize,
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 64, in __init__
    self.download()
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/cifar.py", line 148, in download
    download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 248, in download_and_extract_archive
    download_url(url, download_root, filename, md5)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 74, in download_url
    makedir_exist_ok(root)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/site-packages/torchvision/datasets/utils.py", line 50, in makedir_exist_ok
    os.makedirs(dirpath)
  File "/home/dw7445/miniconda3/envs/LTH/lib/python3.7/os.py", line 221, in makedirs
    mkdir(name, mode)
FileNotFoundError: [Errno 2] No such file or directory: 'DATA/cifar10'
