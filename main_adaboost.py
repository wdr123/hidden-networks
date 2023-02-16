import os
import pathlib
import random
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.net_utils import (
    set_model_prune_rate,
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    save_sample_weights
)
from utils.schedulers import get_policy


from args import args
import importlib

import data
import models
import numpy as np 
import data as datas


def main():

    if args.seed is not None:
        random.seed(args.seed)
    args.name = args.arch+'_'+args.set
    if args.seed is not None:
        args.name = args.name+'_'+str(args.seed)
    args.prune_rate = float(args.prune_rate)/100
    if args.evaluate:
        
        dir_name = args.config.split('/')[-1].replace('.yml', '')
        if'baseline' in args.config:
            model_dir = 'runs/'+dir_name+'/'+args.name+'/prune_rate=0.0/checkpoints'
            file_names = os.listdir(model_dir)
        else:
            model_dir = 'runs/'+dir_name+'/'+args.name+'/prune_rate='+str(args.prune_rate)+'/checkpoints'
            file_names = os.listdir(model_dir)
        print("Model directory is", model_dir, 'file names', file_names)
        if 'epoch_199.state' in file_names:
            args.pretrained = os.path.join(model_dir, 'epoch_199.state')
        else:
            args.pretrained = os.path.join(model_dir, 'model_best.pth')
        
            

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    args.gpu = None
    train, validate, modifier, get_output, get_weights = get_trainer(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)
    total_param = sum(p.numel() for p in model.parameters())
    total_train_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Trainable Parameters", total_train_param, "Total Parameters", total_param)


    if args.pretrained:
        pretrained(args, model)

    optimizer = get_optimizer(args, model)
    data = get_dataset(args)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    if args.label_smoothing is None:
        criterion =   nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        best_acc1 = resume(args, model, optimizer)

    # Data loading code
   
    if args.evaluate:
        if 'baseline' in args.pretrained or 'dense' in args.pretrained:
            net_type = 'dense'
        else:
            net_type = str(args.prune_rate)

        if args.noise_type is not None:
            if args.set=='CIFAR10':
                val_loader = getattr(datas, 'CIFAR10C')(args).val_loader
            elif args.set=='TinyImageNet':
                val_loader = getattr(datas, 'TinyImageNetC')(args).val_loader
            else:
                print("Not recognized")
            output_path = "outputs/output_"+args.arch+'_'+args.set+'_'+net_type+'_'+args.noise_type
            gt_path = "outputs/gts_"+args.arch+'_'+args.set+'_'+net_type+'_'+args.noise_type
            loss_path = "outputs/losses_"+args.arch+'_'+args.set+'_'+net_type+'_'+args.noise_type
            if args.seed is None:
                tail_part = ".npy"
            else:
                tail_part = "_"+str(args.seed)+".npy"
                
        else:
            val_loader = data.val_loader
            output_path = "outputs/output_"+args.arch+'_'+args.set+'_'+net_type
            gt_path = "outputs/gts_"+args.arch+'_'+args.set+'_'+net_type
            loss_path = "outputs/losses_"+args.arch+'_'+args.set+'_'+net_type
        if args.seed is None:
            tail_part = ".npy"
        else:
            tail_part = "_"+str(args.seed)+".npy"

        output_path+=tail_part
        gt_path+=tail_part
        loss_path+=tail_part

        acc1, acc5, all_gt, all_loss, all_output = get_output(
            val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        
        np.save(output_path, all_output)
        np.save(gt_path, all_gt)
        np.save(loss_path, all_loss)
        return

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir
    
 
 
    
    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )
    losses = []
    acc1s = []
    # Start training
    for epoch in range(args.start_epoch, args.epochs):
        lr_policy(epoch, iteration=None)
        modifier(args, epoch, model)

        cur_lr = get_lr(optimizer)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5, train_loss = train(
            data.train_loader, model, criterion, optimizer, epoch, args, writer=None
        )
       
        losses.append(train_loss)
        acc1s.append(train_acc1)
        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5, _ = validate(data.val_loader, model, criterion, args, None, epoch)


        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

  
        

        if args.conv_type == "SampleSubnetConv":
            count = 0
            sum_pr = 0.0
            for n, m in model.named_modules():
                if isinstance(m, SampleSubnetConv):
                    # avg pr across 10 samples
                    pr = 0.0
                    for _ in range(10):
                        pr += (
                            (torch.rand_like(m.clamped_scores) >= m.clamped_scores)
                            .float()
                            .mean()
                            .item()
                        )
                    pr /= 10.0
                    
                    sum_pr += pr
                    count += 1

            args.prune_rate = sum_pr / count
            

        
        end_epoch = time.time()
    train_infer_loader = data.train_infer_loader
    if args.seed==1:
        n_samples = len(train_infer_loader.dataset)
        w_old = np.zeros(n_samples)
        w_old[:] = 1/n_samples
    else:
        w_old = np.load(os.path.join('runs/global/sample_weights/'+args.arch+'_'+args.set+'_'+str(args.prune_rate)+'_'+str(args.seed)+'.npy'))

    w_new = get_weights(train_infer_loader, model, args, w_old)
    save_sample_weights(w_new, 'runs/global/sample_weights/'+args.arch+'_'+args.set+'_'+str(args.prune_rate)+'_'+str(args.seed+1)+'.npy')
    losses = np.array(losses)
    acc1s = np.array(acc1s)
    np.save(os.path.join(run_base_dir, 'losses.npy'), losses)
    np.save(os.path.join(run_base_dir, 'acc1.npy'), acc1s)
    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        prune_rate=args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
    )


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier, trainer.get_output, trainer.get_weights


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")

def load_global_model(args, model):
    global_model_path = 'runs/global/models/'+args.arch+'_'+args.set+'.state'
    if os.path.isfile(global_model_path):
        print("=> loading global model from '{}'".format(global_model_path))
        global_model_checkpoint = torch.load(global_model_path, map_location = torch.device("cuda:{}".format(args.multigpu[0])), )["state_dict"]
        model_state_dict = model.state_dict()
        for k, v in global_model_checkpoint.items():
            if k not in model_state_dict or v.size()!=model_state_dict[k].size():
                print("IGNORE:", k)
        global_model_checkpoint = {k:v for k, v in global_model_checkpoint.items() if (k in model_state_dict and v.size()==model_state_dict[k].size())}
        model_state_dict.update(global_model_checkpoint)
        model.load_state_dict(model_state_dict)
    else:
        # Save the initial state
        save_checkpoint(
            {
                "state_dict": model.state_dict(),
            },
            False,
            filename=global_model_path,
            save=False,
        )
    return model 






def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch]()

    # applying sparsity to the network
    if (
        args.conv_type != "DenseConv"
        and args.conv_type != "SampleSubnetConv"
        and args.conv_type != "ContinuousSparseConv"
    ):
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")
        model = load_global_model(args, model)
        set_model_prune_rate(model, prune_rate=args.prune_rate)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    if args.freeze_weights:
        freeze_model_weights(model)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()

