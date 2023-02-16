import time
import torch
import tqdm
import numpy as np

from utils.eval_utils import accuracy, eval_calibration
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate", "modifier"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    ece = AverageMeter("ECE", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, ece],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        if args.KL or args.L2:
            embed = model.embedding(images)
            loss = criterion(images, output, target, embed)
        else:
            loss = criterion(output, target)

        # measure ece, accuracy and record loss
        if output.size(dim=-1) == 10:
            Bm = 3
        elif output.size(dim=-1) == 100:
            Bm = 10
        else:
            Bm = 15

        weighted_ece = eval_calibration(output, target, M = Bm)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        ece.update(weighted_ece.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg, ece.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    ece = AverageMeter("ECE", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5, ece], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            if args.KL or args.L2:
                embed = model.embedding(images)
                loss = criterion(output, target, embed)
            else:
                loss = criterion(output, target)

            # measure accuracy and record loss
            if output.size(dim=-1) == 10:
                Bm = 3
            elif output.size(dim=-1) == 100:
                Bm = 10
            else:
                Bm = 15

            if i==0:
                with open('output.npy', 'wb') as f:
                    np.save(f, output.cpu().numpy())
                with open('target.npy', 'wb') as f:
                    np.save(f, target.cpu().numpy())

            weighted_ece = eval_calibration(output, target, M=Bm)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            ece.update(weighted_ece.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg, ece.avg, losses.avg

def modifier(args, epoch, model):
    return
