from functools import partial
import os
import pathlib
import shutil
import math
from args import args as parse_args
from models.ensemble import Ensemble

import torch
import torch.nn as nn
import torch.nn.functional as F


def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None


def freeze_model_subnet(model):
    print("=> Freezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad = False
            print(f"==> No gradient to {n}.scores")
            if m.scores.grad is not None:
                print(f"==> Setting gradient of {n}.scores to None")
                m.scores.grad = None


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True


def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True


def set_model_prune_rate(model, prune_rate):
    print(f"==> Setting prune rate of network to {prune_rate}")

    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            m.set_prune_rate(prune_rate)
            print(f"==> Setting prune rate of {n} to {prune_rate}")


def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc



class KLoss(nn.Module):
    """
    KLoss with KL divergence regularizer using previous base learners' losses
    """

    def __init__(self, regularize=1.0):
        self.regularize = regularize
        self.ensemble = Ensemble(parse_args.arch)
        if parse_args.ensemble_subnet_init is None:
            self.subnet_init = ["unsigned_constant", "signed_constant", "kaiming_normal", "kaiming_uniform"]
        else:
            self.subnet_init = parse_args.ensemble_subnet_init
        config = pathlib.Path(parse_args.config).stem
        losses = []

        for idx in range(len(self.subnet_init)):
            search_dir = pathlib.Path(
                f"{parse_args.log_dir}/{config}/{parse_args.name}/prune_rate={parse_args.prune_rate}/subnet_init={self.subnet_init[idx]}")
            if search_dir.exists():
                losses.append(float((search_dir / "avg_evaloss.txt").read_text()))

        self.weights = F.normalize(torch.tensor(losses), dim=0).detach()

    def forward(self, x, target, embed, ensemble_embed):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        backup = embed.clone()

        if parse_args.L2:
            l2_loss = nn.MSELoss()
        else:
            kl_loss = nn.KLDivLoss(reduction="none")
        weights = torch.Tensor(ensemble_embed.size())
        for idx in range(self.weights.size(0)):
            weights[idx,:,:,:] = self.weights[idx]
        embed = torch.log(embed).unsqueeze(0).expand_as(ensemble_embed)
        KL_loss = weights.detach() * kl_loss(embed, ensemble_embed.detach())
        KL_loss = torch.mean(KL_loss.permute(1,0,2), dim=(1,2))
        loss = nll_loss - self.regularize * KL_loss
        return loss.mean()

class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SubnetL1RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, temperature=1.0):
        l1_accum = 0.0
        for n, p in model.named_parameters():
            if n.endswith("scores"):
                l1_accum += (p*temperature).sigmoid().sum()

        return l1_accum
