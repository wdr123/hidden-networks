import sys

import torch
import torch.nn as nn
from utils.conv_type import FixedSubnetConv
import pathlib
from args import args as parse_args
import models
import torch.backends.cudnn as cudnn


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

def set_gpu(model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if parse_args.gpu is not None:
        torch.cuda.set_device(parse_args.gpu)
        model = model.cuda(parse_args.gpu)
    elif parse_args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {parse_args.multigpu} gpus")
        torch.cuda.set_device(parse_args.multigpu[0])
        parse_args.gpu = parse_args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=parse_args.multigpu).cuda(
            parse_args.multigpu[0]
        )

    cudnn.benchmark = True

    return model

def set_model_prune_rate(model, prune_rate):
    print(f"==> Setting prune rate of network to {prune_rate}")

    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            m.set_prune_rate(prune_rate)
            print(f"==> Setting prune rate of {n} to {prune_rate}")

def pretrained(search_dir, model):
    if search_dir.exists():
        if parse_args.adaboost:
            pretrained_path = search_dir / "epoch_199.state"
        else:
            pretrained_path = search_dir / "model_best.pth"
        print("=> loading best pretrained weights from '{}'".format(pretrained_path))
        pretrained = torch.load(
            pretrained_path,
            map_location=torch.device("cuda:{}".format(parse_args.gpu)),
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
        print("=> no pretrained weights found at '{}'".format(search_dir))
        return False

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()

    return True

class Ensemble(nn.Module):
    def __init__(self, arch):
        super(Ensemble, self).__init__()
        self.baseLearner = []
        self.predictions = []
        self.embedding = []
        self.learnerCount = 0
        self.subnet_init = None
        self.dataset = parse_args.name
        backup = parse_args.conv_type
        parse_args.conv_type = 'FixedSubnetConv'
        config = pathlib.Path(parse_args.config).stem

        if parse_args.ensemble_subnet_init is None:
            self.subnet_init = ["unsigned_constant", "signed_constant", "kaiming_normal", "kaiming_uniform"]
        else:
            self.subnet_init = parse_args.ensemble_subnet_init

        if arch.startswith('e'):
            self.arch = arch[1:]
        elif parse_args.KL:
            self.arch = arch[1:]
        else:
            raise ValueError("Either KL mode or Ensemble mode please!")

        runs_count = 0

        if parse_args.adaboost:
            for idx in range(3):
                search_dir = pathlib.Path(
                    f"runs/resnet101-ukn-unsigned/cResNet101_CIFAR100_{idx+1}/prune_rate=0.05/0/checkpoints")

                cur_model = models.__dict__[self.arch]()
                set_model_prune_rate(cur_model, parse_args.prune_rate)
                set_gpu(cur_model)
                if parse_args.freeze_weights:
                    freeze_model_weights(cur_model)

                if pretrained(search_dir, cur_model):
                    cur_model.eval()
                    self.baseLearner.append(cur_model)
                    self.learnerCount += 1
        else:
            for idx in range(len(self.subnet_init)):
                search_dir = pathlib.Path(
                    f"edge/{config}/{parse_args.name}/prune_rate={parse_args.prune_rate}/subnet_init={self.subnet_init[idx]}/checkpoints")
                if parse_args.KL or parse_args.L2:
                    search_dir = pathlib.Path(
                        f"runs1_KL/{config}/{parse_args.name}/prune_rate={parse_args.prune_rate}/subnet_init={self.subnet_init[idx]}/checkpoints")
                if not search_dir.exists():
                    search_dir = pathlib.Path(
                        f"edge/{config}/{parse_args.name}/prune_rate={parse_args.prune_rate}/subnet_init=kaiming_uniform/{runs_count}/checkpoints")
                    runs_count += 1

                cur_model = models.__dict__[self.arch]()
                set_model_prune_rate(cur_model, parse_args.prune_rate)
                set_gpu(cur_model)
                if parse_args.freeze_weights:
                    freeze_model_weights(cur_model)

                if pretrained(search_dir, cur_model):
                    cur_model.eval()
                    self.baseLearner.append(cur_model)
                    self.learnerCount += 1


        parse_args.conv_type = backup

    def empty(self, ):
        del self.baseLearner
        del self.predictions
        self.baseLearner = []
        self.predictions = []
        self.embedding = []
        self.learnerCount = 0
        self.subnet_init = None

    def forward(self, x):
        with torch.no_grad():
            if self.learnerCount > 0:
                for cur_model in self.baseLearner:
                    cur_predict = cur_model(x)
                    self.predictions.append(cur_predict)
            else:
                raise ValueError("leaner counter equal to 0, please add new base learner!"
                                 "")
        out = torch.mean(torch.stack(self.predictions), dim=0)
        del self.predictions
        self.predictions = []
        return out.flatten(1)

    def get_embedding(self, x):
        with torch.no_grad():
            if self.learnerCount > 0:
                for cur_model in self.baseLearner:
                    cur_embed = cur_model.embedding(x)
                    self.embedding.append(cur_embed)
            else:
                raise ValueError("leaner counter equal to 0, please add new base learner!")

        out = torch.stack(self.embedding)
        del self.embedding
        self.embedding = []
        return out.detach()



def ecResNet18():
    return Ensemble("ecResNet18")

def ecResNet50():
    return Ensemble("ecResNet50")

def ecResNet101():
    return Ensemble("ecResNet101")



