import torch
import torch.nn as nn
from utils.conv_type import FixedSubnetConv
import pathlib
from args import args as parse_args
import models



def pretrained(search_dir, model):
    if search_dir.exists():
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
            self.arch = arch
        else:
            raise ValueError("Either KL mode or Ensemble mode please!")

        for idx in range(len(self.subnet_init)):
            search_dir = pathlib.Path(
                f"{parse_args.log_dir}/{config}/{parse_args.name}/prune_rate={parse_args.prune_rate}/subnet_init={self.subnet_init[idx]}/checkpoints")
            cur_model = models.__dict__[self.arch]()
            if pretrained(search_dir, cur_model):
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
        if self.learnerCount > 0:
            for cur_model in self.baseLearner:
                cur_predict = cur_model(x)
                self.predictions.append(cur_predict)
        else:
            raise ValueError("leaner counter equal to 0, please add new base learner!"
                             "")
        out = torch.max(torch.stack(self.predictions), dim=0)
        return out.flatten(1)

    def embedding(self, x):
        if self.learnerCount > 0:
            for cur_model in self.baseLearner:
                cur_embed = cur_model.embedding(x)
                self.predictions.append(cur_embed)
        else:
            raise ValueError("leaner counter equal to 0, please add new base learner!")

        out = torch.stack(self.predictions)
        return out.detach()



def ecResNet18():
    return Ensemble("cResNet18")

def ecResNet50():
    return Ensemble("cResNet50")

def ecResNet101():
    return Ensemble("cResNet101")



