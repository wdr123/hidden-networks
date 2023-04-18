import matplotlib.pyplot as plt
import torch
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def eval_calibration(output, labels, M=15, draw=False):
    """
    M: number of bins for confidence scores
    """

    probs = torch.nn.functional.softmax(output, dim=-1)
    predictions = torch.argmax(probs, dim=-1)
    confidences = torch.max(probs, dim=-1)[0]

    num_Bm = torch.zeros((M,), dtype=torch.int32)
    accs = torch.zeros((M,), dtype=torch.float32)
    confs = torch.zeros((M,), dtype=torch.float32)
    for m in range(M):
        interval = [m / M, (m+1) / M]
        Bm = torch.tensor((confidences > interval[0]) & (confidences <= interval[1])).nonzero()
        if len(Bm) > 0:
            acc_bin = torch.sum(predictions[Bm] == labels[Bm]) / len(Bm)
            conf_bin = torch.mean(confidences[Bm])
            # gather results
            num_Bm[m] = len(Bm)
            accs[m] = acc_bin
            confs[m] = conf_bin

    if draw:
        plt.bar(range(10), confs)
        plt.show()

    weighted_ece = torch.sum(torch.abs(accs - confs) * num_Bm / output.size(dim=0)) * 100

    return weighted_ece