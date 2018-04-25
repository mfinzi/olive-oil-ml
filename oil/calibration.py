import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from oil.utils import to_var_gpu


def comparison(net, devset, testset):
    devlogits, devlabels = getLogitsAndLabels(net, devset)

    standard_calib_rule = lambda logits: torch.max(F.softmax(logits, dim=1),1)
    temperature = get_temperature(devlogits, devlabels)
    temp_calib_rule = lambda logits: standard_calib_rule(logits/temperature)
    maxl_calib_rule = max_logit_calib_rule(devlogits, devlabels)

    for i, calib_rule in enumerate([standard_calib_rule, temp_calib_rule, maxl_calib_rule]):
        print("ECE %i: %.4f"%(i,evaluate_ECE(calib_rule, net, testset)))

def evaluate_ECE(calib_rule, net, testset, n_bins=15):
    logits, labels = getLogitsAndLabels(net, testset)
    confidences, predictions = calib_rule(logits)
    return _ECELoss(n_bins).cuda()(confidences, predictions, labels).cpu().data[0]

def getLogitsAndLabels(net, devset):
    logits_list = []
    labels_list = []
    for xy in devset:
        x,y = to_var_gpu(xy)
        logits_list.append(net(x).detach())
        labels_list.append(y)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    return logits, labels

def get_logit_mean_std(devlogits):
    max_logits, predictions = torch.max(devlogits, 1)
    max_logits_mean = max_logits.mean(dim=0)
    max_logits_std = max_logits.std(dim=0)
    return max_logits_mean, max_logits_std

def max_logit_calib_rule(devlogits, devlabels):
    mean, std = get_logit_mean_std(devlogits)
    bce_criterion = nn.BCEWithLogitsLoss()
    a = nn.Parameter(torch.ones(1)).cuda()
    b = nn.Parameter(torch.ones(1)).cuda()
    optimizer = optim.LBFGS([a,b], lr=0.01, max_iter=50)
    _, predictions = torch.max(devlogits, 1)
    correct = predictions.eq(devlabels)
    def eval_closure():
        loss = bce_criterion(a*(devlogits-mean)/std + b, correct)
        loss.backward()
        return loss
    optimizer.step(eval_closure)
    print("mean: %.3f, std: %.3f, a: %.3f, b: %.3f."%(mean.data[0],std.data[0],a.data[0],b.data[0]))
    calib_rule = lambda logits: F.sigmoid(a*(logits-mean)/std + b)
    return calib_rule


def get_temperature(devlogits, devlabels):
    # Next: optimize the temperature w.r.t. NLL
    nll_criterion = nn.CrossEntropyLoss()
    temperature = nn.Parameter(torch.ones(1)*1.5).cuda()
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)
    def eval_closure():
        loss = nll_criterion(devlogits/temperature, devlabels)
        loss.backward()
        return loss
    optimizer.step(eval_closure)
    print("temperature = %.3f"%temperature.data[0])
    return temperature
    
class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, confidences, predictions, labels):
        accuracies = predictions.eq(labels)

        ece = Variable(torch.zeros(1)).type_as(confidences)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower) * confidences.le(bin_upper)
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.data[0] > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin- accuracy_in_bin) * prop_in_bin

        return ece
