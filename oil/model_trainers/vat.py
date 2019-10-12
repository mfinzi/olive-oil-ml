import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.nn.functional as F
import torchvision.utils as vutils
import itertools
import torch
import numpy as np
import torch.nn as nn

from .piModel import PiModel
from ..utils.losses import softmax_mse_loss, softmax_mse_loss_both
from ..utils.utils import Eval, izip, icycle

def _l2_normalize(d):
    d /= (1e-12 + torch.max(torch.abs(d)))
    d /= norm(d)
    return d

def norm(d,keepdim=True):
    if len(d.shape)==4:
        norm = torch.sqrt(1e-6+(d**2).sum(3).sum(2).sum(1))
        return norm[:,None,None,None] if keepdim else norm
    elif len(d.shape)==2:
        norm = torch.sqrt(1e-6+(d**2).sum(1))
        return norm[:,None] if keepdim else norm
    else:
        assert False, "only supports 0d and 2d now"

def kl_div_withlogits(p_logits, q_logits):
    kl_div = nn.KLDivLoss(size_average=True).cuda()
    LSM = nn.LogSoftmax(dim=1)
    SM = nn.Softmax(dim=1)
    return kl_div(LSM(q_logits), SM(p_logits))

def cross_ent_withlogits(p_logits,q_logits):
    LSM = nn.LogSoftmax(dim=1).cuda()
    SM = nn.Softmax(dim=1)
    return -1*(SM(p_logits)*LSM(q_logits)).sum(dim=1).mean(dim=0)

class Vat(PiModel):
    def __init__(self, *args, cons_weight=.3, advEps=32, entMin=True, **kwargs):
        super().__init__(*args,**kwargs)
        self.hypers.update({'cons_weight':cons_weight,'advEps':advEps, 'entMin':entMin})

    def unlabLoss(self, x_unlab):
        """ Calculates LDS loss according to https://arxiv.org/abs/1704.03976 """
        wasTraining = self.model.training; self.model.train(False)

        r_adv = self.hypers['advEps'] * self.getAdvPert(self.model, x_unlab)
        perturbed_logits = self.model(x_unlab + r_adv)
        logits = self.model(x_unlab).detach()
        unlabLoss = kl_div_withlogits(logits, perturbed_logits)/(self.hypers['advEps'])**2
        self.model.train(wasTraining)
        return unlabLoss

    @staticmethod
    def getAdvPert(model, X, powerIts=1, xi=1e-6):
        wasTraining = model.training; model.train(False)

        ddata = torch.randn(X.size()).to(X.device)
        # calc adversarial direction
        d = Variable(xi*_l2_normalize(ddata), requires_grad=True)
        logit_p = model(X).detach()
        perturbed_logits = model(X + d)
        adv_distance = kl_div_withlogits(logit_p, perturbed_logits)
        d_grad = torch.autograd.grad(adv_distance,d)[0]
        #model.zero_grad()
        #ddata = d.grad.data
        #print("2 max = %.4E, min = %.4E"%(torch.max(norm(ddata)),torch.min(norm(ddata))))
        #model.zero_grad()
        #print("3 max = %.4E, min = %.4E"%(torch.max(norm(ddata)),torch.min(norm(ddata))))
        model.train(wasTraining)
        return _l2_normalize(d_grad)#.detach()

    # def logStuff(self, step, minibatch=None):
    #     if minibatch:
    #         x_unlab = trainData[1]; someX = x_unlab[:16]
    #         r_adv = self.hypers['advEps'] * self.getAdvPert(self.model, someX)
    #         adversarialImages = (someX + r_adv).cpu().data
    #         imgComparison = torch.cat((adversarialImages, someX.cpu().data))
    #         self.writer.add_image('adversarialInputs',
    #                 vutils.make_grid(imgComparison,normalize=True,range=(-2.5,2.5)), step)
    #         self.metricLog.update({'Unlab_loss(batch)':
    #                 self.unlabLoss(x_unlab).cpu().data[0]})
    #     super().logStuff(step, minibatch)