import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
import torch.nn.functional as F
import torchvision.utils as vutils
from oil.cnnTrainer import CnnTrainer
from oil.utils import to_var_gpu, prettyPrintLog
import itertools

def _l2_normalize(d):
    d /= (1e-25 + torch.max(torch.abs(d)))
    norm = torch.sqrt((d**2).sum(3).sum(2).sum(1) + 1e-12)
    d /= norm.view(-1,1,1,1)
    return d

def norm(d):
    norm = torch.sqrt((d**2).sum(3).sum(2).sum(1))
    return norm

def kl_div_withlogits(p_logits, q_logits):
    kl_div = nn.KLDivLoss(size_average=True).cuda()
    LSM = nn.LogSoftmax(dim=1)
    SM = nn.Softmax(dim=1)
    return kl_div(LSM(q_logits), SM(p_logits))

def cross_ent_withlogits(p_logits,q_logits):
    LSM = nn.LogSoftmax(dim=1).cuda()
    SM = nn.Softmax(dim=1)
    return -1*(SM(p_logits)*LSM(q_logits)).sum(dim=1).mean(dim=0)

class VatCnnTrainer(CnnTrainer):
    def __init__(self, *args, regScale=1, advEps=32, entMin=True,
                     **kwargs):
        def initClosure():
            self.hypers.update(
                {'regScale':regScale, 'advEps':advEps, 'entMin':entMin})
            self.numBatchesPerEpoch = len(self.unl_train)
        super().__init__(*args, extraInit = initClosure, **kwargs)

    def getTrainIter(self):
        return zip(iter(self.lab_train), iter(self.unl_train))

    def loss(self, *data):
        (x_lab, y_lab), (x_unlab, _) = data
        # Unlab loss must be calculated first!! Otherwise gradient computation 
        # of lab_loss will be messed up
        unlab_loss = self.unlabLoss(x_unlab)
        lab_loss = nn.CrossEntropyLoss()(self.CNN(x_lab),y_lab)
        return lab_loss + unlab_loss   

    def unlabLoss(self, x_unlab):
        """ Calculates LDS loss according to https://arxiv.org/abs/1704.03976 """
        wasTraining = self.CNN.training; self.CNN.train(False)

        r_adv = self.hypers['advEps'] * self.getAdvPert(self.CNN, x_unlab)
        perturbed_logits = self.CNN(x_unlab + r_adv)
        logits = self.CNN(x_unlab).detach()
        unlabLoss = self.hypers['regScale']*kl_div_withlogits(logits, perturbed_logits)

        self.CNN.train(wasTraining)
        return unlabLoss

    @staticmethod
    def getAdvPert(model, X, powerIts=1, xi=1e-6):
        wasTraining = model.training; model.train(False)

        ddata = torch.randn(X.size()).cuda()
        # calc adversarial direction
        d = Variable(xi*_l2_normalize(ddata), requires_grad=True)
        logit_p = model(X).detach()
        perturbed_logits = model(X + d)
        adv_distance = kl_div_withlogits(logit_p, perturbed_logits)
        adv_distance.backward()
        ddata = d.grad.data
        #print("2 max = %.4E, min = %.4E"%(torch.max(norm(ddata)),torch.min(norm(ddata))))
        model.zero_grad()
        #print("3 max = %.4E, min = %.4E"%(torch.max(norm(ddata)),torch.min(norm(ddata))))
        model.train(wasTraining)
        return Variable(_l2_normalize(ddata), requires_grad=False)

    def getLabeledXYonly(self, trainingData):
        labeledData, unlabeledData = trainingData
        return labeledData

    def logStuff(self, i, epoch, numEpochs, trainData):
        step = i + epoch*self.numBatchesPerEpoch
        if step%2000==0:
            x_unlab = trainData[1][0]; someX = x_unlab[:16]
            r_adv = self.hypers['advEps'] * self.getAdvPert(self.CNN, someX)
            adversarialImages = (someX + r_adv).cpu().data
            imgComparison = torch.cat((adversarialImages, someX.cpu().data))
            self.writer.add_image('adversarialInputs',
                    vutils.make_grid(imgComparison,normalize=True,range=(-2.5,2.5)), step)
            self.metricLog.update({'Unlab_loss(batch)':
                    self.unlabLoss(x_unlab).cpu().data[0]})
        super().logStuff(i, epoch, numEpochs, trainData)