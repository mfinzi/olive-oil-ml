import torch
import numpy as np
import torch.nn as nn
from contextlib import contextmanager

from oil.cnnTrainer import CnnTrainer
from oil.losses import softmax_mse_loss, softmax_mse_loss_both
from oil.schedules import sigmoidConsRamp

class SPiTrainer(CnnTrainer):
    def __init__(self, *args, cons_weight=200, eps=, rampup_epochs=20,
                     **kwargs):
        def initClosure():
            self.hypers.update({'cons_weight':cons_weight,
                                'rampup_epochs':rampup_epochs})
            self.numBatchesPerEpoch = len(self.unl_train)
            #self.consRamp = sigmoidConsRamp(rampup_epochs)
            self.train_iter = zip(iter(self.lab_train), iter(self.unl_train))
        super().__init__(*args, extraInit = initClosure, **kwargs)

    # def getTrainIter(self):
    #     return zip(iter(self.lab_train), iter(self.unl_train))
    @contextmanager
    def weightPerturbation(self, model, eps):
        """context manager for additive gaussian weight perturbations """
        noise = [torch.randn(p.size()).cuda()*eps for p in model.parameters()]
        for p, pz in zip(model.parameters(), noise):
            p.data += pz
        yield noise
        for p, pz in zip(model.parameters(), noise):
            p.data -= pz

    def unlabLoss(self, x_unlab):
        weight = self.hypers['cons_weight']#*self.consRamp(self.epoch)
        eps = self.hypers['eps']
        with self.weighterturbation(self.CNN, eps):
            pred1 = self.CNN(x_unlab)
        pred2 = self.CNN(x_unlab)
        cons_loss =  weight*softmax_mse_loss(pred1, pred2.detach())/(x_unlab.size(0)*eps**2)
        return cons_loss

    def loss(self, *data):
        (x_lab, y_lab), (x_unlab, _) = data
        lab_loss = nn.CrossEntropyLoss()(self.CNN(x_lab),y_lab)
        unlab_loss = self.unlabLoss(x_unlab)# + self.unlabLoss(x_lab)
        return lab_loss + unlab_loss

    def getLabeledXYonly(self, trainingData):
        labeledData, unlabeledData = trainingData
        return labeledData

    def logStuff(self, i, epoch, numEpochs, trainData):
        step = i + epoch*self.numBatchesPerEpoch
        if step%2000==0:
            self.metricLog['Unlab_loss(batch)'] = self.unlabLoss(trainData[1][0]).cpu()
        super().logStuff(i, epoch, numEpochs, trainData)