import torch
import torch.nn as nn
from ..utils.utils import Eval, cosLr, loader_to
from .trainer import Trainer

class Regressor(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def loss(self, minibatch, model = None):
        """ Standard cross-entropy loss """
        x,y = minibatch
        if model is None: model = self.model
        return nn.MSELoss()(model(x),y)

    def evalMSE(self, loader, model = None):
        """ Gets the full dataset MSE evaluated on the data in loader """
        return self.getAverageLoss(loader,model,nn.MSELoss())

    __metrics__ = {**Trainer.__metrics__,'MSE':evalMSE}
    