from .classifier import Classifier
from ..utils.losses import softmax_mse_loss, softmax_mse_loss_both
from ..utils.utils import Eval, izip, icycle
import torch.nn as nn

class PiModel(Classifier):
    def __init__(self, *args, cons_weight=15,
                     **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'cons_weight':cons_weight})
        self.dataloaders['train'] = izip(icycle(self.dataloaders['train']),self.dataloaders['_unlab'])

    def unlabLoss(self, x_unlab):
        logits1 = self.model(x_unlab)
        logits2 = self.model(x_unlab)
        cons_loss =  softmax_mse_loss(logits1, logits2.detach())
        return cons_loss

    def loss(self, minibatch):
        (x_lab, y_lab), x_unlab = minibatch
        unlab_loss = self.unlabLoss(x_unlab)*float(self.hypers['cons_weight'])
        lab_loss = nn.CrossEntropyLoss()(self.model(x_lab),y_lab)
        return lab_loss + unlab_loss

    def logStuff(self, step, minibatch=None):
        if minibatch:
            extra_metrics = {'Unlab_loss(batch)':self.unlabLoss(minibatch[1]).cpu().item()}
            self.logger.add_scalars('metrics',extra_metrics,step)
        super().logStuff(step, minibatch)