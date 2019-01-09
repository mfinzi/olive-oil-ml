import torch
import torch.nn as nn
from ..utils.utils import Eval
from .trainer import Trainer

class SequenceTrainer(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def loss(self, x, y):
        """ Standard cross-entropy loss """
        return nn.CrossEntropyLoss()(self.model(x),y)
    
    def bleu(self, loader):
        pass

    def logStuff(self, i, minibatch):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """
        step = i+1 + (self.epoch+1)*len(self.dataloaders['train'])

        metrics = {}
        #metrics['Train_Acc(Batch)'] = self.batchAccuracy(minibatch)
        try: metrics['Test_Acc'] = self.getAccuracy(self.dataloaders['test'])
        except KeyError: pass
        try: metrics['Dev_Acc'] = self.getAccuracy(self.dataloaders['dev'])
        except KeyError: pass
        self.logger.add_scalars('metrics', metrics, step)

        schedules = {}
        schedules['lr'] = self.lr_scheduler.get_lr()[0]
        self.logger.add_scalars('schedules', schedules, step)
        super().logStuff(i,minibatch)