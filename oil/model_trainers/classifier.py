import torch
import torch.nn as nn
from ..utils.utils import Eval
from .trainer import Trainer

class Classifier(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def loss(self, minibatch, model = None):
        """ Standard cross-entropy loss """
        x,y = minibatch
        if model is None: model = self.model
        return nn.CrossEntropyLoss()(model(x),y)

    def batchAccuracy(self, minibatch, model = None):
        """ Evaluates the minibatch accuracy """
        if model is None: model = self.model
        with Eval(model), torch.no_grad():
            x, y = minibatch
            predictions = model(x).max(1)[1].type_as(y)
            accuracy = predictions.eq(y).cpu().data.numpy().mean()
        return accuracy
    
    def getAccuracy(self, loader, model = None):
        """ Gets the full dataset accuracy evaluated on the data in loader """
        num_correct, num_total = 0, 0
        for minibatch in loader:
            mb_size = minibatch[1].size(0)
            batch_acc = self.batchAccuracy(minibatch, model=model)
            num_correct += batch_acc*mb_size
            num_total += mb_size
        return num_correct/num_total

    def logStuff(self, i, minibatch=None):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """
        step = i+1 + (self.epoch+1)*len(self.dataloaders['train'])

        metrics = {}
        #if minibatch: metrics['Train_Acc(Batch)'] = self.batchAccuracy(minibatch)
        try: metrics['Test_Acc'] = self.getAccuracy(self.dataloaders['test'])
        except KeyError: pass
        try: metrics['Dev_Acc'] = self.getAccuracy(self.dataloaders['dev'])
        except KeyError: pass
        self.logger.add_scalars('metrics', metrics, step)
        super().logStuff(i,minibatch)