import torch
import torch.nn as nn
from oil.utils.utils import Eval, cosLr, export
from oil.model_trainers.trainer import Trainer

@export
class Classifier(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    
    def loss(self, minibatch, model = None):
        """ Standard cross-entropy loss """
        x,y = minibatch
        if model is None: model = self.model
        try: class_weights = self.dataloaders['train'].dataset.class_weights
        except AttributeError: class_weights=None
        try: ignored_index = self.dataloaders['train'].dataset.ignored_index
        except AttributeError: ignored_index=-100
        criterion = nn.CrossEntropyLoss(weight=class_weights,ignore_index=ignored_index)
        return criterion(model(x),y)

    def metrics(self,loader):
        acc = lambda mb: self.model(mb[0]).max(1)[1].type_as(mb[1]).eq(mb[1]).cpu().data.numpy().mean()
        return {'Acc':self.evalAverageMetrics(loader,acc)}

@export
class Regressor(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def loss(self, minibatch, model = None):
        """ Standard cross-entropy loss """
        x,y = minibatch
        if model is None: model = self.model
        return nn.MSELoss()(model(x),y)

    def metrics(self,loader):
        mse = lambda mb: nn.MSELoss()(self.model(mb[0]),mb[1]).cpu().data.numpy()
        return {'MSE':self.evalAverageMetrics(loader,mse)}
