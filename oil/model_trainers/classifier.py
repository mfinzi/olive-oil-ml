import torch
import torch.nn as nn
from ..utils.utils import Eval, cosLr
from .trainer import Trainer

class Classifier(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    
    def loss(self, minibatch, model = None):
        """ Standard cross-entropy loss """
        x,y = minibatch
        if model is None: model = self.model
        class_weights = self.dataloaders['train'].dataset.class_weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        return criterion(model(x),y)

    def evalAccuracy(self,loader,model=None):
        acc = lambda mb: model(mb[0]).max(1)[1].type_as(mb[1]).eq(mb[1]).float().mean()
        return self.evalAverageLoss(loader,model,acc)

    metrics = {**Trainer.metrics,'Acc':evalAccuracy}

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
        return self.evalAverageLoss(loader,model,nn.MSELoss())

    metrics = {**Trainer.metrics,'MSE':evalMSE}











# Convenience function for that covers a common use case of training the model using
#   the cosLr schedule, and logging the outcome and returning the results
from ..utils.utils import to_device_layer
from ..tuning.study import train_trial
from ..datasetup.dataloaders import getLabLoader
from ..datasetup.datasets import CIFAR10
from ..architectures.img_classifiers import layer13s
import collections

def recursively_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursively_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def simpleClassifierTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'dataset': CIFAR10,'network':layer13s,'net_config': {},
            'loader_config': {'amnt_dev':5000,'lab_BS':50},
            'opt_config':{'lr':.1, 'momentum':.9, 'weight_decay':1e-4},
            'num_epochs':100,'trainer_config':{},
            }
        recursively_update(cfg,config)
        trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']))
        device = torch.device('cuda')
        fullCNN = torch.nn.Sequential(
            to_device_layer(device),
            trainset.default_aug_layers(),
            cfg['network'](num_classes=trainset.num_classes,**cfg['net_config']).to(device)
        )
        dataloaders = {}
        dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
        opt_constr = lambda params: torch.optim.SGD(params, **cfg['opt_config'])
        lr_sched = cosLr(cfg['num_epochs'])
        return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)
    