import torch
import torch.nn as nn
from ..utils.utils import Eval, cosLr, loader_to
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
        if not num_total: raise KeyError("dataloader is empty")
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













# Convenience function for that covers a common use case of training the model using
#   the cosLr schedule, and logging the outcome and returning the results

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
            trainset.default_aug_layers(),
            cfg['network'](num_classes=trainset.num_classes,**cfg['net_config']).to(device)
        )
        dataloaders = {}
        dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
        dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}
        opt_constr = lambda params: torch.optim.SGD(params, **cfg['opt_config'])
        lr_sched = cosLr(cfg['num_epochs'])
        return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)
    