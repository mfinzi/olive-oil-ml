import torch
import torch.nn as nn
from oil.utils.utils import Eval, cosLr
from oil.model_trainers.trainer import Trainer

class Classifier(Trainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    
    def loss(self, minibatch, model = None):
        """ Standard cross-entropy loss """
        x,y = minibatch
        if model is None: model = self.model
        class_weights = self.dataloaders['train'].dataset.class_weights
        ignored_index = self.dataloaders['train'].dataset.ignored_index
        criterion = nn.CrossEntropyLoss(weight=class_weights,ignore_index=ignored_index)
        return criterion(model(x),y)

    def metrics(self,loader):
        acc = lambda mb: self.model(mb[0]).max(1)[1].type_as(mb[1]).eq(mb[1]).cpu().data.numpy().mean()
        return {'Acc':self.evalAverageMetrics(loader,acc)}

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










# Convenience function for that covers a common use case of training the model using
#   the cosLr schedule, and logging the outcome and returning the results
from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice
from oil.tuning.study import train_trial
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.datasets import CIFAR10
from oil.architectures.img_classifiers import layer13s
import collections

def simpleClassifierTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'dataset': CIFAR10,'network':layer13s,'net_config': {},
            'loader_config': {'amnt_dev':5000,'lab_BS':50, 'pin_memory':True,'num_workers':2},
            'opt_config':{'lr':.1, 'momentum':.9, 'weight_decay':1e-4,'nesterov':True},
            'num_epochs':100,'trainer_config':{},'parallel':False,
            }
        recursively_update(cfg,config)
        trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        fullCNN = torch.nn.Sequential(
            trainset.default_aug_layers(),
            cfg['network'](num_classes=trainset.num_classes,**cfg['net_config']).to(device)
        )
        if cfg['parallel']: fullCNN = multigpu_parallelize(fullCNN,cfg)
        dataloaders = {}
        dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
        dataloaders['Train'] = islice(dataloaders['train'],10000//cfg['loader_config']['lab_BS'])
        if len(dataloaders['dev'])==0:
            testset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),train=False)
            dataloaders['test'] = DataLoader(testset,batch_size=cfg['loader_config']['lab_BS'],shuffle=False)
        dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
        opt_constr = lambda params: torch.optim.SGD(params, **cfg['opt_config'])
        lr_sched = cosLr(cfg['num_epochs'])
        return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)
    

if __name__=='__main__':
    Trial = simpleClassifierTrial(strict=True)
    Trial({'num_epochs':100,'loader_config':{'amnt_dev':0}})