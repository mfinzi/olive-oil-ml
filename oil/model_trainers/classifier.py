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
from torch.optim import SGD
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice
from oil.tuning.study import train_trial
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.datasets import CIFAR10
from oil.architectures.img_classifiers import layer13s,layer13
from oil.utils.parallel import try_multigpu_parallelize
from collections import Iterable,defaultdict
import collections

def makeTrainer(config):
    cfg = {
        'dataset': CIFAR10,'network':layer13,'net_config': {},
        'loader_config': {'amnt_dev':5000,'lab_BS':64, 'pin_memory':True,'num_workers':0},
        'opt_config':{'optim':SGD}, 
        'num_epochs':100,'trainer_config':{},
        }
    if config.get('opt_config',{}).get('optim',SGD)==SGD: cfg['opt_config'].update({'lr':.1, 'momentum':.9, 'weight_decay':1e-4,'nesterov':True})
    recursively_update(cfg,config)
    
    trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']))
    device = torch.device('cuda')
    fullCNN = torch.nn.Sequential(
        trainset.default_aug_layers(),
        cfg['network'](num_classes=trainset.num_classes,**cfg['net_config'])
    ).to(device)
    fullCNN = try_multigpu_parallelize(fullCNN,cfg,scalelr=False)#cfg['opt_config']['optim']==SGD)

    dataloaders = {}
    dataloaders['train'], dataloaders['Dev'] = getLabLoader(trainset,**cfg['loader_config'])
    dataloaders['Train'] = islice(dataloaders['train'],10000//cfg['loader_config']['lab_BS'])
    if len(dataloaders['Dev'])==0:
        testset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),train=False)
        dataloaders['Test'] = DataLoader(testset,batch_size=cfg['loader_config']['lab_BS'],
                        shuffle=False,num_workers=0,pin_memory=False)
    dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}

    optim = cfg['opt_config'].pop('optim')
    opt_constr = lambda params: optim(params, **cfg['opt_config'])
    lr_sched = cosLr(cfg['num_epochs'])
    return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])

simpleClassifierTrial = train_trial(makeTrainer)

if __name__=='__main__':
    Trial = simpleClassifierTrial
    Trial({'num_epochs':100,'loader_config':{'amnt_dev':0,'lab_BS':64}})
