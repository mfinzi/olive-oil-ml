


import os
import numpy as np
import torch
from oil.utils.utils import cosLr, loader_to
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.model_trainers.classifier import Classifier
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.augLayers import RandomErasing
from oil.architectures.img_classifiers import WideResNet28x10,layer13s
from oil.tuning.study import Study, train_trial
from oil.tuning.configGenerator import uniform,logUniform,sample_config
#import oil.augLayers as augLayers

config_spec = {
    'dataset': [CIFAR10,CIFAR100],
    'network': layer13s,'net_config': {},
    'loader_config': {'amnt_dev':5000,'lab_BS':64},
    'opt_config':{'lr':.2, 'momentum':.9, 'weight_decay':1e-4},
    'num_epochs':200,
    'cutout_config':{'p':uniform(.3,1),'af':logUniform(.1,.5),'ar':logUniform(1,3)},
    'trainer_config':{'log_args':{'no_print':True}}
}
log_dir_base = os.path.expanduser('~/tb-experiments/cutout_wrn')
config_spec['trainer_config']['log_dir'] = lambda cfg:log_dir_base+\
        '/{}'.format(cfg['dataset'])

def makeTrainer(cfg):
    trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']))
    device = torch.device('cuda')
    fullCNN = torch.nn.Sequential(
        trainset.default_aug_layers(),
        RandomErasing(**cfg['cutout_config']),
        cfg['network'](num_classes=trainset.num_classes,**cfg['net_config']).to(device)
    )
    dataloaders = {}
    dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
    dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

    opt_constr = lambda params: torch.optim.SGD(params, **cfg['opt_config'])
    lr_sched = cosLr(cfg['num_epochs'])
    return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'],log_args={'no_print':True})

do_trial = train_trial(makeTrainer)
cutout_study = Study(do_trial,config_spec, slurm_cfg={'time':'24:00:00'})
cutout_study.run(num_trials=30,max_workers=10)