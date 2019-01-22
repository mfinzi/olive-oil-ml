


import os
import numpy as np
import torch
from oil.utils.utils import cosLr, loader_to
from oil.datasetup.datasets import CIFAR10,CIFAR100, SVHN
from oil.model_trainers.classifier import Classifier
from oil.datasetup.datasets import CIFAR10
from oil.model_trainers.classifier import Classifier, simpleClassifierTrial
import oil.architectures.img_classifiers as models
from oil.tuning.configGenerator import sample_config
from oil.architectures.img_classifiers.networkparts import layer13
from oil.tuning.study import Study, train_trial
from oil.tuning.configGenerator import uniform,logUniform,sample_config
#import oil.augLayers as augLayers

#import pandas as pd
#pd.set_option('display.max_colwidth', 1000)
#pd.set_option('display.expand_frame_repr', False)
config_spec = {
    'dataset': [SVHN,CIFAR10,CIFAR100],
    'loader_config': {'amnt_dev':5000,'lab_BS':50},
    'opt_config':{'lr':.1, 'momentum':.9, 'weight_decay':1e-4},
    'num_epochs':100,
    'trainer_config':{}
}
log_dir_base = os.path.expanduser('~/tb-experiments/cutout/')
config_spec['trainer_config']['log_dir'] = lambda cfg:log_dir_base+\
        '{}/'.format(cfg['dataset'])

def makeTrainer(cfg):

    trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']))
    device = torch.device('cuda')
    fullCNN = torch.nn.Sequential(
        trainset.default_aug_layers(),
        RandomErasing(**cfg['cutout_config']),
        layer13(**cfg['net_config']).to(device)
    )
    dataloaders = {}
    dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
    dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

    opt_constr = lambda params: torch.optim.SGD(params, **cfg['opt_config'])
    lr_sched = cosLr(cfg['num_epochs'])
    return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])

do_trial = train_trial(makeTrainer)
cutout_study = Study(do_trial,config_spec, slurm_cfg={'time':'2:00:00'})
cutout_study.run(num_trials=10,max_workers=5)