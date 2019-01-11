import torch, torchvision,dill
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
#import oil.augLayers as augLayers
from oil.model_trainers.classifier import Classifier
from oil.datasetup.datasets import CIFAR10,CIFAR100
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.augLayers import RandomErasing
from oil.architectures.img_classifiers.networkparts import layer13
from oil.utils.utils import cosLr, loader_to
from oil.tuning.configGenerator import uniform,logUniform,sampleFrom
from oil.tuning.slurmDispatch import TrainerFit, Study

config_spec = {
    'dataset': [CIFAR10,CIFAR100],
    'net_config': {'numClasses':sampleFrom(lambda cfg: cfg['dataset'].num_classes)},
    'loader_config': {'amnt_dev':5000,'lab_BS':50},
    'opt_config':{'lr':.1, 'momentum':.9, 'weight_decay':1e-4},
    'sched_config':{'cycle_length':50},
    'cutout_config':{'p':uniform(.4,1),'af':logUniform(.1,.5),'ar':logUniform(1,3)},
    'trainer_config':{}
}
log_dir_base = os.path.expanduser('~/tb-experiments/cutout/')
config_spec['trainer_config']['log_dir'] = sampleFrom(lambda cfg:log_dir_base+\
        '{}/{}/'.format(cfg['dataset'].__name__,np.random.randint(10**5)))

def makeTrainer(cfg):
    trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset'].__name__))
    device = torch.device('cuda')
    fullCNN = nn.Sequential(
        trainset.default_aug_layers(),
        RandomErasing(**cfg['cutout_config']),
        layer13(**cfg['net_config']).to(device)
    )
    dataloaders = {}
    dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
    dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

    opt_constr = lambda params: optim.SGD(params, **cfg['opt_config'])
    lr_sched = cosLr(**cfg['sched_config'])
    return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])

make_trial = lambda cfg: TrainerFit(makeTrainer,cfg['sched_config']['cycle_length'],cfg)
cutout_study = Study(make_trial,config_spec)
cutout_study.run(num_trials=20)