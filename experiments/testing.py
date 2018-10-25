import numpy as np
import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import os, copy
#from ray.tune.variant_generator import to_argv
#import oil.augLayers as augLayers
from oil.cnnTrainer import CnnTrainer
from oil.datasets import CIFAR10, C10augLayers
from oil.networkparts import layer13
from oil.schedules import cosLr
import random

def makeTrainer(extra_config, net_config, opt_config, sched_config, trainer_config):
    CNN = oil.networkparts.layer13(**net_config)
    fullCNN = nn.Sequential(oil.datasets.C10augLayers(),CNN)
    datasets = oil.datasets.CIFAR10(aug=False)
    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_lambda = oil.schedules.cosLr(**sched_config)
    args = (fullCNN, datasets, opt_constr, lr_lambda)
    return oil.cnnTrainer.CnnTrainer(*args,**trainer_config)


extra_config =      {'numEpochs':150, 'expt_name':'baseline'}
net_config =        {'numClasses':10}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':extra_config['numEpochs'],'cycle_mult':1}
trainer_config =    {'amntLab':50000, 'amntDev':5000,'dataseed':0,
                    'lab_BS':50, 'ul_BS':50, 'num_workers':4, 'log':True, 
                    }
trainer_config['description'] = "13Layer network, {} dev".format(trainer_config['amntDev'])
trainer_config['savedir'] = '/home/maf388/tb-experiments/'+extra_config['expt_name']+'/'
base_configs = [extra_config, net_config, opt_config, sched_config, trainer_config]

tunable_hypers =    {'weight_decay': 10**random.uniform(-6,-8),
                    'lab_BS':random.choice([32,50,64])}

def to_str(spec):
    return str(hash(frozenset(spec.items())))

def to_str2(dic):
    return "".join(["_{}-{:.3g}".format(key,value) for key,value in dic.items()][:10])
print(to_str2(tunable_hypers))