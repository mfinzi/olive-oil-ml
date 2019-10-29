import torch, torchvision

import torch.optim as optim
import torch.nn as nn
import os
#import oil.augLayers as augLayers
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
from oil.datasetup.datasets import CIFAR10#, C10augLayers
from oil.datasetup.dataloaders import getLabLoader
from oil.architectures.img_classifiers import layer13s,layer13d
#from oil.utils.utils import cosLr, loader_to

# train_epochs = 100
# net_config =        {'num_classes':10}
# loader_config =     {'amnt_dev':5000,'lab_BS':50,'dataseed':0,'num_workers':4}
# opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
# sched_config =      {'cycle_length':train_epochs,'cycle_mult':1}
# trainer_config =    {}
# all_hypers = {**net_config,**loader_config,**opt_config,**sched_config,**trainer_config}

# trainer_config['log_dir'] = os.path.expanduser('~/tb-experiments/baseline_s/')

# def makeTrainer():
#     device = torch.device('cuda')
#     CNN = layer13s(**net_config).to(device)
#     fullCNN = nn.Sequential(C10augLayers(),CNN)
#     trainset, testset = CIFAR10(False, '~/datasets/cifar10/')

#     dataloaders = {}
#     dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**loader_config)
#     dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

#     opt_constr = lambda params: optim.SGD(params, **opt_config)
#     lr_sched = cosLr(**sched_config)
#     return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**trainer_config,tracked_hypers=all_hypers)

# trainer = makeTrainer()
# trainer.train(train_epochs)
# trainer.save_checkpoint()
import os
from oil.tuning.study import Study, train_trial

if __name__=='__main__':
    config_spec = {'num_epochs':5,'loader_config':{'amnt_dev':5000,'lab_BS':256},
        'network':[layer13s],'net_config':{'k':[32,64,96,50]},
        'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/deconv_bs_large/')}}
    Trial = simpleClassifierTrial
    cutout_study = Study(Trial,config_spec)
    cutout_study.run()
