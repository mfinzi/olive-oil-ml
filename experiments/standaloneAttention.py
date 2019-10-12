import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import os
#import oil.augLayers as augLayers
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
from oil.datasetup.datasets import CIFAR10#, C10augLayers
from oil.datasetup.dataloaders import getLabLoader
from oil.architectures.img_classifiers import layer13s,layer13d,layer13a,layer13at

import os
from oil.tuning.study import Study, train_trial

if __name__=='__main__':
    config_spec = {'num_epochs':100,'loader_config':{'amnt_dev':0,'lab_BS':32},
        'network':[layer13at],'net_config':{'k':64,'num_heads':8,'ksize':5},'opt_config': {'lr':.03},
        'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/standalone_transformer2/'),'log_args':{'timeFrac':.5}}}
    Trial = simpleClassifierTrial(strict=True)
    cutout_study = Study(Trial,config_spec)
    cutout_study.run()