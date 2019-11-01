import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import os
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial,base_cfg
from oil.datasetup.datasets import CIFAR10#, C10augLayers
from oil.datasetup.dataloaders import getLabLoader
from oil.architectures.img_classifiers import layer13s,layer13d,layer13a,layer13at,layer13pc,pWideResNet
from oil.architectures.img_classifiers import resnetpc,colorEquivariantLayer13pc,colorEquivariantResnetpc
from oil.tuning.args import argupdated_config
import os
from oil.tuning.study import Study, train_trial


# config_spec = {'num_epochs':100,'loader_config':{'amnt_dev':5000,'lab_BS':50,'amnt_labeled':1},
#     'network':[resnetpc],'net_config':{'k':[16],'ksize':[3.66],'num_layers':8},
#     'opt_config': {'lr':[3e-3],'optim':optim.Adam},
#     'trainer_config':{'log_dir':lambda cfg: os.path.expanduser(f"~/tb-experiments/pointconv_{cfg['network']}_\
#                     larger_lr{cfg['opt_config']['lr']}_k{cfg['net_config']['k']}_\
#                     ks{cfg['net_config']['ksize']}_L{cfg['net_config']['num_layers']}/"),'log_args':{'timeFrac':.2}}}
config_spec = {'num_epochs':100,'loader_config':{'amnt_dev':500,'lab_BS':20,'amnt_labeled':500+5000},
    'network':[pWideResNet],'net_config':{'depth':16,'widen_factor':10,'drop_rate':0},
    'opt_config': {'lr':[0.04],'optim':optim.SGD},
    'trainer_config':{'log_dir':lambda cfg: os.path.expanduser(f"~/tb-experiments/pointconv_{cfg['network']}_\
                    larger_lr{cfg['opt_config']['lr']}/"),'log_args':{'timeFrac':.2}}}
# config_spec = {'num_epochs':200,'loader_config':{'amnt_dev':5000,'lab_BS':50},
#     'network':[colorEquivariantResnetpc],'net_config':{'k':8,'num_layers':6,'ksize':3.66},
#     'opt_config': {'optim':optim.Adam,'lr':[.0003]},
#     'trainer_config':{'log_dir':lambda cfg: os.path.expanduser(f"~/tb-experiments/pointconv_colorequiv_resnet\
#                         larger_lr{cfg['opt_config']['lr']}_k{cfg['net_config']['k']}_\
#                         ks{cfg['net_config']['ksize']}_L{cfg['net_config']['num_layers']}/"),'log_args':{'timeFrac':.2}}}
Trial = simpleClassifierTrial#()#strict=True)
thestudy = Study(Trial,argupdated_config({**base_cfg,**config_spec}),study_name="pointconv")
thestudy.run()
covars = thestudy.covariates()#
covars['Dev_Acc'] = thestudy.outcomes['Dev_Acc'].values
print(covars.drop(['log_suffix','saved_at'],axis=1))
