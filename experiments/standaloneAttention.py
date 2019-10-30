import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import os
#import oil.augLayers as augLayers
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial
from oil.datasetup.datasets import CIFAR10#, C10augLayers
from oil.datasetup.dataloaders import getLabLoader
from oil.architectures.img_classifiers import layer13s,layer13d,layer13a,layer13at,layer13pc
from oil.architectures.img_classifiers import resnetpc,colorEquivariantLayer13pc,colorEquivariantResnetpc
import os
from oil.tuning.study import Study, train_trial

if __name__=='__main__':
    # x = torch.randn((32,3,32,32)).cuda()
    # model = layer13pcs(num_classes=10).cuda()
    # for i in range(10):
    #     output= model(x)
    #     output.sum().backward()
    config_spec = {'num_epochs':100,'loader_config':{'amnt_dev':5000,'lab_BS':50, 'amnt_labeled':1},
        'network':[colorEquivariantResnetpc],'net_config':{'k':[8,12,16],'ksize':[3.66],'num_layers':[8]},
        'opt_config': {'lr':[.003],'optim':optim.Adam},
        'trainer_config':{'log_dir':lambda cfg: os.path.expanduser(f"~/tb-experiments/pointconv_colorlarge_{cfg['network']}_\
                            larger_lr{cfg['opt_config']['lr']}_k{cfg['net_config']['k']}_\
                            ks{cfg['net_config']['ksize']}_L{cfg['net_config']['num_layers']}/"),'log_args':{'timeFrac':.2}}}
    # config_spec = {'num_epochs':200,'loader_config':{'amnt_dev':5000,'lab_BS':50},
    #     'network':[colorEquivariantResnetpc],'net_config':{'k':8,'num_layers':6,'ksize':3.66},
    #     'opt_config': {'optim':optim.Adam,'lr':[.0003]},
    #     'trainer_config':{'log_dir':lambda cfg: os.path.expanduser(f"~/tb-experiments/pointconv_colorequiv_resnet\
    #                         larger_lr{cfg['opt_config']['lr']}_k{cfg['net_config']['k']}_\
    #                         ks{cfg['net_config']['ksize']}_L{cfg['net_config']['num_layers']}/"),'log_args':{'timeFrac':.2}}}
    Trial = simpleClassifierTrial#()#strict=True)
    thestudy = Study(Trial,config_spec,study_name="pointconv_color")
    thestudy.run(ordered=False)
    covars = thestudy.covariates()#
    covars['Dev_Acc'] = thestudy.outcomes['Dev_Acc'].values
    print(covars.drop(['log_suffix','saved_at'],axis=1))
