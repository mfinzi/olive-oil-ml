import torch, torchvision,dill
import torch.optim as optim
import torch.nn as nn
import os
#import oil.augLayers as augLayers
from oil.model_trainers.classifier import Classifier
from oil.datasetup.datasets import CIFAR100
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.augLayers import RandomErasing,Cutout
from oil.architectures.img_classifiers.networkparts import layer13
from oil.utils.utils import cosLr, loader_to
from oil.tuning.configGenerator import uniform,logUniform,sample_config

train_epochs = 100
DataSet = CIFAR100
net_config =        {'numClasses':DataSet.num_classes}
loader_config =     {'amnt_dev':5000,'lab_BS':50}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':train_epochs}
cutout_config = {'p':uniform(.3,1)(0),'af':logUniform(.1,.5)(0),'ar':logUniform(1,3)(0)}
trainer_config =    {}

trainer_config['log_dir'] = os.path.expanduser('~/tb-experiments/c100cutout2/')

def makeTrainer():
    trainset = DataSet('~/datasets/{}/'.format(DataSet))
    device = torch.device('cuda')
    fullCNN = nn.Sequential(
        trainset.default_aug_layers(),
        RandomErasing(**cutout_config),
        layer13(**net_config).to(device)
    )
    
    dataloaders = {}
    dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**loader_config)
    dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_sched = cosLr(**sched_config)
    return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**trainer_config)

trainer = makeTrainer()
trainer.train(train_epochs)
#torch.save(trainer,"c100trainer.t",pickle_module=dill)