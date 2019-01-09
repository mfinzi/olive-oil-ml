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

train_epochs = 100
DataSet = CIFAR100
net_config =        {'numClasses':DataSet.num_classes}
loader_config =     {'amnt_dev':5000,'lab_BS':50,'dataseed':0,'num_workers':4}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':train_epochs}
trainer_config =    {}
all_hypers = {**net_config,**loader_config,**opt_config,**sched_config,**trainer_config}

trainer_config['log_dir'] = os.path.expanduser('~/tb-experiments/c100cutoutproper/')

def makeTrainer():
    trainset = DataSet('~/datasets/{}/'.format(DataSet.__name__))
    device = torch.device('cuda')
    fullCNN = nn.Sequential(
        trainset.default_aug_layers(),
        Cutout(),
        layer13(**net_config).to(device)
    )
    
    dataloaders = {}
    dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**loader_config)
    dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_sched = cosLr(**sched_config)
    return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**trainer_config,tracked_hypers=all_hypers)

trainer = makeTrainer()
trainer.train(train_epochs)
#torch.save(trainer,"c100trainer.t",pickle_module=dill)