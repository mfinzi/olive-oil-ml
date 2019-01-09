import torch, torchvision
import torch.optim as optim
import torch.nn as nn

#import oil.augLayers as augLayers
from oil.model_trainers.classifierTrainer import ClassifierTrainer
from oil.model_trainers.svrgTrainer3 import SVRGTrainer
from oil.datasetup.datasets import CIFAR10, C10augLayers
from oil.datasetup.dataloaders import getUnlabLoader, getLabLoader
from oil.architectures.networkparts import layer13,ConvSmallNWN
from oil.utils.utils import cosLr, loader_to
from oil.extra.SVRG import SVRG

train_epochs = 100
net_config =        {'numClasses':10}
loader_config =     {'amnt_dev':5000,'lab_BS':50,'dataseed':0,'num_workers':4}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':train_epochs,'cycle_mult':1}
trainer_config =    {}

trainer_config['log_dir'] = '/home/maf388/tb-experiments2/svrg'
trainer_config['description'] = 'SVRG Test being made'

def makeTrainer():
    device = torch.device('cuda')
    CNN = layer13(**net_config).to(device)
    fullCNN = nn.Sequential(C10augLayers(),CNN)
    trainset, testset = CIFAR10(False, '/home/maf388/datasets/cifar10/')

    dataloaders = {}
    dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**loader_config)
    dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}


    opt_constr = lambda params: SVRG(params, **opt_config)
    lr_sched = cosLr(**sched_config)
    return SVRGTrainer(fullCNN, dataloaders, opt_constr, lr_sched, **trainer_config)

trainer = makeTrainer()
trainer.train(train_epochs)
trainer.save_checkpoint()