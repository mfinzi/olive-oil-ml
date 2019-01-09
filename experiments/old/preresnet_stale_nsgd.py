import torch, torchvision
import torch.optim as optim
import torch.nn as nn

#import oil.augLayers as augLayers
from oil.model_trainers.classifierTrainer import ClassifierTrainer
from oil.datasetup.datasets import CIFAR10, C10augLayers
from oil.datasetup.dataloaders import getUnlabLoader, getLabLoader
from oil.architectures.preresnet import PreResNet56
from oil.utils.utils import cosLr, loader_to
import sys
sys.path.append('/home/maf388/fisher_ngd/')
from oil.model_trainers.staleNsgd import StaleNsgdTrainer

train_epochs = 100
network = PreResNet56
net_config =        {'num_classes':10}
loader_config =     {'amnt_dev':5000,'lab_BS':32,'dataseed':0,'num_workers':4}
opt_config =        {'lr':30, 'momentum':.9, 'weight_decay':1e-5, 'nesterov':True}
sched_config =      {'cycle_length':train_epochs,'cycle_mult':1}
trainer_config =    {'inner_jitter':1e-6, 'outer_jitter':1e-6,'krylov_size':10,'fisher_method':'FD','epsilon':1e-4}

trainer_config['log_dir'] = '/home/maf388/tb-experiments2/preresnet56_stale_nsgd/glr30'
trainer_config['description'] = 'Stale nsgd Preresnet56 performance'

def makeTrainer():
    device = torch.device('cuda')
    CNN = network.base(*network.args, **network.kwargs, **net_config).to(device)
    fullCNN = nn.Sequential(C10augLayers(),CNN)
    trainset, testset = CIFAR10(False, '/home/maf388/datasets/cifar10/')

    dataloaders = {}
    dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**loader_config)
    dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_sched = cosLr(**sched_config)
    return StaleNsgdTrainer(fullCNN, dataloaders, opt_constr, lr_sched, **trainer_config)

trainer = makeTrainer()
trainer.train(train_epochs)
trainer.save_checkpoint()