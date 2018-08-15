import torch, torchvision
import torch.optim as optim
import torch.nn as nn

#import oil.augLayers as augLayers
from oil.cnnTrainer import CnnTrainer
from oil.datasets import CIFAR10, C10augLayers
from oil.networkparts import layer13,ConvSmallNWN
from oil.schedules import cosLr

train_epochs = 150
net_config =        {'numClasses':10}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':train_epochs,'cycle_mult':1}
trainer_config =    {'amntLab':1, 'amntDev':5000,'dataseed':0,
                    'lab_BS':50, 'num_workers':4, 'log':True, 
                    }
trainer_config['description'] = "13Layer network, {} dev".format(trainer_config['amntDev'])
trainer_config['save_dir'] = '/home/maf388/tb-experiments/layer13half/'

def makeTrainer():
    CNN = layer13(**net_config)
    fullCNN = nn.Sequential(C10augLayers(),CNN)
    datasets = CIFAR10(aug=False)
    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_lambda = cosLr(**sched_config)
    return CnnTrainer(fullCNN, datasets, opt_constr, lr_lambda, **trainer_config)

trainer = makeTrainer()
#trainer.train(2)
trainer.save_checkpoint()