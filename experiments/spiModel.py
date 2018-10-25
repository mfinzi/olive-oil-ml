import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
#import oil.augLayers as augLayers
from oil.cnnTrainer import CnnTrainer
from oil.spiTrainer import SPiTrainer
from oil.datasets import CIFAR10, C10augLayers
from oil.networkparts import layer13
from oil.schedules import cosLr, LRSchedulerWithInherit, ASGD

numEpochs = 200
net_config =        {'numClasses':10}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':numEpochs,'cycle_mult':1}
trainer_config =    {'amntLab':4000+5000, 'amntDev':5000,'dataseed':0,
                    'lab_BS':50, 'ul_BS':50, 'num_workers':4, 'log':True, 
                    'cons_weight':5, 'eps':1e-3,
                    }
trainer_config['description'] = "13Layer network, {} dev".format(trainer_config['amntDev'])
savedir_base = '/home/maf388/tb-experiments/spi/'

def makeArgs(savedir):
    CNN = layer13(**net_config)
    fullCNN = nn.Sequential(C10augLayers(),CNN)
    datasets = CIFAR10(aug=False)
    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_lambda = cosLr(**sched_config)
    return (fullCNN, datasets, opt_constr, lr_lambda, savedir)

cons_weights = np.array([1])#,300,100,30,3,1,.3,.1,.03])
acc_list = []
for i, cweight in enumerate(cons_weights):
    trainer_config['cons_weight'] = cweight
    savedir = savedir_base+'c{}/'.format(cweight)
    trainer = SPiTrainer(*makeArgs(savedir), **trainer_config)
    trainer.train(numEpochs)
    trainer.save_checkpoint()
    acc_list.append(trainer.getMetric())
    torch.save((cons_weights[:i+1],np.array(acc_list)),"tune_cweight.np")
