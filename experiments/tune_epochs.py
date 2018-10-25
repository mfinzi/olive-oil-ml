import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
#import oil.augLayers as augLayers
from oil.cnnTrainer import CnnTrainer
from oil.datasets import CIFAR10, C10augLayers
from oil.networkparts import layer13
from oil.schedules import cosLr, LRSchedulerWithInherit, ASGD
# def random_configs(configs, n):
#     for _ in range(n):
#         yield {k:np.random.choice(values) for (k,values) in configs.items()}

net_config =        {'numClasses':10}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':100,'cycle_mult':1}; interval=5
trainer_config =    {'amntLab':1, 'amntDev':5000,'dataseed':0,
                    'lab_BS':50, 'ul_BS':50, 'num_workers':4, 'log':True, 
                    }
trainer_config['description'] = "13Layer network, {} dev".format(trainer_config['amntDev'])
savedir_base = '/home/maf388/tb-experiments/layer13epochs/'

def makeArgs(savedir):
    CNN = layer13(**net_config)
    fullCNN = nn.Sequential(C10augLayers(),CNN)
    datasets = CIFAR10(aug=False)
    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_lambda = cosLr(**sched_config)
    return (fullCNN, datasets, opt_constr, lr_lambda, savedir)

epochs_list = np.arange(50,350,25)
acc_list = []
for i, epochs in enumerate(epochs_list):
    sched_config['cycle_length'] = epochs
    savedir = savedir_base+'e{}/'.format(epochs)
    trainer = CnnTrainer(*makeArgs(savedir), **trainer_config)
    trainer.train(epochs)
    trainer.save_checkpoint()
    acc_list.append(trainer.getMetric())
    torch.save((epochs_list[:i],np.array(acc_list)),"tune_epochs.np")



