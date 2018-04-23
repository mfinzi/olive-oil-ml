import torch, torchvision
import torch.optim as optim
import torch.nn as nn

#import oil.augLayers as augLayers
from oil.cnnTrainer import CnnTrainer
from oil.datasets import CIFAR10, C10augLayers
from oil.networkparts import layer13
from oil.schedules import cosLr

total_epochs = 250
savedir = '/home/maf388/tb-experiments/layer13dev/'
opt_constr = lambda params, base_lr: optim.SGD(params, base_lr, .9, weight_decay=1e-4, nesterov=True)
lr_lambda = cosLr(total_epochs, 1)

descr = "13Layer network, with layer augs, normalization .247,.243,.261, trainset = 45000"
config = {'base_lr':.1, 'amntLab':1, 'amntDev':5000,
          'lab_BS':50, 'ul_BS':50, 'num_workers':4,
          'lr_lambda':lr_lambda, 'opt_constr':opt_constr,
          'description':descr,
          }

CNN = layer13(numClasses=10)
fullCNN = nn.Sequential(
    C10augLayers(),
    CNN,
)

datasets = CIFAR10(aug=False)

trainer = CnnTrainer(fullCNN, datasets, savedir, **config)
trainer.train(total_epochs)
trainer.save_checkpoint()