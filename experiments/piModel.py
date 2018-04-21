import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
import numpy as np
from oil.cnnTrainer import CnnTrainer
from oil.networkparts import layer13
import oil.augLayers as augLayers
from oil.piTrainer import PiTrainer
from oil.schedules import cosLr, sigmoidConsRamp
from oil.datasets import CIFAR10
import torch.optim as optim


datasets = CIFAR10(aug=False, ZCA=True)



epochs = int(350)#*(50000/4000)))

opt_constr = lambda params, base_lr: optim.SGD(params, base_lr, .9, weight_decay=1e-4, nesterov=True)
lr_lambda = cosLr(epochs, 1)

savedir = None #'/home/maf388/tb-experiments/mtparamsPIhalved/'
config = {'base_lr':.1, 'amntLab':4000, 
          'lab_BS':50, 'ul_BS':50, 'num_workers':2,
          'lr_lambda':lr_lambda, 'opt_constr':opt_constr,
          'cons_weight':100, 'rampup_epochs':5
          }

baseCNN = layer13(numClasses=10)
fullCNN = nn.Sequential(
    augLayers.RandomTranslate(4),
    augLayers.RandomHorizontalFlip(),
    augLayers.GaussianNoise(0.15),
    baseCNN,
)

trainer = PiTrainer(fullCNN,datasets,savedir,**config)
trainer.train(epochs)
trainer.save_checkpoint()