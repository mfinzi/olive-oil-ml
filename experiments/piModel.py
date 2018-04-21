import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn
import numpy as np
from bgan.cnnTrainer import CnnTrainer
from bgan.networkparts import layer13
import bgan.augLayers as augLayers
from bgan.piTrainer import PiTrainer
from bgan.schedules import cosLr, sigmoidConsRamp
import torch.optim as optim


img_size = 32
transform_dev = transforms.Compose(
    [transforms.Resize(img_size),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_train = transform_dev
# transform_train = transforms.Compose(
#     [transforms.RandomCrop(32, padding=4),
#      transforms.RandomHorizontalFlip(),
#      transform_dev])

pathToDataset = '/scratch/datasets/cifar10/'
trainset = CIFAR10(pathToDataset, download=True, transform=transform_train)
devset = CIFAR10(pathToDataset, train=False, download=True, transform=transform_dev)
testset = None
datasets = (trainset, devset, testset)



epochs = int(350)#*(50000/4000)))

opt_constr = lambda params, base_lr: optim.SGD(params, base_lr, .9, weight_decay=1e-4, nesterov=True)
lr_lambda = cosLr(epochs, 1)

savedir = '/home/maf388/tb-experiments/mtparamsPIhalved/'
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