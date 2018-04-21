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
from bgan.schedules import cosLr







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


# def sigmoid_rampup(current, rampup_length):
#     """Exponential rampup from https://arxiv.org/abs/1610.02242"""
#     if rampup_length == 0:
#         return 1.0
#     else:
#         current = np.clip(current, 0.0, rampup_length)
#         phase = 1.0 - current / rampup_length
#         return float(np.exp(-5.0 * phase * phase))

total_epochs = 2120
#rampup_epochs = (4/15)*total_epochs
#cons_lambda = lambda epoch: 100 #*sigmoid_rampup(epoch, rampup_epochs)



lr_lambda = cosLr(50, 1.3)
savedirBase = '/home/maf388/tb-experiments/piModel/sgd/'
config = {'base_lr':5e-4, 'amntLab':4000, 
          'lab_BS':50, 'ul_BS':50, 'num_workers':2,
          'lr_lambda':lr_lambda,
          }

for i in [100, 30, 10 ,5, 1]:
    config['base_lr'] = i*1e-4
    savedir = savedirBase + str(i) + '/'

    baseCNN = layer13(numClasses=10)
    fullCNN = nn.Sequential(
        augLayers.RandomTranslate(4),
        augLayers.RandomHorizontalFlip(),
        augLayers.GaussianNoise(0.15),
        baseCNN,
    )       
    trainer = CnnTrainer(fullCNN,datasets,savedir,**config)
    trainer.train(total_epochs)