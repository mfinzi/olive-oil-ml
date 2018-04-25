import torch
import torch.optim as optim
import torch.nn as nn

from oil.datasets import CIFAR10, C10augLayers
from oil.architectures.wide_resnet import WideResNet28x10
from oil.cnnTrainer import CnnTrainer
from oil.schedules import cosLr

num_classes = 10
datasets = CIFAR10(aug=False)


def makeCNN():
    CNN_cfg = WideResNet28x10
    CNN = CNN_cfg.base(*CNN_cfg.args, num_classes=num_classes, **CNN_cfg.kwargs)
    fullCNN = nn.Sequential(C10augLayers(),CNN)
    return fullCNN

total_epochs = 200
opt_constr = lambda params, base_lr: optim.SGD(params, base_lr, .9, weight_decay=1e-4, nesterov=True)
lr_lambda = cosLr(total_epochs, 1)


savedir = '/home/maf388/tb-experiments/wide_resnet/'
descr = "wide resnet with layer augs, devset size 5000"
config = {'base_lr':.1, 'amntLab':1, 'amntDev':5000,
          'lab_BS':50, 'ul_BS':50, 'num_workers':4,
          'lr_lambda':lr_lambda, 'opt_constr':opt_constr,
          'description':descr, 'log':False,
          }

# Get the samplers for test and train dataloaders from reference to use for each network
trainer = CnnTrainer(makeCNN(), datasets, None, **config)
lab_sampler = trainer.lab_train.batch_sampler
dev_sampler = trainer.dev.batch_sampler

savedirbase = '/home/maf388/tb-experiments/wide_resnets5000dev/'
for i in range(5):
    # Construct new CNN and trainer for each network
    trainer = CnnTrainer(makeCNN(), datasets, save_dir=None, **config)
    # Change the dataloaders to those used in the reference (to fix devset)
    trainer.lab_sampler = lab_sampler
    trainer.dev_sampler = dev_sampler
    trainer.train_iter = trainer.getTrainIter()
    # Train network i
    trainer.train(total_epochs)
    trainer.save_checkpoint(savedirbase+"net"+str(i)+".ckpt")