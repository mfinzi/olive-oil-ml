import torch
import torch.optim as optim

from oil.datasets import CIFAR10
from oil.architectures.wide_resnet import WideResNet28x10
from oil.cnnTrainer import CnnTrainer
from oil.schedules import cosLr

num_classes = 10
datasets = CIFAR10(aug=True)

CNN_cfg = WideResNet28x10
CNN = CNN_cfg.base(*CNN_cfg.args, num_classes=num_classes, **CNN_cfg.kwargs)

total_epochs = 200
opt_constr = lambda params, base_lr: optim.SGD(params, base_lr, .9, weight_decay=1e-4, nesterov=True)
lr_lambda = cosLr(total_epochs, 1)


savedir = '/home/maf388/tb-experiments/wide_resnet/'
config = {'base_lr':.1, 'amntLab':1, 'amntDev':5000,
          'lab_BS':50, 'num_workers':4,
          'lr_lambda':lr_lambda, 'opt_constr':opt_constr,
          }

trainer = CnnTrainer(CNN, datasets, savedir, **config)
trainer.train(total_epochs)
trainer.save_checkpoint()