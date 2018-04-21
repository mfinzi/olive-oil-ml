import torch, torchvision
import torch.optim as optim

from oil.cnnTrainer import CnnTrainer
from oil.datasets import CIFAR10
from oil.networkparts import layer13
from oil.schedules import cosLr, sigmoidConsRamp

total_epochs = 1000
savedir = '/home/maf388/tb-experiments/layer13newnorm/'
opt_constr = lambda params, base_lr: optim.SGD(params, base_lr, .9, weight_decay=1e-4, nesterov=True)
lr_lambda = cosLr(total_epochs, 1)

config = {'base_lr':.1, 'amntLab':4000, 
          'lab_BS':50, 'ul_BS':50, 'num_workers':2,
          'lr_lambda':lr_lambda, 'opt_constr':opt_constr,
          }

CNN = layer13(numClasses=10)
datasets = CIFAR10(aug=True)
trainer = CnnTrainer(CNN, datasets, savedir, **config)
trainer.train(total_epochs)
trainer.constSWA(100)
trainer.save_checkpoint()