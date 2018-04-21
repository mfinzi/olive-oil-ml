import torch, torchvision
import torch.optim as optim

from bgan.cnnTrainer import CnnTrainer
from bgan.datasets import CIFAR10
from bgan.networkparts import layer13
from bgan.schedules import cosLr, sigmoidConsRamp

total_epochs = 1000
savedir = '/home/maf388/tb-experiments/layer13ZCA/'
opt_constr = lambda params, base_lr: optim.SGD(params, base_lr, .9, weight_decay=1e-4, nesterov=True)
lr_lambda = cosLr(total_epochs, 1)

config = {'base_lr':.1, 'amntLab':4000, 
          'lab_BS':50, 'ul_BS':50, 'num_workers':0,
          'lr_lambda':lr_lambda, 'opt_constr':opt_constr,
          }

CNN = layer13(numClasses=10)
datasets = CIFAR10(aug=True, ZCA=True)
trainer = CnnTrainer(CNN, datasets, savedir, **config)
trainer.train(total_epochs)
trainer.save_checkpoint()