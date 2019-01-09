import torch, torchvision
import torch.optim as optim
import torch.nn as nn

#import oil.augLayers as augLayers
from oil.vatCnnTrainer import VatCnnTrainer

from oil.datasets import CIFAR10, C10augLayers
from oil.networkparts import layer13
from oil.schedules import cosLr


def makeCNN():
    fullCNN = nn.Sequential(
        C10augLayers(),
        layer13(numClasses=10),
    )
    return fullCNN

total_epochs = 350
opt_constr = lambda params, base_lr: optim.SGD(params, base_lr, .9, weight_decay=1e-4, nesterov=True)
lr_lambda = cosLr(total_epochs, 1)

descr = "13Layer network trained with vat, layer augs, train=4000, val=0"
config = {'base_lr':.1, 'amntLab':4000,
          'lab_BS':50, 'ul_BS':50, 'num_workers':4,
          'lr_lambda':lr_lambda, 'opt_constr':opt_constr,
          'description':descr, 'log':True,
          }

datasets = CIFAR10(aug=False)

savedirbase = '/home/maf388/tb-experiments/vatruns/'
for regscale in [400, 100, 25, 5, 1]:
    for eps in [16, 8, 4, 1, .1]:
        # Construct new CNN and trainer for each network
        savedir = savedirbase +"net{}r{}e/".format(regscale,eps)
        config['regScale'] = regscale
        config['advEps'] = eps
        trainer = VatCnnTrainer(makeCNN(), datasets, save_dir=savedir, **config)
        trainer.train(total_epochs)
        trainer.save_checkpoint()