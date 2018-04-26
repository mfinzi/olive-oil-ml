import torch, torchvision
import torch.optim as optim
import torch.nn as nn

#import oil.augLayers as augLayers
from oil.cnnTrainer import CnnTrainer
from oil.datasets import CIFAR10, C10augLayers
from oil.networkparts import layer13
from oil.schedules import cosLr

def makeCNN():
    fullCNN = nn.Sequential(
        C10augLayers(),
        layer13(numClasses=10),
    )
    return fullCNN

total_epochs = 250
opt_constr = lambda params, base_lr: optim.SGD(params, base_lr, .9, weight_decay=1e-4, nesterov=True)
lr_lambda = cosLr(total_epochs, 1)

descr = "13Layer network, with layer augs, devset = 500"
config = {'base_lr':.1, 'amntLab':1, 'amntDev':500,
          'lab_BS':50, 'ul_BS':50, 'num_workers':4,
          'lr_lambda':lr_lambda, 'opt_constr':opt_constr,
          'description':descr, 'log':False,
          }

datasets = CIFAR10(aug=False)

# Get the samplers for test and train dataloaders from reference to use for each network
#load_path = '/home/maf388/tb-experiments/layer13dev/checkpoints/c.249.ckpt'
trainer = CnnTrainer(makeCNN(), datasets, None, **config)
lab_sampler = trainer.lab_train.batch_sampler
dev_sampler = trainer.dev.batch_sampler

savedirbase = '/home/maf388/tb-experiments/layer13s500dev/'
for i in range(30):
    # Construct new CNN and trainer for each network
    trainer = CnnTrainer(makeCNN(), datasets, save_dir=None, **config)
    # Change the dataloaders to those used in the reference (to fix devset)
    trainer.lab_train.batch_sampler = lab_sampler
    trainer.dev.batch_sampler = dev_sampler
    trainer.train_iter = trainer.getTrainIter()
    # Train network i
    trainer.train(total_epochs)
    trainer.save_checkpoint(savedirbase+"net"+str(i)+".ckpt")