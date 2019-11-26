import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import os
#import oil.augLayers as augLayers
from oil.model_trainers.gan import Gan
from oil.model_trainers.classifier import Classifier
from oil.datasetup.datasets import CIFAR10#, C10augLayers
from oil.datasetup.dataloaders import getLabLoader
#from oil.architectures.networkparts import layer13,ConvSmallNWN
from oil.architectures.img_gen import dcgan,resnetgan,dclikegan
from oil.utils.utils import cosLr, LoaderTo,imap,islice


train_epochs = 100
loader_config =     {'amnt_dev':0,'lab_BS':64,'dataseed':0,'num_workers':1}
trainer_config =    {'n_disc':5}
trainer_config['log_dir'] = os.path.expanduser('~/tb-experiments/baseline_gan_resnet_nd5_test/')


device = torch.device('cuda')
G = resnetgan.Generator().to(device)
D = resnetgan.Discriminator().to(device)
trainset = CIFAR10('~/datasets/cifar10/',gan_normalize=True)

dataloaders = {}
dataloaders['train'], _ = getLabLoader(trainset,**loader_config)
dataloaders['dev'] = islice(dataloaders['train'],5000//loader_config['lab_BS'])
dataloaders = {k: imap(lambda z:z[0].to(device),v) for k,v in dataloaders.items()}

trainer = Gan(G,dataloaders,D=D,**trainer_config)

trainer.train(train_epochs)
trainer.save_checkpoint()