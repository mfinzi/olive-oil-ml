import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import os
#import oil.augLayers as augLayers
from oil.model_trainers.cGan import cGan
from oil.model_trainers.classifier import Classifier
from oil.datasetup.datasets import CIFAR10, C10augLayers
from oil.datasetup.dataloaders import getLabLoader
<<<<<<< HEAD
from oil.architectures.img_gen import conditionalgan
=======
#from oil.architectures.networkparts import layer13,ConvSmallNWN
from oil.architectures import conditionalgan
>>>>>>> c5d2fa6cde6a32286192b92717fdfd242448c3ed
from oil.utils.utils import cosLr, loader_to,imap,islice


train_epochs = 400
loader_config =     {'amnt_dev':0,'lab_BS':64,'dataseed':0,'num_workers':1}
trainer_config = {'n_disc':2}
trainer_config['log_dir'] = os.path.expanduser('~/tb-experiments/cgan_k128_3/')
sched_config =      {'cycle_length':train_epochs,'cycle_mult':1}
opt_config = {'lr':2e-4,'betas':(.5,.999)}


device = torch.device('cuda')
G = conditionalgan.Generator(num_classes=10).to(device)
D = conditionalgan.Discriminator(num_classes=10).to(device)
trainset, testset = CIFAR10(False, '~/datasets/cifar10/',gan_normalize=True)

dataloaders = {}
dataloaders['train'], _ = getLabLoader(trainset,**loader_config)
imgs_only  = imap(lambda z: z[0], dataloaders['train'])
dataloaders['dev'] = islice(imgs_only,5000//loader_config['lab_BS'])
dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

opt_constr = lambda params: optim.Adam(params,**opt_config)
lr_sched = cosLr(**sched_config)
trainer = cGan(G,dataloaders,opt_constr=opt_constr,lr_sched=lr_sched,D=D,**trainer_config)

for i in range(4):
    trainer.train(train_epochs//4)
    trainer.save_checkpoint()