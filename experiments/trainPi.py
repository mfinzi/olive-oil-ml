import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
#import oil.augLayers as augLayers
from oil.model_trainers.piModel import PiModel
from oil.datasetup.datasets import CIFAR10, C10augLayers
from oil.datasetup.dataloaders import getLabLoader
from oil.architectures.networkparts import layer13,ConvSmallNWN
from oil.utils.utils import cosLr, loader_to, imap



train_epochs = 200
net_config =        {'numClasses':10}
loader_config =     {'amnt_labeled':4000+5000,'amnt_dev':5000,'lab_BS':50,'dataseed':0,'num_workers':2}
unlab_loader_config = {'batch_size':50,'num_workers':2}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':train_epochs,'cycle_mult':1}
trainer_config =    {'cons_weight':.3}
all_hypers = {**net_config,**loader_config,**opt_config,**sched_config,**trainer_config}
trainer_config['log_dir'] = os.path.expanduser('~/tb-experiments/pi200/')
#trainer_config['description'] = '13Layer network Pi Model'

def makeTrainer():
    device = torch.device('cuda')
    CNN = layer13(**net_config).to(device)
    fullCNN = nn.Sequential(C10augLayers(),CNN)
    trainset, testset = CIFAR10(False, '~/datasets/cifar10/')

    dataloaders = {}
    dataloaders['lab'], dataloaders['dev'] = getLabLoader(trainset,**loader_config)
    full_cifar_loader = DataLoader(trainset,shuffle=True,**unlab_loader_config)
    dataloaders['unlab'] = imap(lambda z: z[0], full_cifar_loader)
    dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_sched = cosLr(**sched_config)
    return PiModel(fullCNN, dataloaders, opt_constr, lr_sched, **trainer_config,tracked_hypers=all_hypers)


trainer = makeTrainer()
trainer.train(train_epochs)
trainer.save_checkpoint()
# cons_weights = np.array([3.,1.,10.,.3,.1,30.,.03,100.])
# results_frame = pd.DataFrame()
# log_dir_base = os.path.expanduser('~/tb-experiments/pi/')
# for i, cweight in enumerate(cons_weights):
#     trainer_config['cons_weight'] = cweight
#     trainer_config['log_dir'] = log_dir_base+'c{}/'.format(cweight)
#     trainer = makeTrainer()
#     trainer.train(train_epochs)
#     result = trainer.logger.emas()
#     result['cons'] = cweight
#     results_frame = results_frame.append(result)
#     results_frame.to_pickle(log_dir_base+'results.pkl')