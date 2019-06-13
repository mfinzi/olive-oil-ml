import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from torch.utils.data import DataLoader
#import oil.augLayers as augLayers
from oil.model_trainers.piModel import PiModel
from oil.datasetup.datasets import CIFAR10
from oil.datasetup.dataloaders import getLabLoader
from oil.architectures.img_classifiers.networkparts import layer13
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice, imap
from oil.tuning.study import Study, train_trial


def makeTrainer(config):
    cfg = {
        'dataset': CIFAR10,'network':layer13,'net_config': {},
        'loader_config': {'amnt_labeled':4000+5000,'amnt_dev':5000,'lab_BS':50, 'pin_memory':True,'num_workers':2},
        'opt_config': {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True},
        'num_epochs':200,'trainer_config':{'cons_weight':.3},
        'unlab_loader_config':{'batch_size':50,'num_workers':2},
        'trainer_config':{'cons_weight':.3,'log_dir':os.path.expanduser('~/tb-experiments/elu_semi_flow/pi/')},
        }
    recursively_update(cfg,config)
    trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']))
    device = torch.device('cuda')
    CNN = cfg['network'](num_classes=trainset.num_classes,**cfg['net_config']).to(device)
    fullCNN = nn.Sequential(trainset.default_aug_layers(),CNN)
    dataloaders = {}
    dataloaders['lab'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
    dataloaders['Train'] = islice(dataloaders['lab'],10000//cfg['loader_config']['lab_BS'])
    full_cifar_loader = DataLoader(trainset,shuffle=True,**cfg['unlab_loader_config'])
    dataloaders['_unlab'] = imap(lambda z: z[0], full_cifar_loader)
    if len(dataloaders['dev'])==0:
        testset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),train=False)
        dataloaders['test'] = DataLoader(testset,batch_size=cfg['loader_config']['lab_BS'],shuffle=False)
    dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}
    opt_constr = lambda params: torch.optim.SGD(params, **cfg['opt_config'])
    lr_sched = cosLr(cfg['num_epochs'])
    return PiModel(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])


PI_trial = train_trial(makeTrainer,strict=True)

cfg_spec = {
        'dataset': CIFAR10,'network':layer13,'net_config': {},
        'loader_config': {'amnt_labeled':4000+5000,'amnt_dev':5000,'lab_BS':50, 'pin_memory':True,'num_workers':2},
        'opt_config': {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True},
        'num_epochs':200,'trainer_config':{'cons_weight':.3},
        'unlab_loader_config':{'batch_size':50,'num_workers':2},
        'trainer_config':{'cons_weight':.3,'log_dir':os.path.expanduser('~/tb-experiments/elu_semi_flow/pi/')},
        }
ode_study = Study(PI_trial,cfg_spec,study_name='pi_baseline')
ode_study.run()
# trainer = makeTrainer()
# trainer.train(train_epochs)
# trainer.save_checkpoint()
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