import torch, torchvision,dill
import torch.optim as optim
import torch.nn as nn
import os
#import oil.augLayers as augLayers
from oil.model_trainers.classifier import Classifier
from oil.datasetup.datasets import CIFAR100
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.augLayers import RandomErasing
from oil.architectures.img_classifiers.networkparts import layer13
from oil.utils.utils import cosLr, loader_to,log_uniform

train_epochs = 100
DataSet = CIFAR100





config_gen = {
    'dataset': 'CIFAR100',
    'net_config': {'numClasses':dfg['dataset'].num_classes},
    'loader_config': {'amnt_dev':5000,'lab_BS':[50,1000]},
    'opt_config':{'lr':logUniform(.03,.3), 'momentum':.9, 'weight_decay':logUniform(1e-5,1e-3)},
    'sched_config':{'cycle_length':sampleFrom(lambda cfg: np.sqrt(cfg['bs']/50)*train_epochs)},
    'trainer_config':{}
}


trainer_config['log_dir'] = os.path.expanduser('~/tb-experiments/c100cutout/')

def makeTrainer(config):
    trainset = datasets[cfg['dataset']]('~/datasets/{}/'.format(cfg['dataset']))
    device = torch.device('cuda')
    fullCNN = nn.Sequential(
        trainset.default_aug_layers(),
        cfg['additional_aug'],
        cfg['network'](**config['net_config']).to(device)
    )
    dataloaders = {}
    dataloaders['train'], dataloaders['dev'] = getLabLoader(trainset,**cfg['loader_config'])
    dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

    opt_constr = lambda params: optim.SGD(params, **config['opt_config'])
    lr_sched = cosLr(**config['sched_config'])
    return Classifier(fullCNN,dataloaders,opt_constr,lr_sched,**config['trainer_config'],all_hypers=config)

trainer = makeTrainer()
trainer.train(train_epochs)
torch.save(trainer,pickle_module=dill)