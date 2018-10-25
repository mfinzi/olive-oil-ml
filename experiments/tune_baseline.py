import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import oil
from oil.tuning import tune
from ray.tune.async_hyperband import AsyncHyperBandScheduler


def makeTrainer(extra_config, net_config, opt_config, sched_config, trainer_config):
    CNN = oil.networkparts.layer13(**net_config)
    fullCNN = nn.Sequential(oil.datasets.C10augLayers(),CNN)
    datasets = oil.datasets.CIFAR10(aug=False)
    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_lambda = oil.schedules.cosLr(**sched_config)
    args = (fullCNN, datasets, opt_constr, lr_lambda)
    return oil.cnnTrainer.CnnTrainer(*args,**trainer_config)


extra_config =      {'numEpochs':5, 'expt_name':'baseline'}
net_config =        {'numClasses':10}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':extra_config['numEpochs'],'cycle_mult':1}
trainer_config =    {'amntLab':50000, 'amntDev':5000,'dataseed':0,
                    'lab_BS':50, 'ul_BS':50, 'num_workers':4, 'log':True, 
                    }
trainer_config['description'] = "13Layer network, {} dev".format(trainer_config['amntDev'])
trainer_config['save_dir'] = '/home/maf388/tb-experiments/'+extra_config['expt_name']+'/'
base_configs = [extra_config, net_config, opt_config, sched_config, trainer_config]


tunable_hypers =    {'weight_decay': lambda s: 10**random.uniform(-3,-5),
                    'lab_BS':lambda s: random.choice([32,50,64])}

tune(makeTrainer, base_configs, tunable_hypers,train_unit=50,numTrials=2)#,scheduler = hyperband)

# hyperband = AsyncHyperBandScheduler(
#         time_attr="timesteps_total",
#         reward_attr="mean_validation_accuracy",
#         grace_period=3,
#         max_t=extra_config['numEpochs'])





#redis_address="128.253.51.116:40212")




