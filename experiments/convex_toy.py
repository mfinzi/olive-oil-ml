import torch, torchvision
import torch.optim as optim
import torch.nn as nn

#import oil.augLayers as augLayers
from oil.model_trainers.classifierTrainer import ClassifierTrainer
from oil.datasetup.datasets import CIFAR10, C10augLayers
from oil.datasetup.dataloaders import getUnlabLoader, getLabLoader
from oil.architectures.networkparts import layer13,ConvSmallNWN
from oil.utils.utils import cosLr, loader_to
from oil.extra.SVRG import SVRG

train_epochs = 100
problem_config =    {'numClasses':100,'numFeatures':1000,'dataset_size':10000}
loader_config =     {'bs':32}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':train_epochs,'cycle_mult':1}
trainer_config =    {'log_args':{'minPeriod':1/60}}

trainer_config['log_dir'] = '/home/maf388/tb-experiments2/convex_toy'
trainer_config['description'] = 'Convex Problem, SGD'

def makeTrainer():
    device = torch.device('gpu')
    N,D,C = problem_config['dataset_size'], problem_config['numFeatures'], problem_config['numClasses']
    model = nn.Linear(D,C).to(device)
    x = torch.randn(N, D)
    y_logits = x.matmul(D/10. * torch.rand(D, C)) + 0.3 * torch.randn(N, C)
    y = torch.distributions.Categorical(logits=y_logits).sample()

    dataset = torch.utils.data.TensorDataset(x, y).to(device)
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(dataset, batch_size=loader_config['bs'],shuffle=True)
    #dataloaders = {k: loader_to(device)(v) for k,v in dataloaders.items()}

    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_sched = cosLr(**sched_config)
    return ClassifierTrainer(model, dataloaders, opt_constr, lr_sched, **trainer_config)

if __name__ == "__main__":
    trainer = makeTrainer()
    trainer.train(train_epochs)
    trainer.save_checkpoint()