import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from contextlib import contextmanager

from oil.cnnTrainer import CnnTrainer
from oil.datasets import CIFAR10, C10augLayers
from oil.networkparts import layer13
from oil.schedules import cosLr
from oil.utils import to_gpu
from oil.extra.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from oil.extra.mvm import get_eigs,get_eigs_l, Hvm, Fvm, flatten, to_matmul



### Load the network

train_epochs = 150
net_config =        {'numClasses':10}
opt_config =        {'lr':.1, 'momentum':.9, 'weight_decay':1e-4, 'nesterov':True}
sched_config =      {'cycle_length':train_epochs,'cycle_mult':1}
trainer_config =    {'amntLab':1, 'amntDev':5000,'dataseed':0,
                    'lab_BS':50, 'num_workers':4, 'log':False, 
                    }
trainer_config['description'] = "13Layer network, {} dev".format(trainer_config['amntDev'])
savedir = None#'/home/maf388/tb-experiments/layer13dev/'

def makeTrainer():
    CNN = layer13(**net_config)
    fullCNN = nn.Sequential(C10augLayers(),CNN)
    datasets = CIFAR10(aug=False)
    opt_constr = lambda params: optim.SGD(params, **opt_config)
    lr_lambda = cosLr(**sched_config)
    return CnnTrainer(fullCNN, datasets, opt_constr, lr_lambda, **trainer_config)

trainer = makeTrainer()
trainer.load_checkpoint('/home/maf388/tb-experiments/layer13dev/checkpoints/c.150.ckpt')
trainer.CNN.eval()


# v = flatten([torch.ones_like(p).normal_() for p in trainer.CNN.parameters()])
# v = v/v.norm()
# print(v.norm())
# q = Hvm(v.zero_(),trainer.CNN,trainer.train_iter,num_batches=10, auto=False)
# print(q.norm())
# print(q.shape)
# M = to_matmul(Hvm, trainer.CNN,trainer.train_iter,num_batches=10)
# q = M(v.unsqueeze(1))
# print(q)
# print(q.shape)
# rhs = v.unsqueeze(1)
# initial_guess = rhs.new(rhs.size()).zero_()
# print(M(initial_guess).norm())

class Toy(nn.Module):
    def __init__(self, a1, a2):
        super().__init__()
        self.a1 = torch.nn.Parameter(a1)
        self.a2 = torch.nn.Parameter(a2)
    def forward(self, x):
        return (self.a1, self.a2)
        
class ToyTrainer(object):
    def __init__(self):
        self.A1 = torch.eye(5)
        self.A1[2,2]=3
        self.A1[4,4]=-1
        m = torch.normal(torch.zeros(5,5))
        _,rot = torch.symeig(m.t()+m,eigenvectors=True)
        self.A1 = torch.mm(rot.t(), self.A1)
        self.A1 = torch.mm(self.A1, rot)
        self.A2 = torch.eye(10)
        self.A2[3,3]= 100
        self.A2[4,4]= 1e-2
        self.CNN = Toy(torch.zeros(5),torch.zeros(10))
        self.numBatchesPerEpoch = 1
        def gen():
            while True:
                yield torch.zeros(1),torch.zeros(10)
        self.train_iter = gen()
        
    def cuda(self):
        self.A1 = self.A1.cuda()
        self.A2 = self.A2.cuda()
        self.CNN.cuda()
    def cpu(self):
        self.A1 = self.A1.cpu()
        self.A2 = self.A2.cpu()
        self.CNN.cpu()
    def loss(self, outputs, y):
        a1, a2 = outputs
        q = .5*torch.matmul(a1, torch.matmul(self.A1, a1)) + .5*torch.matmul(a2, torch.matmul(self.A2, a2))
        p = (a1*a1*a2[:5]).mean()+(a2[:5]*a1*a2[5:10]).mean()
        r = -5*(a1*a1*a2[:5]*a1).mean()
        t = -.1*(a1*a1*a2[:5]*a1*a2[5:10]).mean()
        y = .02*(a1*a1*a2[5:10]*a1*a2[5:10]*a1).mean()
        return q+2*r+p+t+y


#trainer = ToyTrainer()
it = trainer.dev
lanczos_iters = 30
mvm_config = {'num_batches':2, 'reg':1e-5,'eps':1e-1}# 'auto':True}#, 'loss':trainer.loss}
eigs, t_mat = get_eigs(Fvm, lanczos_iters, trainer.CNN, it, **mvm_config)
torch.save(eigs,"feigs.t"); torch.save(t_mat, "ftmat.t")
print('Finished one')

lanczos_iters = 30
mvm_config = {'num_batches':2, 'reg':1e-5,'eps':1e-1}# 'auto':True}#, 'loss':trainer.loss}
eigs, t_mat_l = get_eigs_l(Fvm, lanczos_iters, trainer.CNN, it, **mvm_config)
torch.save(eigs,"feigs_l.t"); torch.save(t_mat_l, "ftmat_l.t")