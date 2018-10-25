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


def flatten(tensorList):
    flatList = []
    for t in tensorList:
        flatList.append(t.view(-1))
    return torch.cat(flatList)

def unflatten_like(vector, likeTensorList):
    outList = []
    i=0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[i:i+n].view(tensor.shape))
        i+=n
    return outList

@contextmanager
def add(net,vec,mul=1):
    if type(vec)==torch.Tensor:
        veclist = unflatten_like(vec, net.parameters())
    else:
        veclist = vec
    for p,vz in zip(net.parameters(), veclist):
        p.data += mul*vz.data
    yield
    for p,vz in zip(net.parameters(), veclist):
        p.data -= mul*vz.data

def fdHvpBatch(trainer, vec, trainData,  eps=1,stencil=[-1,1]):
    net = trainer.CNN
    vlist = unflatten_like(vec, trainer.CNN.parameters())
    weightedGrads =0
    for i, c in enumerate(stencil):
        with add(net, vlist, mul=eps*(i-(len(stencil)-1)/2)), torch.autograd.enable_grad():
            net.zero_grad()
            batch_loss = trainer.loss(*trainData)
            grad_list = torch.autograd.grad(batch_loss, net.parameters())
            weightedGrads += c*flatten(grad_list).data/eps
    return weightedGrads

def autoHvpBatch(trainer, vec, trainData):
    net = trainer.CNN
    net.zero_grad()
    with torch.autograd.enable_grad():
        batch_loss = trainer.loss(*trainData)
        grad_bl_list = torch.autograd.grad(batch_loss, net.parameters(), create_graph=True)
        grad_bl = flatten(grad_bl_list)
        deriv = grad_bl@vec
        hvp_list = torch.autograd.grad(deriv, trainer.CNN.parameters())
    return flatten(hvp_list)
    
def Hvp(trainer,vec, max_batches=1000, auto=False, **kwargs):
    # Normalize vectore so effective eps for finite diff is best
    batchHvp = autoHvpBatch if auto else fdHvpBatch
    vnorm = vec.data.norm()
    gradSum = 0
    for j in range(min(trainer.numBatchesPerEpoch,max_batches)):
        trainData = to_gpu(next(trainer.train_iter))
        gradSum += batchHvp(trainer, vec/vnorm, trainData, **kwargs)
    return vnorm*gradSum/(j+1)

def matmul(trainer, matrix, **kwargs):
    out = torch.zeros_like(matrix)
    for i in range(matrix.shape[-1]):
        out[:,i] = Hvp(trainer,matrix[:,i],**kwargs)
    return out

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
    return ToyTrainer()

#Simple example sanity check
class Toy(nn.Module):
    def __init__(self, a1, a2):
        super().__init__()
        self.a1 = torch.nn.Parameter(a1)
        self.a2 = torch.nn.Parameter(a2)
    def forward(self, A1, A2):
        return .5*torch.matmul(self.a1, torch.matmul(A1, self.a1)) + .5*torch.matmul(self.a2, torch.matmul(A2, self.a2))
    
def gen():
    while True:
        yield torch.zeros(1)

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
        self.CNN = Toy(torch.zeros(5),torch.ones(10))
        self.numBatchesPerEpoch = 1
        self.train_iter = gen()
        self.A1 = self.A1.cuda()
        self.A2 = self.A2.cuda()
        self.CNN.cuda()
    def loss(self, *trainData):
        return self.CNN(self.A1, self.A2)
    def load_checkpoint(*args):
        pass

# def makeTrainer():
#     CNN = layer13(**net_config)
#     fullCNN = nn.Sequential(C10augLayers(),CNN)
#     datasets = CIFAR10(aug=False)
#     opt_constr = lambda params: optim.SGD(params, **opt_config)
#     lr_lambda = cosLr(**sched_config)
#     return CnnTrainer(fullCNN, datasets, opt_constr, lr_lambda, **trainer_config)

trainer = makeTrainer()
trainer.load_checkpoint('/home/maf388/tb-experiments/layer13dev/checkpoints/c.150.ckpt')
trainer.CNN.eval()


v = flatten([torch.randn(p.size()) for p in trainer.CNN.parameters()]).cuda()
v= v/v.norm()
hess_mul_closure = lambda mat: matmul(trainer, mat, max_batches=100, auto=True)#,eps=1e-3)
# r = hess_mul_closure(v)
# alpha_0 = v.mul(r).sum(-1)
# print(alpha_0)
with torch.autograd.no_grad():
    _, t_mat = lanczos_tridiag(hess_mul_closure, max_iter = 60, tensor_cls = v.new, n_dims = len(v))
torch.save(t_mat.cpu().numpy(), "toy_auto_t.t")
torch.save(t_mat.symeig()[0].cpu().numpy(), "toy_auto_eigs.t")


