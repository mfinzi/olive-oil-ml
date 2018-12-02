import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from contextlib import contextmanager
from itertools import islice, starmap
from ..utils.utils import cur

#from oil.extra.mvm import to_matmul, autoHvpBatch, autoFvpBatch, flatten
autoFvpBatch = NotImplemented

no_log = lambda *args,**kwargs:None

# Standard Stochastic Gradient Descent
def SGD(grads,w,lr,num_epochs,log=no_log):
    for epoch in range(num_epochs):
        for grad in grads:
            w = w - lr(epoch)*grad(w)
            log(w, lr(epoch))
    return w

# Stochastic Variance Reduced Gradient 
def SVRG(grads,w,w_a,lr,num_epochs,log=no_log):
    for epoch in range(num_epochs):
        w_a.data = w.data
        grad_a = grads(w_a) # Anchor grad
        for grad in grads:
            print("sas")
            grad_vr = grad(w)#grad(w) - grad(w_a) + grad_a
#             w.data = w.data - lr(epoch)*grad_vr.data
#             log(w, lr(epoch), grad_a)
    return w

def oja_grad(A,w):
    Aw = A(w)
    return -(Aw - (w@Aw)*w)

def oja_grad2(A,w):
    Aw = A(w)
    return -(Aw - (w@w)*w)

# Care must be taken in the dataloading step that
#    matrices A and B are independent
def SGHA_grad(A,B,w):
    Aw = A(w)
    return -(Aw - (w@Aw)*B(w))

# Care must be taken in the dataloading step that
#    matrices B1 and B2 are independent
def SGHA_grad2(A,B1,B2,w):
    return -(A(w) - (w@B1(w))*B2(w))


# W is an n x k matrix of k eigenvectors
def oja_subspace_grad(A,W):
    AW = A(W)
    return -(AW - W@(W.T@AW))
    pass



# Takes function mbatch -> Matrix, and a dataloader of minibatches,
#    the number of iterations k, and size of vector vec_size and
#    returns the dataloader of Matrices, with shape
class GradLoader(object):
    def __init__(self, grad_func, dataloader):
        self.dataloader = dataloader
        self.cur_grad_func = cur(grad_func) # Curried

    def __iter__(self):
        return starmap(self.cur_grad_func, self.dataloader)

    def __call__(self,w):
        for n,data in enumerate(self.dataloader,1):
            mdata = list(map(lambda d: d(w), data))
            try: means 
            except NameError: means = [0]*len(mdata)
            for i,data in enumerate(mdata):
                means[i] += (data - means[i])/n
        funcs = list(map(lambda c: lambda _: c, means))
        return self.cur_grad_func(*funcs)(w)

    # Full batch evaluation of the gradient
    # def __call__(self,w):
    #     grad_mean = 0
    #     for n, g in enumerate(self,1):
    #         grad_mean += (g(w) - grad_mean)/n
    #     return grad_mean
    ## Evaluates the mean of each element of the data loader
    ## at the particular point w, ie Full batch evaluation



def H_gen(train_loader, CNN):
    for mbatch in train_loader:
        Ht = lambda v: autoHvpBatch(v, CNN, mbatch)
        yield [Ht]

def F_gen(train_loader, CNN):
    for mbatch in train_loader:
        Ft = lambda v: autoFvpBatch(v, CNN, mbatch)
        yield [Ft]

# H, F loader where F_prev is used for F so that H is independent of F
def HF_gen(train_loader, CNN):
    Ft = lambda v: 0
    for mbatch in train_loader:
        Ht = lambda v: autoHvpBatch(v, CNN, mbatch)
        yield [Ht, Ft]
        Ft = lambda v: autoFvpBatch(v, CNN, mbatch)

# H, F loader where Ht and Ft are not independent
def HF_gen2(train_loader, CNN):
    for mbatch in train_loader:
        Ht = lambda v: autoHvpBatch(v, CNN, mbatch)
        Ft = lambda v: autoFvpBatch(v, CNN, mbatch)
        yield [Ht, Ft]

# H, F loader where Ft1 and Ft2 are independent
def HFF_gen(train_loader, CNN):
    Ft2 = lambda v: v
    for mbatch in train_loader:
        Ht = lambda v: autoHvpBatch(v, CNN, mbatch)
        Ft1 = lambda v: autoFvpBatch(v, CNN, mbatch)
        yield [Ht, Ft1, Ft2]
        Ft2 = Ft1
## Example usage
#    loader = multiGen(lambda: HF_gen(train_loader,CNN,reg=0))
#    grads = GradLoader(SGHA_grad, loader)
#    SGD(grads,w0,lr,num_epochs)