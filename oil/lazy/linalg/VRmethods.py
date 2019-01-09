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
from ...utils.utils import cur
from ...utils.mytqdm import tqdm
from ..lazy_matrix import Lazy, LazyMatrix

no_log = lambda *args,**kwargs:None

# Standard Stochastic Gradient Descent
def SGD(grads,w,lr,num_epochs,log=no_log):
    for epoch in tqdm(range(num_epochs)):
        for grad in grads:
            w = w - lr(epoch)*grad(w)
            log(w, lr(epoch),grad(w))
    return w

# Stochastic Variance Reduced Gradient 
def SVRG(grads,w,lr,num_epochs,log=no_log):
    for epoch in tqdm(range(num_epochs)):
        w_a = deepcopy(w)   # Anchor w
        grad_a = grads(w_a) # Anchor grad
        for grad in grads:
            grad_vr = grad(w) - grad(w_a) + grad_a
            w = w - lr(epoch)*grad_vr
            log(w, lr(epoch), grad_vr)
    return w

def oja_grad(A,w):
    Aw = A@w
    return -(Aw - (w@Aw)*w)

def oja_grad2(A,w):
    Aw = A@w
    return -(Aw - (w@w)*w)

# Care must be taken in the dataloading step that
#    matrices A and B are independent
def SGHA_grad(A,B,w):
    Aw = A@w
    return -(Aw - (w@Aw)*(B@w))

# Care must be taken in the dataloading step that
#    matrices B1 and B2 are independent
def SGHA_grad2(A,B1,B2,w):
    return -(A@w - (w@(B1@w))*(B2@w))

def SGHA_subspace_grad(A,B,W):
    AW = A@W
    return -(AW - B@W@(W.T@AW))

def SGHA_subspace_grad2(A,B1,B2,W):
    return -(A@W - B2@W@(W.T@B1@W))

# W is an n x k matrix of k eigenvectors
def oja_subspace_grad(A,W):
    AW = A@W
    return -(AW - W@(W.T@AW))

class GradLoader(object):
    def __init__(self, grad_func, lazy_matrices):
        """Takes function M1,M2,...,Mn,w->grad, for lazyAvg matrices Mi
            and the sequence (M1,M2,...,Mn). Supports iterating through elems
            of the sum or evaluating the full average (gradient)"""
        self.cur_grad_func = cur(grad_func) # Curry the function
        self.lazy_matrices = lazy_matrices
    def __iter__(self):
        """Iterator of the gradients"""
        z = zip(*self.lazy_matrices)
        return starmap(self.cur_grad_func, z)
    def __call__(self,w):
        """Full gradient sum"""
        return self.cur_grad_func(*self.lazy_matrices)(w)

## Example usage
#    H = Hessian(model,dataloader)
#    F = Fisher(model,dataloader)
#    grads = GradLoader(SGHA_grad, (H,F))
#    SGD(grads,w0,lr,num_epochs)

## OR: ##
##   def C_gen():
##      indices = np.random.permutation(n).reshape(n//mb,mb)
##      for batch_ids in indices:
##          m = len(batch_ids)
##          x = X[batch_ids]
##          yield Lazy(x.T)@Lazy(x)/m
## 
##   C = LazyAvg(reusable(C_gen))
##   grads = GradLoader(oja_grad,(C))

