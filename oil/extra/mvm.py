import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from contextlib import contextmanager
from itertools import islice

from oil.cnnTrainer import CnnTrainer
from oil.datasets import CIFAR10, C10augLayers
from oil.networkparts import layer13
from oil.schedules import cosLr
from oil.utils import to_gpu
from oil.extra.lanczos import lanczos_tridiag, lanczos_tridiag_to_diag
from oil.extra.linear_cg import linear_cg

# Converts a list of torch.tensors into a single flat torch.tensor
def flatten(tensorList):
    flatList = []
    for t in tensorList:
        flatList.append(t.view(-1))
    return torch.cat(flatList)

# Takes a flat torch.tensor and unflattens it to a list of torch.tensors
#    shaped like likeTensorList
def unflatten_like(vector, likeTensorList):
    outList = []
    i=0
    for tensor in likeTensorList:
        n = tensor.numel()
        outList.append(vector[i:i+n].view(tensor.shape))
        i+=n
    return outList

# Context manager to add a vector to the parameters and then subtract it
@contextmanager
def add(CNN,vec,mul=1):
    if type(vec)==torch.Tensor:
        veclist = unflatten_like(vec, CNN.parameters())
    else:
        veclist = vec
    for p,vz in zip(CNN.parameters(), veclist):
        p.data += mul*vz.data
    yield
    for p,vz in zip(CNN.parameters(), veclist):
        p.data -= mul*vz.data

# Computes the hessian vector product for a single minibatch using finite differences
#    Currently this is horribly inaccurate due to limitations of single precision arithmetic
def fdHvpBatch(vec, CNN,  trainData,  loss = F.cross_entropy, eps=1.,stencil=[-1.,1.]):
    vlist = unflatten_like(vec, CNN.parameters())
    weightedGrads = 0.
    for i, c in enumerate(stencil):
        with add(CNN, vlist, mul=eps*(i-(len(stencil)-1)/2)), torch.autograd.enable_grad():
            CNN.zero_grad()
            x,y = trainData
            batch_loss = loss(CNN(x),y)
            grad_list = torch.autograd.grad(batch_loss, CNN.parameters())
            weightedGrads += c*flatten(grad_list).data/eps
    return weightedGrads

# Computes the hessian vector product for a single minibatch using second order autograd
def autoHvpBatch(vec, CNN, trainData, loss = F.cross_entropy):
    CNN.zero_grad()
    with torch.autograd.enable_grad():
        x,y = trainData
        batch_loss = loss(CNN(x),y)
        grad_bl_list = torch.autograd.grad(batch_loss, CNN.parameters(), create_graph=True)
        grad_bl = flatten(grad_bl_list)
        deriv = torch.dot(grad_bl, vec)
        hvp_list = torch.autograd.grad(deriv, CNN.parameters())
    return flatten(hvp_list)
    
# Computes the hessian vector produce over the entire dataset    
def Hvm(vec,CNN, train_iter, auto=False, **kwargs):
    # Normalize vectore so effective eps for finite diff is best
    batchHvp = autoHvpBatch if auto else fdHvpBatch
    vnorm = vec.data.norm()
    if vnorm==0: return torch.zeros_like(vec)
    gradSum = 0
    for j, data in enumerate(train_iter):
        trainData = to_gpu(data)
        gradSum += batchHvp(vec/vnorm, CNN,  trainData, **kwargs)
    return vnorm*gradSum/(j+1)

# A loss whose 2nd derivative is the Fisher information matrix
def detached_entropy(logits,y):
    # -1*\frac{1}{m}\sum_{i,k} [f_k(x_i)] \log f_k(x_i), where [] is detach
    log_probs = F.log_softmax(logits,dim=1)
    probs = F.softmax(logits,dim=1)
    return -1*(probs.detach() * log_probs).sum(1).mean(0)

# A wrapper for fisher vector products (entire dataset)
def Fvm(*args, **kwargs):
    return  Hvm(*args, loss = detached_entropy, **kwargs)

# Converts a vector product routine into a matrix multiply routine
def to_matmul(mvm, *args, reg=1e-20, **kwargs):
    def matmul(matrix):
        out = torch.zeros_like(matrix)
        for i in range(matrix.shape[-1]):
            out[:,i] = mvm(matrix[:,i], *args, **kwargs) + reg*matrix[:,i]
        return out
    return matmul

# Method using linear conjugate gradients to get eigenvalues
#    Max iter specifies number of iterations to run CG
def get_eigs(mvm, max_iter, CNN, *args, **kwargs):
    v = flatten([torch.ones_like(p).normal_() for p in CNN.parameters()]).unsqueeze(1)
    with torch.autograd.no_grad():
        mul_closure = to_matmul(mvm, CNN, *args, **kwargs)
        _, t_mat = linear_cg(mul_closure, v, n_tridiag=1, max_iter = max_iter, max_tridiag_iter = max_iter)
        t_mat_np = t_mat[0].cpu().numpy()
        eigs = np.linalg.eigh(t_mat_np)[0]
    return eigs, t_mat_np

# Method using lanzos iteration to get eigenvalues
#    Max iter specifies number of iterations to run lanczos
def get_eigs_l(mvm, max_iter, CNN, *args, **kwargs):
    v = flatten([torch.ones_like(p).normal_() for p in CNN.parameters()])
    with torch.autograd.no_grad():
        mul_closure = to_matmul(mvm, CNN, *args, **kwargs)
        _, t_mat = lanczos_tridiag(mul_closure, max_iter = max_iter, tensor_cls = v.new, n_dims = len(v))
        t_mat_np = t_mat.cpu().numpy()
        eigs = np.linalg.eigh(t_mat_np)[0]
    return eigs, t_mat_np


