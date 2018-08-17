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
from oil.utils import to_gpu, cur
from oil.extra.mvm import to_matmul, autoHvpBatch, autoFvpBatch, flatten

no_log = lambda *args,**kwargs:None
### Explicit update implementation
def ojas_top_eig(A_gen, lr, num_epochs, log=no_log):
    w = A_gen().get_random_unit()
    for epoch in range(num_epochs):
        for A in A_gen():
            w = w + lr(epoch)*A(w)
            w /= np.sqrt(w@w)
            log(A_gen,w,lr(epoch))
    return w

def ojas_top_eig2(A_gen, lr, num_epochs, log=no_log):
    w = A_gen().get_random_unit()
    for epoch in range(num_epochs):
        for A in A_gen():
            w = w + lr(epoch)*(A(w) - (w@w)*w)
            log(A_gen,w,lr(epoch))
    return w

def ojas_top_eig3(A_gen, lr, num_epochs, log=no_log):
    w = A_gen().get_random_unit()
    for epoch in range(num_epochs):
        for A in A_gen():
            Aw  = A(w)
            w = w + lr(epoch)*(Aw - (w@Aw)*w)
            log(A_gen,w,lr(epoch))
    return w

def chris_top_eig_general(AB_gen, lr, num_epochs, log=no_log):
    w = AB_gen.get_random_unit()
    for epoch in range(num_epochs):
        for A, (B1, B2) in AB_gen:
            ritz = w@B2(w)
            w = w + lr(epoch)*(A(w) -  ritz*B1(w))
            log(AB_gen,w,lr(epoch))
        # B1 and B2 need to be independent
    return w

def SGHA_top_eig(AB_gen, lr, num_epochs, log=no_log):
    w = AB_gen.get_random_unit()
    for epoch in range(num_epochs):
        for A , B in AB_gen:
            Aw = A(w)
            ritz = w@Aw
            w = w + lr(epoch)*(Aw - ritz*B(w))
            log(AB_gen,w,lr)
        # A and B need to be independent
    return w

def oja_grad(w, mbatch):
    Aw  = A(w)
    return - (Aw - (w@Aw)*w)



# Standard Stochastic Gradient Descent
def SGD(grads,w,lr,num_epochs,log=no_log):
    for epoch in range(num_epochs):
        for grad in grads:
            w = w - lr(epoch)*grad(w)
            log(w, lr(epoch))
    return w

# Stochastic Variance Reduced Gradient 
def SVRG(grads,w,lr,num_epochs,log=no_log):
    for epoch in range(num_epochs):
        w_a = deepcopy(w) #Anchor w
        grad_a = full_grad(w_a, grads) #Anchor grad
        for grad in grads:
            grad_vr = grad(w) - grad(w_a) + grad_a
            w = w - lr(epoch)*grad_vr
            log(w, lr(epoch))
    return w

# Gets the mean of grad(w) for grad in grads
def full_grad(w, grads):
    grad_sum = 0
    for n, grad in enumerate(grads,1):
        grad_sum += grad(w)
    return grad_sum/n

def partition(mbatch,k):
    """ Splits a minibatch into k parts """
    b = len(mbatch[0])
    partitions = [[data[b*i/k:b*(i+1)/k] for data in mbatch] for i in range(k)]
    return partitions

def H_gen(train_loader, CNN, reg = 0):
    H_mb = lambda mbatch: lambda v: autoHvpBatch(v, CNN, mbatch) + reg*v
    p = num_params(CNN)
    return MatrixLoader(H_mb, train_loader, lambda:random_param_vec(CNN))

## Yields a dataloader for independent samples of H, and F. Where
##     a single minibatch is split into two (for use with SGHA)
def HF_gen(train_loader, CNN, reg = 0):
    def HF_mb(mbatch):
        firstHalf,secondHalf = partition(mbatch,2)
        H_mb = lambda v: autoHvpBatch(v, CNN, firstHalf) + reg*v
        F_mb = lambda v: autoFvpBatch(v, CNN, secondHalf) + reg*v
        return (H_mb, F_mb)
    p = num_params(CNN)
    return MatrixLoader(HF_mb, train_loader, lambda:random_param_vec(CNN))

## Yields a dataloader for samples of H and two independent samples
#      of F. The entire batch is used for H, and is split for F1, F2
##     for use with chris's generalized eigen scheme
def HFF_gen(train_loader, CNN, reg = 0):
    def HFF_mb(mbatch):
        H_mb = lambda v: autoHvpBatch(v, CNN, mbatch) + reg*v
        firstHalf,secondHalf = partition(mbatch,2)
        F1_mb = lambda v: autoFvpBatch(v, CNN, firstHalf) + reg*v
        F2_mb = lambda v: autoFvpBatch(v, CNN, secondHalf) + reg*v
        return (H_mb, (F1_mb, F2_mb))
    return MatrixLoader(HFF_mb, train_loader, lambda:random_param_vec(CNN))

def num_params(model):
    return sum(p.numel() for p in model.parameters())

def random_param_vec(CNN):
    v = flatten([torch.ones_like(p).normal_() for p in CNN.parameters()])
    return v/v.norm()


class mvmMatrix(object):
    def __init__(self, mvm, shape):
        self.mvm = mvm
        self.shape = shape
    def __matmul__(self, v):
        return self.mvm(v)



# Takes function mbatch -> Matrix, and a dataloader of minibatches,
#    the number of iterations k, and size of vector vec_size and
#    returns the dataloader of Matrices, with shape
class MatrixLoader(object):
    def __init__(self, M_batch, dataloader, make_random_vec, k=1000):
        self.M_batch = M_batch
        self.dloader = dataloader
        self.k = k
        self.vec_constr = make_random_vec
        
    def get_random_unit(self):
        w = self.vec_constr()
        return w/np.sqrt(w@w)

    def __iter__(self):
        return map(self.M_batch, islice(self.dloader,self.k))
    
    # Full batch evaluation of the mvm
    def __call__(self,vec):
        return sum([M(vec)/len(self) for M in self])

    def __len__(self):
        return self.k



# #### Implicit loss implementation

# # Simple oja (unconstrained w)
# def oja_loss(trainer, *data):
#     A,  = data; w = trainer.w
#     with torch.no_grad(): Aw = A(w)
#     return -w@A(w)

# # Oja with constrained w
# def oja_loss2(trainer, *data):
#     A,  = data; w = trainer.w
#     with torch.no_grad(): Aw = A(w)
#     return -w@(Aw) + (1/4)*(w@w)**2

# def SGHA_loss(trainer, *data):
#     A, B = data; w = trainer.w
#     with torch.no_grad():
#         Aw = A(w)
#         ritz = w@Aw
#         Bw = B(w)
#     return -w@Aw + ritz*w@Bw
    
# def chris_general_loss(trainer, *data):
#     A, (B1,B2) = data; w = trainer.w
#     with torch.no_grad():
#         Aw = A(w)
#         B1w = B1(w)
#         B2w = B2(w)
#         ritz = w@B2w
#     return -w@Aw + ritz*w@B1w

# class EigenSgd(CnnTrainer):
#     def __init__(self, mat_gen, loss_func, save_dir=None, log=True,true_eigs=None):
#         # Setup tensorboard logger
#         self.save_dir = save_dir
#         self.writer = SummaryWriter(save_dir, log)
#         self.metricLog = {}
#         self.scheduleLog = {}
#         # Setup the other objects
#         self.w = Parameter(mat_gen.new_vec).cuda()
#         self.train_iter = mat_gen
#         self.loss_func = loss_func
#         self.numBatchesPerEpoch = len(mat_gen)
#         self.epoch = 0
#         self.true_eigs = true_eigs
#         # Setup Optimizers
#         if opt_constr is None: opt_constr = optim.SGD
#         self.optimizer = opt_constr([self.w])
#         try: self.lr_scheduler = lr_sched(self.optimizer)
#         except: self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,lr_sched)

#     def logStuff(self, i, epoch, numEpochs, mvms):
#         num_batches_per_epoch = len(self.mat_gen)
#         step = i+1 + (epoch+1)*num_batches_per_epoch
#         numSteps = numEpochs*num_batches_per_epoch
#         if self.true_eigs is not None:
#             (eig, u) = self.true_eigs
#             self.metricLog['Eig_Relative_Error'] = (self.ritz - eig)/eig
#             self.metricLog['Sin2_Error'] = 1 - ((u@self.w)**2)/(self.w@self.w)
#         else:
#             self.metricLog['Eig_Estimate_Batch'] = self.ritz
#         self.writer.add_scalars('metrics', self.metricLog, step)
#         self.scheduleLog['lr'] = self.lr_scheduler.get_lr()[0]
#         self.writer.add_scalars('schedules', self.scheduleLog, step)
#         prettyPrintLog(self.writer.emas(), epoch+1, numEpochs, step, numSteps)
    
#     def loss(self, *data):
#         return self.loss_func(self, *data)