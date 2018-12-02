from .lazy_matrix import LazyMatrix
from .lazy_types import LazyAvg
#from ..utils.utils import reusable
import torch
import torch.nn.functional as F
from ..utils.utils import imap
from contextlib import contextmanager


def Hessian(model,dataloader, loss = F.cross_entropy):
    """LazyAvg Hessian for model evaluated on dataloader. 
        Assumes CE loss. Supports __iter__ method to get mb hessians"""
    mb_hessians = imap(lambda mb:mb_Hessian(model,mb,loss), dataloader)
    return LazyAvg(mb_hessians)
    
def Fisher(model,dataloader):
    """Same as for Hessian"""
    mb_fishers = imap(lambda mb:mb_Fisher(model,mb), dataloader)
    return LazyAvg(mb_fishers)

def mb_Hessian(model, minibatch, loss = F.cross_entropy):
    """ LazyMatrix Hessian of Cross Entropy Loss on minibatch"""
    mvm = lambda vec: autoHvpBatch(vec, model, minibatch,loss)
    params = flatten(model.parameters())
    baseAttributes = params.shape*2, params.dtype, params.device,type(params)
    return LazyMatrix(mvm,baseAttributes,rmvm=mvm)

def mb_Fisher(model, minibatch):
    """ LazyMatrix Fisher matrix (Avg) on minibatch, for classification"""
    mvm = lambda vec: autoFvpBatch(vec, model, minibatch)
    params = flatten(model.parameters())
    baseAttributes = params.shape*2, params.dtype, params.device,type(params)
    return LazyMatrix(mvm,baseAttributes,rmvm=mvm)

# Converts a list of torch.tensors into a single flat torch.tensor
def flatten(tensorList):
    flatList = []
    for t in tensorList:
        flatList.append(t.contiguous().view(t.numel()))
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


# Computes the hessian vector product for a single minibatch using second order autograd
def autoHvpBatch(vec, CNN, trainData, loss = F.cross_entropy):
    CNN.zero_grad()
    with torch.autograd.no_grad():
        vec_list = unflatten_like(vec, CNN.parameters())
    with torch.autograd.enable_grad():
        x,y = trainData
        batch_loss = loss(CNN(x.to(CNN.device)),y.to(CNN.device))
        grad_bl_list = torch.autograd.grad(batch_loss, CNN.parameters(), create_graph=True,only_inputs=True)
        deriv=0
        for vec_part, grad_part in zip(vec_list, grad_bl_list):
           deriv += torch.sum(vec_part.detach()*grad_part)
        #grad_bl = flatten(grad_bl_list)
        #deriv = torch.dot(grad_bl, vec)
        hvp_list = torch.autograd.grad(deriv, CNN.parameters(), only_inputs=True)
    return flatten(hvp_list)

# A loss whose 2nd derivative is the Fisher information matrix
def detached_entropy(logits,y):
    # -1*\frac{1}{m}\sum_{i,k} [f_k(x_i)] \log f_k(x_i), where [] is detach
    log_probs = F.log_softmax(logits,dim=1)
    probs = F.softmax(logits,dim=1)
    return -1*(probs.detach() * log_probs).sum(1).mean(0)

def autoFvpBatch(*args, **kwargs):
    return autoHvpBatch(*args, loss = detached_entropy, **kwargs)