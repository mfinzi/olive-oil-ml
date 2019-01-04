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

def trace_est_stoch(A,num_epochs=2):
    """ A should be a stochastically iterable lazy matrix,
    performs Gaussian trace estimation using num_epochs epochs,
    returns both the estimate for the trace and the uncertainty"""
    iterates = []
    for k in range(num_epochs):
        for Ai in A:
            zi = A.xp.new_randn(A,[A.shape[-1]])
            iterates.append((zi.T@(Ai@zi)).cpu().numpy())
    iterates = np.array(iterates)
    return np.mean(iterates), np.std(iterates)/np.sqrt(len(iterates))

def trace_est(A,num_epochs=100):
    """ A should be a matrix,
    performs Gaussian trace estimation using num_epochs samples,
    returns both the estimate for the trace and the uncertainty"""
    iterates = []
    for k in range(num_epochs):
        zi = A.xp.new_randn(A,[A.shape[-1]])
        iterates.append((zi.T@(A@zi)).cpu().numpy())
    iterates = np.array(iterates)
    return np.mean(iterates), np.std(iterates)/np.sqrt(len(iterates))





    




