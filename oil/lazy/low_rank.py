import types
import numpy as np
import torch
from .lazy_matrix import LazyMatrix, AbstractAttribute

class LowRank(LazyMatrix):
    """"""
    def __init__(self,U,V=None):
        """Low rank matrix formed by UV.T, [U] = n x r, [V] = m x r"""
        self.U = U
        self.V = V if V is not None else U
        self.shape = (U.shape[0], self.V.shape[0])

    def _mvm(self, x):
        return self.U@(self.V.T@x)
    
    



    