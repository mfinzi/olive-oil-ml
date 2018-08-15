import types
import numpy as np
import torch

class AbstractAttribute(object):
    def __get__(self, obj, type):
        raise NotImplementedError("This attribute was not set in a subclass")

class LazyMatrix(object):
    """"""
    #_mvm = AbstractAttribute()
    shape = AbstractAttribute()
    #out_class = AbstractAttribute()
    def __init__(self,mvm,shape):
        self._mvm = mvm
        self.shape = shape

    def _mvm(self, v):
        raise NotImplementedError
        
    def __matmul__(self,M):
        assert self.shape[-1]==M.shape[0], "Incompatible Matrix shapes"
        if len(M.shape)==1: # Vector input
            return self._mvm(M)
        new_shape = self.shape[:-1]+M.shape[1:]
        if isinstance(M,LazyMatrix):
            # Lazily compose the matmul in a lambda
            return LazyMatrix(lambda N: self@(M@N), new_shape)

        if isinstance(M,np.ndarray):
            out = np.ndarray(new_shape)
        if isinstance(M,torch.Tensor):
            out = M.new(*new_shape)
        for i in range(M.shape[-1]):
            out[i] = self@M[:,i]
        return out
    
    def __add__(self,M):
        assert isinstance(M,LazyMatrix), "Addition only supported for Lazy,Lazy"
        return LazyMatrix(lambda N: self@N + M@N, self.shape)

    def __sub__(self,M):
        raise NotImplementedError

    def __mul__(self,M):
        """Elementwise multiplication"""
        raise NotImplementedError

    @property
    def T(self):
        raise NotImplementedError
    def t(self):
        return self.T

    def __rmatmul__(self,M):
        if isinstance(M,LazyMatrix) or isinstance(M,np.ndarray):
            return (self.T @ M.T).T
        if isinstance(M,torch.Tensor):
            return (self.T @ M.t()).t()

    def diag(self):
        # use the stochastic diagonal estimator
        raise NotImplementedError

    def exact_diag(self): # O(n^2)
        assert self.shape[0]==self.shape[1], "Must be square"
        out = self.tensor_cls(self.shape[0])
        for i,ei in torch.eye(self.shape[0]):
            out[i] = ei@(self@ei)
        return out

    def approx_diag(self): # O(n/eps^2)
        raise NotImplementedError

    def evaluate(self):
        if type(self.shape) is torch.Size:
            return self@torch.eye(self.shape[-1])
        if type(self.shape) is tuple:
            return  self@np.eye(self.shape[-1])


    
    
    



    