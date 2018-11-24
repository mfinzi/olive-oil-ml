from .translation import translate_methods
import numpy as np

class LazyMatrix(object):
    
    """"""
    def __init__(self,mvm,baseAttributes,rmvm=NotImplementedError):
        self._setbaseAttributes(baseAttributes)
        self._mvm = mvm
        self._rmvm = rmvm

    def _setBaseAttributes(self,base_attributes):
        self.shape,self.dtype,self.device,self.cls = base_attributes
        self.np = translate_methods[self.cls]
    def baseAttributes(self):
        return self.shape,self.dtype,self.device,self.cls

    def _mvm(self, v):
        raise NotImplementedError
    def _rmvm(self,v):
        raise NotImplementedError
    def _mm(self,V):
        out = self.np.new_zeros(V,self.shape[:-1]+V.shape[1:])
        for i in range(V.shape[-1]):
            out[:,i] = self@V[:,i]

    def __matmul__(self,M):
        assert self.shape[-1]==M.shape[0], "Incompatible Matrix shapes"
        if len(M.shape)==1: # Vector input
            return self._mvm(M)
        new_shape = self.shape[:-1]+M.shape[1:]
        if isinstance(M,LazyMatrix):
            return LazyMatmul(self,M)
        if isinstance(M,self.cls):
            return self._mm(M)
    def __rmatmul__(self,M):
        raise NotImplementedError

    @property
    def T(self):
        return LazyTranspose(self)

    # def __rmatmul__(self,M):
    #     return (self.T @ M.T).T

    def __add__(self,M):
        return LazySum(self,M)
    def __radd__(self,M):
        return self+M
    def __mul__(self,c):
        return LazyMul(self,c)
    def __rmul__(self,c):
        return self*c
    def __sub__(self,M):
        return LazySum(self,-1*M)
    def __rsub__(self,M):
        return -1*self + M

    def diag(self):
        # use the stochastic diagonal estimator
        raise NotImplementedError

    def exact_diag(self): # O(n^2)
        assert self.shape[0]==self.shape[1], "Must be square"
        out = self.cls(self.shape[0])
        for i,ei in self.np.eye(self.shape[0]):
            out[i] = ei@(self@ei)
        return out

    def approx_diag(self): # O(n/eps^2)
        raise NotImplementedError

    def evaluate(self):
        return self@self.np.eye(self.shape[-1])
    

class Lazy(LazyMatrix):
    def __init__(self,array):
        self.array = array
        try: self.device
        except AttributeError: self.device = 'cpu'
        self.cls = type(array)
        self.np = translate_methods[self.cls]
        self._mm = self._mvm = array.__matmul__
        self._rmm = self._rmvm = array.T.__matmul__
    # Passes attributes through to underlying array    
    def __getattr__(self, name,default=None):
        return getattr(self.array,name,default)
    def __setattr__(self, name, value):
        if name=="array": super().__setattr__(name,value)
        elif hasattr(self.array,name):
            setattr(self.array,name,value)
        else: super().__setattr__(name,value)

# Needs to be at the bottom to prevent circular imports
from .lazy_types import LazySum, LazyMul, LazyMatmul, LazyTranspose


    