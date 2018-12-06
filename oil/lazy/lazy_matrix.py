from .translation import translate_methods

class LazyMatrix(object):
    
    """"""
    def __init__(self,mvm,baseAttributes,rmvm=NotImplementedError):
        self._setBaseAttributes(baseAttributes)
        self._mvm = mvm
        self._rmvm = rmvm

    def _setBaseAttributes(self,base_attributes):
        self.shape,self.dtype,self.device,self.cls = base_attributes
        self.xp = translate_methods[self.cls]
    def baseAttributes(self):
        return self.shape,self.dtype,self.device,self.cls

    def _mvm(self, v):
        raise NotImplementedError
    def _rmvm(self,v):
        raise NotImplementedError
    def _mmm(self,V):
        out = self.xp.new_zeros(V,self.shape[:-1]+V.shape[1:])
        for i in range(V.shape[-1]):
            out[:,i] = self@V[:,i]
        return out
    def _rmm(self,V):
        out = self.xp.new_zeros(V,V.shape[:-1]+self.shape[1:])
        for i in range(V.shape[0]):
            out[i,:] = V[i,:]@self
        return out
        
    def __matmul__(self,M):
        try: assert self.shape[-1]==M.shape[0], "Incompatible Matrix shapes"
        except AttributeError: pass #Some matrices are shapeless like identity
        if isinstance(M,LazyMatrix):
            return LazyMatmul(self,M)
        #if isinstance(M,self.cls):
        if len(M.shape)==1: # If no mvm is implemented, assume _mmm is overwritten
            try: return self._mvm(M) 
            except NotImplementedError: pass
        return self._mmm(M)
        #assert False, "Unknown matmullable object M with type: {}".format(type(M))

    # def __rmatmul__(self,M):
    #     return (self.T @ M.T).T
    def __rmatmul__(self,M):
        try: assert self.shape[0]==M.shape[-1], "Incompatible Matrix shapes"
        except AttributeError: pass #Some matrices are shapeless like identity
        if isinstance(M,LazyMatrix):
            return LazyMatmul(M,self)
        #if isinstance(M,self.cls):
        if len(M.shape)==1: # If no mvm is implemented, assume _mmm is overwritten
            try: return self._rmvm(M) 
            except NotImplementedError: pass
        return self._rmm(M)
        #assert False, "Unknown matmullable object M with type: {}".format(type(M))

    __array_ufunc__ = None # So that numpy arrays don't get confused looking for ufuncs

    @property
    def T(self):
        return LazyTranspose(self)

    def __add__(self,M):
        if M==0: return self
        else: return LazyAdd(self,M)
    def __radd__(self,M):
        return self+M
    def __sub__(self,M):
        return LazyAdd(self,-1*M)
    def __rsub__(self,M):
        return -1*self + M
    def __mul__(self,c):
        return LazyMul(self,c)
    def __rmul__(self,c):
        return self*c
    def __truediv__(self,c):
        return self*(1/c)
    def __rtruediv__(self,c):
        return self/c

    def diag(self):
        # use the stochastic diagonal estimator
        raise NotImplementedError

    def exact_diag(self): # O(n^2)
        assert self.shape[0]==self.shape[1], "Must be square"
        out = self.cls(self.shape[0])
        eye = self.xp.eye(self.shape[0],dtype=self.dtype,device=self.device)
        for i,ei in enumerate(eye):
            out[i] = ei@(self@ei)
        return out

    def approx_diag(self): # O(n/eps^2)
        raise NotImplementedError

    def evaluate(self):
        eye = self.xp.eye(self.shape[-1],dtype=self.dtype,device=self.device)
        return self@eye

    def __str__(self):
        alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz'
        return alphabet[hash(id(self))%48]
    

class Lazy(LazyMatrix):
    def __init__(self,array):
        self.array = array
        try: self.device
        except AttributeError: self.device = 'cpu'
        self.cls = type(array)
        self.xp = translate_methods[self.cls]
        self._mmm = self._mvm = array.__matmul__
        self._rmm = self._rmvm = array.__rmatmul__
    # Passes attributes through to underlying array    
    def __getattr__(self, name,default=None):
        return getattr(self.array,name,default)
    def __setattr__(self, name, value):
        if name=="array": super().__setattr__(name,value)
        elif hasattr(self.array,name):
            setattr(self.array,name,value)
        else: super().__setattr__(name,value)

# Needs to be at the bottom to prevent circular imports
from .lazy_types import LazyAdd, LazyMul, LazyMatmul, LazyTranspose


    