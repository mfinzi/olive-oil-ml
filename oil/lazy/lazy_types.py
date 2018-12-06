from .lazy_matrix import LazyMatrix

class LazySum(LazyMatrix):
    def __init__(self,Ms):
        """Inputs sequence of lazys, [M1,M2,...,Mn] -> M1+M2+...+Mn"""
        self.Ms = Ms
        for Mi in Ms:
            try: return self._setBaseAttributes(Mi.baseAttributes())
            except AttributeError: continue # Check shape compatibility?
        
    def _mmm(self, U): #TODO: Add parallelization support
        return sum(M@U for M in self.Ms)
    def _rmm(self, V):
        return sum(V@M for M in self.Ms)
    def __str__(self):
        try: return ("({}"+"+{}"*(len(self.Ms)-1)+")").format(*self.Ms)
        except TypeError: return "VeryLazySum"
    def __iter__(self):
        """Allows explicit iterating through elements in sum """
        if isinstance(self.Ms,tuple): # Collapse the sum if unpackable
            return (LazySum(ms) for ms in zip(*self.Ms))
        else: return iter(self.Ms)

class LazyAdd(LazySum):
    def __init__(self,*Ms):
        super().__init__(Ms)

class LazyMul(LazyMatrix):
    def __init__(self,M,c):
        try: self._setBaseAttributes(M.baseAttributes())
        except AttributeError: pass #shapeless matrices
        self.M, self.c = M,c
        assert not isinstance(c,LazyMatrix), "No lazy elementwise mul"
    def _mmm(self,U):
        return self.c*(self.M@U) #potential problem here with recursion?
    def _rmm(self,V):
        return self.c*(V@self.M)
    def __str__(self):
        return "({}*{})".format(self.c,self.M)

class LazyMatmul(LazyMatrix):
    def __init__(self,M,N):
        self._setBaseAttributes(M.baseAttributes())
        self.shape = M.shape[:-1]+N.shape[1:]
        assert M.shape[1] == N.shape[0], "Incompatible shapes"
        self.M,self.N = M,N  
    def _mmm(self,U):
        return self.M@(self.N@U)
    def _rmm(self,V):
        return (V@self.M)@self.N
    def __str__(self):
        return "({}@{})".format(self.M,self.N)

class LazyTranspose(LazyMatrix):
    def __init__(self,A):
        self._setBaseAttributes(A.baseAttributes())
        self.shape = A.shape[::-1]
        self.__matmul__ = A.__rmatmul__
        self.__rmatmul__ = A.__matmul__
        self.A = A
    def __str__(self):
        return "{}.T".format(self.A)
    @property
    def T(self):
        return self.A
    
# class Symmetric(LazyMatrix):
#     @property
#     def T(self):
#         return self

class LazyDiag(LazyMatrix):
    def __init__(self, vec):
        assert len(vec.shape)==1, "Only flat vectors allowed"
        self.vec = vec
        try: vdevice = vec.device
        except AttributeError: vdevice = 'cpu'
        baseAttributes = vec.shape*2,vec.dtype,vdevice,type(vec)
        mvm = rmvm = lambda v: self.vec*v
        super().__init__(mvm,baseAttributes,rmvm)
    def diag(self):
        return self.vec
    def __str__(self):
        return "Diag({})".format(self.vec)

class LazyAvg(LazySum):
    """ Convenience version of LazySum that averages and allows iterating
        through the elements in the average """
    def _mmm(self, U):
        return super()._mmm(U)/self._get_n()
    def _rmm(self,V):
        return super()._rmm(V)/self._get_n()
    def _get_n(self):
        try: return self._n # Check if value has been cached
        except AttributeError:
            try: self._n = len(self.Ms)
            except TypeError: self._n = sum(1 for _ in self.Ms)
        return self._n
        
    def __str__(self):
        return super().__str__()+"/n"

# Singleton
class LazyIdentity(LazyMatrix):
    def __init__(self):
        pass
    def _mmm(self,U):
        return U
    def _rmm(self,V):
        return V
    def __str__(self):
        return "I"
I = LazyIdentity()