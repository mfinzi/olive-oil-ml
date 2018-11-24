from .lazy_matrix import LazyMatrix

class LazyDiag(LazyMatrix):
    def __init__(self, vec):
        assert len(vec.shape)==1, "Only flat vectors allowed"
        super().__init__(lambda v: vec*v, vec.shape*2,type(vec))

class LazySum(LazyMatrix):
    def __init__(self,*Ms):
        self._setBaseAttributes(Ms[0].baseAttributes())
        self.Ms = Ms
        for M in Ms: #Todo: add broadcasting support
            assert M.shape == Ms[0].shape, "Incompatible shapes" 
    def __matmul__(self,U): #TODO: Add parallelization support
        return sum(M@U for M in self.Ms)
    def __rmatmul__(self, V):
        return sum(V@M for M in self.Ms)
    def __str__(self):
        return "({}"+"+{}"*len(self.Ms[1:])+")".format(self.Ms)
    def __iter__(self):
        return self.Ms

class LazyMul(LazyMatrix):
    def __init__(self,M,c):
        self._setBaseAttributes(M.baseAttributes())
        self.M, self.c = M,c
        assert not isinstance(c,LazyMatrix), "No lazy elementwise mul"
    def __matmul__(self,U):
        return self.c*(self.M@U) #potential problem here with recursion?
    def __rmatmul__(self,V):
        return self.c*(V@self.M)
    def __str__(self):
        return "({}*{})".format(self.c,self.M)

class LazyMatmul(LazyMatrix):
    def __init__(self,M,N):
        self._setBaseAttributes(M.baseAttributes())
        self.shape = M.shape[:-1]+N.shape[1:]
        assert M.shape[1] == N.shape[0], "Incompatible shapes"
        self.M,self.N = M,N  
    def __matmul__(self,U):
        return self.M@(self.N@U)
    def __rmatmul__(self,V):
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
    


