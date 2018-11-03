
from .lazy_matrix import LazyMatrix, lazy

class LazyDiag(LazyMatrix):
    def __init__(self, vec):
        assert len(vec.shape)==1, "Only flat vectors allowed"
        super().__init__(lambda v: vec*v, vec.shape*2,type(vec))

class LazyKron(LazyMatrix):
    def _mvm(self, v):
        raise NotImplementedError