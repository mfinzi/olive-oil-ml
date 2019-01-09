import numpy as np
import torch
# Use .new_empty()
class xp2numpy():
    @staticmethod
    def new_randn(M,shape):
        return np.random.randn(*shape)
    @staticmethod
    def new_zeros(M,shape):
        return np.zeros(shape or M.shape,dtype=M.dtype)
    @staticmethod
    def eye(n,device=None,**kwargs):
        return np.eye(n,**kwargs)
    def __getattr__(self,name):
        return getattr(np,name)
class xp2torch():
    @staticmethod
    def new_randn(M,shape):
        return torch.randn(*shape,dtype=M.dtype,device =M.device)
    @staticmethod
    def new_zeros(M,shape):
        return M.new_zeros(*shape)
    def __getattr__(self,name):
        return getattr(torch,name)

translate_methods = {
    np.ndarray:xp2numpy(),
    torch.Tensor:xp2torch(),
}

@property
def T(self):
    return self if len(self.shape)==1 else self.t()
torch.Tensor.T = T
