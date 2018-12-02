import numpy as np
import torch
import torch.nn as nn
import unittest
from oil.utils.utils import reusable
from oil.lazy.lazy_matrix import LazyMatrix, Lazy
from oil.lazy.lazy_types import LazyAvg,I
from oil.lazy.hessian import Hessian,Fisher

class LazyHessianTests(unittest.TestCase):

    def test_computes_deterministic_hessian_with_multiple_params(self):
        n = 100
        A = 1.*torch.rand(n,n)
        A = A + A.t()
        B = 2.*torch.rand(n,n)
        B = B + B.t()
        w1 = torch.rand(n)
        w2 = torch.rand(n)
        toy_net = QuadraticNetwork(w1,w2,A,B)
        def gen():
            for i in range(10):
                yield torch.ones(n),torch.zeros(n)
        dataloader = reusable(gen)
        eps = .1
        lazy_H = Hessian(toy_net,dataloader,loss = lambda x,y:x)+eps*I
        H = lazy_H.evaluate()
        self.assertTrue(torch.allclose(H[:n,:n],A+eps*torch.eye(n)))
        self.assertTrue(torch.allclose(H[n:,n:],B+eps*torch.eye(n)))
        self.assertTrue(torch.all(H[:n,n:]==0))


class QuadraticNetwork(nn.Module):
    
    def __init__(self, w1, w2,A1,A2):
        super().__init__()
        self.device = torch.device('cpu')
        self.w1 = torch.nn.Parameter(w1)
        self.w2 = torch.nn.Parameter(w2)
        self.A1 = A1
        self.A2 = A2
    def forward(self, x):
        return self.w1@self.A1@self.w1/2 + self.w2@self.A2@self.w2/2 + 5
        

if __name__=="__main__":
    unittest.main()