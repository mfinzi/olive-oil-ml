import numpy as np
import torch
import unittest

from oil.lazy.lazy_matrix import LazyMatrix, Lazy
from oil.lazy.lazy_types import LazyAvg

class LazyMethodsTests(unittest.TestCase):
    def test_encodes_and_evaluates(self):
        M = np.random.rand(10,10)
        self.assertTrue(np.all(Lazy(M).evaluate()==M))
        N = torch.from_numpy(M)
        self.assertTrue(torch.all(Lazy(N).evaluate()==N))
        
    def pytorch_and_numpy_transpose(self):
        pass

    def test_adds_unpackable_sum(self):
        x = np.ones(10)
        A = np.ones((10,10))
        B = np.ones((10,10))
        lA = Lazy(A)
        lB = Lazy(B)
        self.assertTrue(np.all((A+A+A+B)@x==(lA+lA+lA+lB)@x))
        self.assertTrue(np.all((3*A+B)@x==(3*lA+lB)@x))

    def adds_very_lazy_sum(self):
        pass
    
    def iterates_through_very_lazy_sum(self):
        pass

    def unpackable_sum_passes_iterables(self):
        pass

    def device_is_set(self):
        pass

    def attributes_are_accessible(self):
        pass

    def tt(self):
        # np.allclose(A.evaluate(),A_full)
        # np.allclose(B.evaluate(),B_full)
        pass
if __name__=="__main__":
    unittest.main()