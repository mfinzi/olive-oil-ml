
from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from functools import partial
from sklearn.metrics.pairwise import rbf_kernel

def sine_kernel(X1,X2,gamma):
    X1n = X1/np.linalg.norm(X1,axis=1)[:,None]
    X2n = X2/np.linalg.norm(X2,axis=1)[:,None]
    return np.exp(-gamma*(1-X1n@X2n.T))

def oh(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

class GraphSSL(BaseEstimator,ClassifierMixin,metaclass=ABCMeta):
    
    def __init__(self,gamma=2,reg=1,kernel='sin'):
        super().__init__()
        if kernel=='sin': self.kernel = partial(sine_kernel,gamma=gamma)
        elif kernel=='rbf': self.kernel = partial(rbf_kernel,gamma=gamma)
        elif callable(kernel): self.kernel = kernel
        else: raise NotImplementedError(f"Unknown kernel {kernel}")
        self.reg=reg

    def fit(self,X,y):
        """ Assumes y is -1 for unlabeled """
        n,d = X.shape
        c = max(y)+1
        Wxx = self.kernel(X,X)
        Wxx -= np.diag(np.diag(Wxx))
        D = np.diag(np.sum(Wxx,axis=-1))
        self.dm2 = dm2 = np.sum(Wxx,axis=-1)[:,None]**-.5
        L = np.eye(n) - dm2*Wxx*dm2.T
        Y = np.zeros((n,c))
        Y[y!=-1] = oh(y[y!=-1],c)
        self.X = X
        self.Ys = np.linalg.solve(L+self.reg*np.eye(n),Y)

    def predict(self,X_test):
        Wtx = self.kernel(X_test,self.X)
        dm2t = np.sum(Wtx,axis=1)**-.5
        Stx =  dm2t[:,None]*Wtx*self.dm2.T
        return (Stx@self.Ys).argmax(-1)
