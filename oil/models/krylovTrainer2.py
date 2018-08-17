import torch
import torch.nn as nn
from torch.optim import Optimizer, SGD
from fvp_second_order import FVP_AG, flatten, unflatten_like
from fvp import FVP_FD
import gpytorch
from minres import minres
import copy
import time

def params_add_vec(params,vec):
    if isinstance(vec,torch.Tensor):
        veclist = unflatten_like(vec,params)
    else:
        veclist = vec
    for p, vz in zip(params,veclist):
        p.data += vz.data

    
        
class KrylovTrainer(object):
    r"""
    """

    def __init__(self, model, dataloader, optimizer, criterion=nn.CrossEntropyLoss(),
                 fisher_method='AG', krylov_size=10, num_steps=0, jitter=1e-3):
        self.model = model # Necessary to calculate fisher vector products with KL
        self.dataloader = dataloader # Necessary for full batch calculations
        self.optimizer = optimizer
        self.criterion = criterion
        self.hypers = {'krylov_size':krylov_size,'num_steps':num_steps,
                               'jitter':jitter,'fisher_method':fisher_method}
        
        FVP = FVP_AG if fisher_method=='AG' else FVP_FD
        self.F = FVP(*list(model.parameters()), model=model, train_iter=dataloader)
        #self.Q = None # To reduce the statefullness these have been removed
        #self.T = None
    
    def full_grad(self,model):
        # Accumulate (sum) grads from minibatches
        for i, (x,y) in enumerate(self.dataloader):
            batch_loss = self.criterion(model(x),y)
            batch_loss.backward()
        # Divide by the number of minibatches
        for p in model.parameters():
            if p.grad is None: continue
            p.grad.data /= i+1
        return flatten([p.grad for p in model.parameters()]).clone()
    
    def krylov_basis(self, F, g):
        """Construct an orthonormal Krylov basis of size krylov_size
            from the Krylov matrix [g,Fg,F^2g,...,F^{k-1}g]"""
        Q = g.new(g.size()[0],self.hypers['krylov_size']).zero_()
        
        for i in range(self.hypers['krylov_size']):
            Q = self._addvec_to_krylov(self.F, Q, g, i)
        return Q
        
    @staticmethod
    def _addvec_to_krylov(F, Q, g, k):
        """ adds the k'th vector F^{k-1}g to the orthonormal Krylov basis Q """
        #print(Q.shape)
        if k==0: Q[:,k] = g
        else:    Q[:,k] = F.matmul(Q[:,k-1])#F@Q[k-1]
        Q[:k+1], R = torch.qr(Q[:k+1])
        return Q
    
    @staticmethod
    def calc_T(Q, F):
        """ Calculates the tridiagonal matrix T from Q and F """
        T = Q.t()@F.matmul(Q)#(F@Q)
        return T
        
    def _condition_grads(self, Q, T, closure):
        """ Uses the Krylov projection matrix Q and the tridiagonal approximation of the
        Fisher matrix in the Krylov subspace T, to precondition the gradients. Optionally
        with num_steps steps of gradient descent to further improve the update."""
        
        # Get the conditioned gradients
        g = flatten([p.grad for p in self.model.parameters()])
        u = Q@T.inv_matmul(Q.t()@g)
        # Possibly apply multiple steps of sgd to optimize the update direction
        if self.hypers['num_steps']!=0:
            original_model_state = copy.deepcopy(self.model.state_dict())
            for i in self.hypers['num_steps']:# Potential difficulty here with learning rate schedule not updating defaults?
                params_add_vec(self.model.parameters(),-self.optimizer.defaults['lr']*u)
                loss = closure()
                g = flatten([p.grad for p in self.model.parameters()])
                u += Q@(Q.t()@g)
            self.model.load_state_dict(original_model_state)
            
        # Update the gradients with the new conditioned gradients
        u_list = unflatten_like(u,self.model.parameters())
        for p, conditioned_grad in zip(self.model.parameters(),u_list):
            if p.grad is None:
                continue
            p.grad = conditioned_grad
        return loss
            
    def train_epoch_fullF(self, callback=lambda *args,**kwargs:None):
        """ Trains for one epoch using the krylov basis calculated from the full
            batch Fisher evaluated at the beginning of the epoch (on the full grad g)"""
        F = self.F
        g = self.full_grad(self.model)
        Q = self.krylov_basis(F,g)
        T = self.calc_T(Q,F)
        
        for i, (x,y) in enumerate(self.dataloader):
            
            def batch_loss_backwards():
                self.optimizer.zero_grad()
                batch_loss = self.criterion(self.model(x),y)
                batch_loss.backward()
                return batch_loss
            
            bloss = batch_loss_backwards()
            self._condition_grads(Q,T,batch_loss_backwards)
            self.optimizer.step()
            callback(bloss)
        return g,Q,T
            
            
            
            
            
# class FisherKrylovTrainer(SGD):
#     r"""
#     """

#     def __init__(self, model, dataloader, loss=nn.CrossEntropyLoss(), fisher_method='AG', krylov_size=10, num_steps=0, jitter=1e-3, **kwargs):
#         self.model = model # Necessary to calculate fisher vector products with KL
#         self.dataloader = dataloader # Necessary for full batch calculations
#         #self.Q = None
#         #self.T = None
#         FVP = FVP_AG if fisher_method=='AG' else FVP_FD
#         self.F = FVP(*list(group['params']), model=model, train_iter=dataloader)
#         self.loss = loss
#         super().__init__(model.parameters(),**kwargs)
#         additional_defaults = {'krylov_size':krylov_size,'num_steps':num_steps,
#                                'jitter':jitter,'fisher_method':fisher_method}
#         self.defaults.update(additional_defaults)
        