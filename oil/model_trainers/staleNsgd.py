import torch
import torch.nn as nn
from fvp_second_order import FVP_AG, flatten, unflatten_like
from fvp import FVP_FD
import gpytorch
from minres import minres
#from ..logging.lazyLogger import LazyLogger
from ..utils.utils import Eval
from ..utils.mytqdm import tqdm
from .classifierTrainer import ClassifierTrainer

def params_add_vec(params,vec):
    if isinstance(vec,torch.Tensor):
        veclist = unflatten_like(vec,params)
    else:
        veclist = vec
    for p, vz in zip(params,veclist):
        p += vz

class StaleNsgdTrainer(ClassifierTrainer):
    r"""
    """
    def __init__(self, *args, fisher_method='AG', krylov_size=10,
                inner_jitter=1e-3, outer_jitter=1e-5,epsilon=1e-2, **kwargs):
        def initClosure():
            self.hypers = {'krylov_size':krylov_size,'fisher_method':fisher_method,
                            'inner_jitter':inner_jitter,'outer_jitter':outer_jitter,
                            'epsilon':epsilon}
            if fisher_method=='AG':
                self.FVP_constr = lambda data: FVP_AG(model=self.model,data=data)
            elif fisher_method=='FD':
                self.FVP_constr = lambda data: FVP_FD(model=self.model,data=data,epsilon=epsilon)
        super().__init__(*args, extraInit=initClosure, **kwargs)

    # @staticmethod
    # def orthog_krylov_matrix(F,g,k):
    #     #pre-define krylov basis
    #     krylov_basis = g.new(len(g), k).zero_()
    #     #first term is normalized rhs
    #     krylov_basis[:, 0] = (g / g.norm()).squeeze(-1)
        
    #     #pre-define coefficients
    #     krylov_coeffs = g.new(k, k).zero_()
    #     for i in range(1,k):
    #         new_vec = F(krylov_basis[:, i-1].unsqueeze(-1)).squeeze(-1)
    #         for j in range(i):
    #             krylov_coeffs[j, i] = krylov_basis[:, j].matmul(new_vec)
    #             new_vec = new_vec - krylov_coeffs[j, i] * krylov_basis[:, j]
    #         #print('Step', i, 'gs ', j, 'value is ', krylov_coeffs[i+1,i].item())
    #         krylov_basis[:, i] = new_vec / (new_vec.norm() + 1e-20)
    #     return krylov_basis

    # def solve_func(self,K, outer_jitter=1e-5):
    #     FK = self.full_F_matmul(K)
    #     #print(FK)
    #     #print(FK.shape)
    #     Q,R = torch.qr(FK)
    #     #print(Q.t().shape, R.shape)
    #     FK_inv_krylov = lambda v: torch.trtrs((Q.t()@v).unsqueeze(-1),R)[0].squeeze(-1)
    #     FK_inv = lambda u: K@FK_inv_krylov(u) + outer_jitter*u
    #     return FK_inv

    def _replace_grads(self, new_grads):
        """ Replaces the data in p.grad with new_grads"""
        # Update the gradients with the new conditioned gradients
        new_grad_list = unflatten_like(new_grads,self.model.parameters())
        for p, new_grad in zip(self.model.parameters(),new_grad_list):
            if p.grad is None:
                continue
            p.grad = new_grad.detach()

    def step(self,x,y, F_inv):
        self.optimizer.zero_grad()
        batch_loss = self.loss(x,y)
        batch_loss.backward()

        g = flatten([p.grad for p in self.model.parameters()]).clone()
        #print(g)
        nat_grads = F_inv(g)
        #if i%10==0: print(nat_grads)
        self._replace_grads(nat_grads)
        self.optimizer.step()

    def full_F_matmul(self,v):
        Fv = torch.zeros_like(v)
        num_minibatches = len(self.dataloaders['train'])
        for i,(x,y) in enumerate(self.dataloaders['train']):
            F_mb = self.FVP_constr(x)
            Fv += F_mb._matmul(v)/num_minibatches
        return Fv

    @staticmethod
    def get_Q_H(full_F,full_g,k):
        H = torch.zeros(k + 1, k, device=full_g.device, dtype=full_g.dtype)
        Q = torch.zeros(full_g.size(-1),k + 1, device=full_g.device, dtype=full_g.dtype)
    
        Q[:, 0] = full_g / full_g.norm()
        
        for j in range(k):
            new_vec = full_F(Q[:, j].unsqueeze(-1)).squeeze(-1)
            for i in range(j+1):
                H[i, j] = new_vec.dot(Q[:, i])
                new_vec = new_vec - H[i, j] * Q[:, i]
            H[j + 1, j] = new_vec.norm()
            Q[:, j + 1] = new_vec / H[j + 1, j]
        
        return Q, H

    @staticmethod
    def gels_solve_func(Q,H,inner_jitter=1e-6, outer_jitter=1e-6):
        k = H.size(-1)
        id = torch.eye(k+1,k,device=H.device,dtype = H.dtype)
        reg_H = torch.cat((H,inner_jitter*id))
        def F_inv(v):
            u = (Q.t()@v)
            reg_u = torch.cat((u,torch.zeros_like(u))).unsqueeze(-1)
            w = torch.gels(reg_u, reg_H)[0][:k].squeeze(-1)
            return Q[:,:-1]@w + outer_jitter*v
        return F_inv

    def get_stale_solver(self):
        self.model.zero_grad()
        for i, mb in enumerate(self.dataloaders['train']):
            (self.loss(*mb)/len(self.dataloaders['train'])).backward()
        full_g = flatten([p.grad for p in self.model.parameters()]).clone().detach()
        full_F = lambda v: self.full_F_matmul(v)
        #deterministic_K = self.orthog_krylov_matrix(full_F,full_g,self.hypers['krylov_size'])
        #F_inv = self.solve_func(deterministic_K,self.hypers['outer_jitter'])
        Q,H = self.get_Q_H(full_F,full_g,self.hypers['krylov_size'])
        F_inv = self.gels_solve_func(Q,H,self.hypers['inner_jitter'],self.hypers['outer_jitter'])
        return F_inv

    def train(self, num_epochs=100):
        """ The main training loop"""
        start_epoch = self.epoch
        for self.epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
            self.lr_scheduler.step(self.epoch)
            F_inv = self.get_stale_solver()
            self.logger.add_text("f_update","Updated F_inv")
            for self.i, minibatch in enumerate(self.dataloaders['train']):
                self.iteration = self.i+1 + (self.epoch+1)*len(self.dataloaders['train'])
                self.step(*minibatch, F_inv)
                with self.logger as do_log:
                    if do_log: self.logStuff(self.i, minibatch)