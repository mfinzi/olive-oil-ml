import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import itertools
import torch.nn.functional as F
from .trainer import Trainer
from ..utils.metrics import FID, IS

def hinge_loss_G(fake_logits):
    return -torch.mean(fake_logits)

def hinge_loss_D(real_logits,fake_logits):
    return F.relu(1-real_logits).mean() + F.relu(1 + fake_logits).mean()

class Gan(Trainer):
    
    def __init__(self, *args, D, n_disc = 1, **kwargs):
        def initClosure():
            self.G = self.model
            self.D = D
            self.logger.add_text('ModelSpec','Discriminator: {}'.format(D))
            if opt_constr is None: 
                opt_constr = lambda params, lr: optim.Adam(params, 0.0002, betas=(.5,.999))
            self.d_optimizer = opt_constr(self.D.parameters())
            self.g_optimizer = opt_constr(self.G.parameters())
            self.hypers.update({'n_disc':n_disc})

            self.fixed_z = self.G.sample_z(32)
        super().__init__(*args, extraInit = initClosure, **kwargs)
        
    def step(self, *data):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        # Step the Generator
        G_loss = self.genLoss(*data)
        G_loss.backward()
        self.g_optimizer.step()
        # Step the Discriminator
        for _ in range(self.hypers['n_disc']):
            self.d_optimizer.zero_grad()
            self.g_optimizer.zero_grad()
            D_loss = self.discLoss(*data)
            D_loss.backward()
            self.d_optimizer.step()

    # def sampleZ(self, n_samples=1):
    #     return torch.randn(n_samples, self.G.z_dim).to(self.G.device)

    # def sampleImages(self, n_samples=1):
    #     return self.G(self.sampleZ(n_samples))

    def genLoss(self, *data):
        """ hinge based generator loss -E[D(G(z))] """
        x,  = data
        z = self.G.sample_z(len(x))
        fake_logits = self.D(self.G(z))
        return hinge_loss_G(fake_logits)

    def discLoss(self, *data):
        x,  = data
        z = self.G.sample_z(len(x))
        fake_logits = self.D(self.G(z))
        real_logits = self.D(x)
        return hinge_loss_D(real_logits,fake_logits)

    def logStuff(self, i, minibatch):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """
        step = i+1 + (self.epoch+1)*len(self.dataloaders['train'])

        metrics = {}
        metrics['G_loss'] = self.genLoss(*trainData).cpu().data.numpy()
        metrics['D_loss'] = self.discLoss(*trainData).cpu().data.numpy()
        metrics['FID'] = FID(self.as_dataloader(),self.dataloaders['dev'])
        metrics['IS'] = IS(self.as_dataloader())
        self.logger.add_scalars('metrics', metrics, step)

        fake_images = self.G(self.fixed_z).cpu().data
        img_grid = vutils.make_grid(fakeImages, normalize=True)
        self.logger.add_image('fake_samples', img_grid, step)
        super().logStuff(i,minibatch)
    
    def as_dataloader(self,N=4096,bs=32):
        return GanLoader(self.G,N,bs)

    def state_dict(self):
        extra_state = {
            'G_state':self.G.state_dict(),
            'G_optim_state':self.g_optimizer.state_dict(),
            'D_state':self.D.state_dict(),
            'D_optim_state':self.d_optimizer.state_dict(),
        }
        return super().state_dict.update(extra_state)

    def load_state_dict(self,state):
        super().load_state_dict(state)
        self.G.load_state_dict(state['G_state'])
        self.g_optimizer.load_state_dict(state['G_optim_state'])
        self.D.load_state_dict(state['D_state'])
        self.d_optimizer.load_state_dict(state['D_optim_state'])

# TODO: Deal with the the cGan case, where G(z,y)
# TODO: inherit from dataloader, allow for arbitrary samplers
class GanLoader(object):
    """ Dataloader class for the generator"""
    def __init__(self,G,N=10**10,bs=32):
        self.G, self.N, self.bs = G,N,bs
    def __len__(self):
        return self.N
    def __iter__(self):
        with Eval(self.G),torch.no_grad():
            for z in range(self.N):
                yield self.G.sample(self.bs)


# class GanLoader(object):
#     """ Dataloader class for the generator"""
#     def __init__(self,G,N=np.inf,bs=32):
#         self.G, self.N, self.bs = G,N,bs
#         if N < np.inf: self.Zs = G.sample_z(N*bs).reshape((N,bs,-1))
#         else: self.Zs = []
#     def __len__(self):
#         return self.N
#     def __iter__(self):
#         with Eval(self.G),torch.no_grad():
#             for z in self.Zs:
#                 yield self.G(z)
#             else: while True:
#                 yield self.G(self.G.sample_z(self.bs))