import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import itertools
import os
import torch.nn.functional as F
from .trainer import Trainer
import scipy.misc
from ..utils.metrics import FID_and_IS
from ..utils.utils import Eval
from itertools import islice

def hinge_loss_G(fake_logits):
    return -torch.mean(fake_logits)

def hinge_loss_D(real_logits,fake_logits):
    return F.relu(1-real_logits).mean() + F.relu(1 + fake_logits).mean()

class Gan(Trainer):
    
    def __init__(self, *args,D=None,opt_constr=None,lr_sched=lambda e:1,
                        n_disc = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'n_disc':n_disc})

        self.G = self.model
        self.D = D
        if not opt_constr:
            g_opt=d_opt = lambda params: optim.Adam(params,2e-4,betas=(.5,.999))
            self.hypers.update({'lr':2e-4,'betas':(.5,.999)})
        elif isinstance(opt_constr,(list,tuple)):
            g_opt,d_opt = opt_constr
        else:
            g_opt=d_opt = opt_constr
        self.g_optimizer = g_opt(self.G.parameters())
        self.d_optimizer = d_opt(self.D.parameters())
        g_sched = optim.lr_scheduler.LambdaLR(self.g_optimizer,lr_sched)
        d_sched = optim.lr_scheduler.LambdaLR(self.d_optimizer,lr_sched)
        self.lr_schedulers = [g_sched,d_sched]
        self.fixed_input = (self.G.sample_z(32),)

    def step(self, data):
        # Step the Generator
        self.g_optimizer.zero_grad()
        G_loss = self.genLoss(data)
        G_loss.backward()
        self.g_optimizer.step()
        # Step the Discriminator
        for _ in range(self.hypers['n_disc']):
            self.d_optimizer.zero_grad()
            D_loss = self.discLoss(data)
            D_loss.backward()
            self.d_optimizer.step()

    def genLoss(self, x):
        """ hinge based generator loss -E[D(G(z))] """
        z = self.G.sample_z(x.shape[0])
        fake_logits = self.D(self.G(z))
        return hinge_loss_G(fake_logits)

    def discLoss(self, x):
        z = self.G.sample_z(x.shape[0])
        fake_logits = self.D(self.G(z))
        real_logits = self.D(x)
        return hinge_loss_D(real_logits,fake_logits)

    def logStuff(self, step, minibatch=None):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """

        metrics = {}
        if minibatch is not None:
            metrics['G_loss'] = self.genLoss(minibatch).cpu().data.numpy()
            metrics['D_loss'] = self.discLoss(minibatch).cpu().data.numpy()
        try: metrics['FID'],metrics['IS'] = FID_and_IS(self.as_dataloader(),self.dataloaders['test'])
        except KeyError: pass
        self.logger.add_scalars('metrics', metrics, step)
        # what if (in case of cycleGAN, there is no G?)
        fake_images = self.G(*self.fixed_input).cpu().data
        img_grid = vutils.make_grid(fake_images[:,:3], normalize=True)
        self.logger.add_image('fake_samples', img_grid, step)
        super().logStuff(step,minibatch)

    def as_dataloader(self,N=5000,bs=64):
        return GanLoader(self.G,N,bs)

    def state_dict(self):
        extra_state = {
            'G_state':self.G.state_dict(),
            'G_optim_state':self.g_optimizer.state_dict(),
            'D_state':self.D.state_dict(),
            'D_optim_state':self.d_optimizer.state_dict(),
        }
        return {**super().state_dict(),**extra_state}

    def load_state_dict(self,state):
        super().load_state_dict(state)
        self.G.load_state_dict(state['G_state'])
        self.g_optimizer.load_state_dict(state['G_optim_state'])
        self.D.load_state_dict(state['D_state'])
        self.d_optimizer.load_state_dict(state['D_optim_state'])

# TODO: ????
class GanLoader(object):
    """ Dataloader class for the generator"""
    def __init__(self,G,N=10**10,bs=64):
        self.G, self.N, self.bs = G,N,bs
    def __len__(self):
        return self.N
    def __iter__(self):
        with torch.no_grad(),Eval(self.G):
            for i in range(self.N//self.bs):
                yield self.G.sample(self.bs)
            if self.N%self.bs!=0:
                yield self.G.sample(self.N%self.bs)

    def write_imgs(self,path):
        np_images = np.concatenate([img.cpu().numpy() for img in self],axis=0)
        for i,img in enumerate(np_images):
            scipy.misc.imsave(path+'img{}.jpg'.format(i), img)