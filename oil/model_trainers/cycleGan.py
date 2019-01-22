import torch
import torch.nn as nn
from ..utils.utils import Eval, join_opts, stateful_zip
from .gan import Gan, hinge_loss_D, hinge_loss_G, cGan
from torch.nn.functional import l1_loss as L1

class CycleGan(Gan):
    def __init__(self,*args,cycle_strength=10,**kwargs):
        super().__init__(*args,**kwargs)
        self.gan1 = cGan(*args,**kwargs)
        self.gan2 = cGan(*args,**kwargs)
        # The join needs to be stateful so that gopt and dopt loads/saves work
        # or we just add the state in the state dict explicitly
        self.g_optimizer = join_opts(self.gan1.g_optimizer,self.gan2.g_optimizer)
        self.d_optimizer = join_opts(self.gan1.d_optimizer,self.gan2.d_optimizer)
        self.dataloaders['train'] = stateful_zip(self.dataloaders['A'],self.dataloaders['B'])


    def discLoss(self, data):
        return self.gan1.disLoss(data) + self.gan2.disloss(data[::-1])

    def genLoss(self,data):
        """ Adversarial and cycle loss"""
        xa,xb = data
        adversarial_loss = self.gan1.genLoss(data[::-1]) + self.gan2.genLoss(data)
        G1,G2 = self.gan1.G, self.gan2.G 
        cycle_loss = L1(G2(G1(xa)),xa) + L1(G1(G2(xb)),xb)
        return adversarial_loss + self.hypers['cycle_strength']*cycle_loss

    def logStuff(self, i, minibatch=None):
        raise NotImplementedError

    def state_dict(self):
        extra_state = {
            'gan1_state':self.gan1.state_dict(),
            'gan2_state':self.gan2.state_dict(),
            'AB_loader_state':self.dataloaders['train'].state_dict(),
        }
        return {**super(cGan,self).state_dict(),**extra_state}

    def load_state_dict(self,state):
        super(cGan,self).load_state_dict(state)
        self.gan1.load_state_dict(state['gan1_state'])
        self.gan2.load_state_dict(state['gan2_state'])
        self.dataloaders['train'].load_state_dict()