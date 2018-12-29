import torch
import torch.nn as nn
from torch.nn.functional import l1_loss as L1
from ..utils.utils import Eval
from .gan import Gan, hinge_loss_D, hinge_loss_G

class cGan(Gan):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fixed_input = (self.G.sample_y(32),self.G.sample_z(32))

    def discLoss(self, data):
        """ Hinge loss for discriminator"""
        x,y = data
        fake_logits = self.D(self.G(y),y)
        real_logits = self.D(x,y)
        return hinge_loss_D(real_logits,fake_logits)

    def genLoss(self,data):
        """ Hinge based generator loss -E[D(G(z))] """
        x,y = data
        fake_logits = self.D(self.G(y),y)
        return hinge_loss_G(fake_logits)


class Pix2Pix(cGan):
    def __init__(self,*args,l1=10,**kwargs):
        def initClosure(): self.hypers['l1'] = l1
        super().__init__(*args,extraInit=initClosure,**kwargs)

    def genLoss(self,data):
        y,x = data # Here y is the output image and x is input
        fake_y = self.G(y)
        adversarial = hinge_loss_G(self.D(fake_y,x))
        return adversarial + self.hypers['l1']*L1(fake_y,y)

    def logStuff(self, i, minibatch=None):
        raise NotImplementedError