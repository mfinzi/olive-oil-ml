import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
from bgan.ganTrainer import GanTrainer
from bgan.utils import to_var_gpu

class WganTrainer(GanTrainer):

    def __init__(self, *args, gradPenalty=10, n_critic=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'gradPenalty':gradPenalty, 'n_critic':n_critic})

    def step(self, x_unlab, *_):
        x_unlab = to_var_gpu(x_unlab)
        for _ in range(self.hypers['n_critic']):
            self.d_optimizer.zero_grad()
            z = self.getNoise(self.hypers["ul_BS"]) #*.1
            x_fake = self.G(z).detach()
            wass_loss = self.D(x_fake).mean() - self.D(x_unlab).mean()
            d_loss = wass_loss + self.grad_penalty(x_unlab, x_fake)
            d_loss.backward()
            self.d_optimizer.step()
        
        self.g_optimizer.zero_grad()
        z = self.getNoise(self.hypers["ul_BS"]) #*.1
        x_fake = self.G(z)
        g_loss = -self.D(x_fake).mean()
        g_loss.backward()
        self.g_optimizer.step()
        return d_loss, g_loss

    def grad_penalty(self,x_real,x_fake):
        assert x_real.size()==x_fake.size(), "real and fake dims don't match"

        alpha = torch.rand(self.hypers['ul_BS'],1,1,1)
        alpha = Variable(alpha.expand_as(x_real).cuda(),requires_grad=False)
        interpolates = alpha * x_real + (1 - alpha) * x_fake
        interpolates = Variable(interpolates.data, requires_grad=True)
        disc_interpolates = self.D(interpolates)
        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.hypers['gradPenalty']
        return gradient_penalty

    