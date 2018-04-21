import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable, grad
#from torch.distributions import Dirichlet
from bgan.cnnTrainer import CnnTrainer
from bgan.einsum import einsum
from bgan.utils import wassersteinLoss2

class WcnnTrainer(CnnTrainer):
    def __init__(self, *args, gpscale = 10, dirparam = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'gpscale':gpscale, 'dirparam':dirparam})
        #print(self.hypers['batchSize'])
        #self.dirichletSampler = Dirichlet(torch.Tensor([0.5,0.5,0.5])) #torch.ones(self.hypers['batchSize'])

    def gradientPenalty(self, x):
        # xx = x.data#.type(torch.cuda.DoubleTensor)
        # alphas = self.hypers['dirparam']*np.ones(self.hypers['lab_BS'])
        # convexComb = np.random.dirichlet(alphas, self.hypers['lab_BS'])
        # convexComb = torch.from_numpy(convexComb).type(torch.FloatTensor).cuda()
        # interpolates = einsum('nm,mcij->ncij', convexComb, xx)
        interpolates = Variable(x.data, requires_grad=True)
        logits = self.CNN(interpolates)
        gradients = grad(outputs=logits, inputs=interpolates,
                            grad_outputs=torch.ones(logits.size()).cuda(),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2,dim=1)-1)**2).mean()
        return gradient_penalty
        

    def loss(self, x, y):
        # Losses for labeled samples are -log(P(y=yi|x))
        #loss = nn.CrossEntropyLoss()(self.CNN(x),y)
        loss = wassersteinLoss2(self.CNN(x),y.data)
        loss += self.hypers['gpscale']*self.gradientPenalty(x)
        
        return loss
