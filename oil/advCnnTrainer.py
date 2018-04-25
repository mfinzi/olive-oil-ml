#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 23:53:56 2018

@author: wesleymaddox
implementation of fast sign adversarial training
not tested yet
"""

import torch
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from cnnTrainer import CnnTrainer

class AdvCnnTrainer(CnnTrainer):
    def __init__(self, *kwargs):
        CnnTrainer.__init__(self, *kwargs)
        self.lossfn = CrossEntropyLoss()
        self.epsilon = 0.001
        
    def loss(self, *data):
        x, y = data
        
        #may not be the best way of doing this as this is a copy
        #if x.requires_grad is False:
        inputs = Variable(x.data, requires_grad = True)
        
        #first pass through to pull out gradients of x
        outputs = self.CNN(inputs)
        loss = self.lossfn(outputs, y)
        loss.backward(retain_gradients = True)
        
        #now compute sign of gradients
        inputs_grad = torch.sign(inputs.grad)
        
        #perturb inputs and use clamped output
        inputs_perturbed = torch.clamp(inputs + self.epsilon * inputs_grad, 0.0, 1.0)
        outputs_adv = self.CNN(inputs_perturbed)
        
        loss += self.lossfn(outputs_adv, y)
        return loss
        
        
        