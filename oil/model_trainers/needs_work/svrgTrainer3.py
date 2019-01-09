import torch
import torch.nn as nn
import copy
from torch.nn.parallel.replicate import replicate
from ..logging.lazyLogger import LazyLogger
from ..utils.utils import Eval
from ..utils.mytqdm import tqdm
from .classifierTrainer import ClassifierTrainer
from ..extra.SVRG import SVRG

def flatten(tensorList):
    flatList = []
    for t in tensorList:
        flatList.append(t.contiguous().view(t.numel()))
    return torch.cat(flatList)

class SVRGTrainer(ClassifierTrainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    

    def train(self, num_epochs=100):
        """ The main training loop"""
        def full_backward():
            self.model.zero_grad()
            #loss = 0
            for i, mb in enumerate(self.dataloaders['train']):
                #loss += (self.loss(*mb) - loss)/(i+1)
                (self.loss(*mb)/len(self.dataloaders['train'])).backward()
            #loss.backward()
        #self.optimizer.update_snapshot(full_backward)
        start_epoch = self.epoch
        for self.epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
            self.lr_scheduler.step(self.epoch)
            self.optimizer.update_snapshot(full_backward)
            #self.step(*(1,2))
            for self.i, minibatch in enumerate(self.dataloaders['train']):
                self.iteration = self.i+1 + (self.epoch+1)*len(self.dataloaders['train'])
                self.step(*minibatch)
                with self.logger as do_log:
                    if do_log: self.logStuff(self.i, minibatch)

    def step(self, *minibatch):
        """Replaces the gradient sample \tilde{g}(w) with
           $g_vr = \tilde{g}(w) + g(w_0) - \tilde{g}(w_0)$  """
        # Get minibatch model grads \tilde{g}(w)
        self.model.zero_grad()
        loss = self.loss(*minibatch)
        loss.backward()
        #self.model.zero_grad()
        # for i, mb in enumerate(self.dataloaders['train']):
        #     (self.loss(*mb)/len(self.dataloaders['train'])).backward()
        # def backward_closure():
        #     neg_loss = self.loss(*minibatch)
        #     neg_loss.backward()

        self.optimizer.step(lambda: self.loss(*minibatch).backward())
        return loss

