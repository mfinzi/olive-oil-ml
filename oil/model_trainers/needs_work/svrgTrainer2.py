import torch
import torch.nn as nn
import copy
from torch.nn.parallel.replicate import replicate
from ..logging.lazyLogger import LazyLogger
from ..utils.utils import Eval
from ..utils.mytqdm import tqdm
from .classifierTrainer import ClassifierTrainer
from .SVRG import SVRG

def flatten(tensorList):
    flatList = []
    for t in tensorList:
        flatList.append(t.contiguous().view(t.numel()))
    return torch.cat(flatList)

class SVRGTrainer(ClassifierTrainer):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    def __init__(self, *args, **kwargs):
        def initClosure():
            # Possible bug introduced for multiple gpus?
            self.snapshot_model = replicate(self.model,[0], detach=True)[0]
            self.snapshot_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            for param in self.snapshot_model.parameters():
                param.requires_grad = True
            self.optimizer = SVRG(self.model.parameters(),self.snapshot_model.parameters(),lr=.03,momentum=.9,nesterov=True,weight_decay=1e-4)
        #opt_constr = lambda params: SVRG(params,lr=.03,momentum=.9,nesterov=True,weight_decay=1e-4)
        super().__init__(*args, extraInit = initClosure,  **kwargs)

    

    def train(self, num_epochs=100):
        """ The main training loop"""
        def snapshot_closure(x,y):
            snapshot_loss = self.loss(x,y,model=self.snapshot_model)
            self.optimizer.zero_grad()
            snapshot_loss.backward()
            return snapshot_loss

        start_epoch = self.epoch
        for self.epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
            self.lr_scheduler.step(self.epoch)
            self.optimizer.update_snapshot(self.dataloaders['train'],snapshot_closure)
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
        loss = self.loss(*minibatch, model = self.model)
        loss.backward()
        #g_w = copy.deepcopy(torch.autograd.grad(loss,self.model.parameters()))
        #print("grads have norm {}".format(flatten([p.grad for p in self.model.parameters()]).norm()))
        # Get minibatch snapshot model grads \tilde{g}(w_0)
        self.snapshot_model.zero_grad()
        snapshot_loss = self.loss(*minibatch, model = self.snapshot_model)
        snapshot_loss.backward()
        #g_w0 = copy.deepcopy(torch.autograd.grad(snapshot_loss,self.snapshot_model.parameters()))
        #print("Snapshot grads have norm {}".format(flatten([p.grad for p in self.snapshot_model.parameters()]).norm()))

        debugstring = "Loss: {:.3f}, Snap Loss: {:.3f}".format(loss,snapshot_loss)
        self.logger.add_text('debug_loss', debugstring)

        self.optimizer.step()
        return loss

