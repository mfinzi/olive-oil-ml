import torch
import torch.nn as nn
import copy
from torch.nn.parallel.replicate import replicate
from ..logging.lazyLogger import LazyLogger
from ..utils.utils import Eval
from ..utils.mytqdm import tqdm
from .classifierTrainer import ClassifierTrainer


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
            self.snapshot_gradient = None
        super().__init__(*args, extraInit = initClosure, **kwargs)


    def train(self, num_epochs=100):
        """ The main training loop"""
        start_epoch = self.epoch
        for self.epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
            self.lr_scheduler.step(self.epoch)
            self.update_snapshot()
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
        g_w = copy.deepcopy(torch.autograd.grad(loss,self.model.parameters()))
        #print("grads have norm {}".format(flatten([p.grad for p in self.model.parameters()]).norm()))
        # Get minibatch snapshot model grads \tilde{g}(w_0)
        self.snapshot_model.zero_grad()
        snapshot_loss = self.loss(*minibatch, model = self.snapshot_model)
        g_w0 = copy.deepcopy(torch.autograd.grad(snapshot_loss,self.snapshot_model.parameters()))
        #print("Snapshot grads have norm {}".format(flatten([p.grad for p in self.snapshot_model.parameters()]).norm()))

        flat_gw = flatten(g_w)
        flat_gw0 = flatten(g_w0)
        debugstring = "Loss: {:.3f}, Snap Loss: {:.3f}".format(loss,snapshot_loss)
        self.logger.add_text('debug_loss', debugstring)
        debugstring = "|g_w|= {:.3f}, |g_w0|= {:.3f}".format(flat_gw.norm(),flat_gw0.norm())
        self.logger.add_text('debug_norms', debugstring)
        debugstring = "|g_w - g_w0|/|g_w|= {:.3f}".format((flat_gw - flat_gw0).norm()/flat_gw.norm())
        self.logger.add_text('debug_grads', debugstring)

        # Update the gradients in place using above & full snapshot grads
        for grad, snapshot_grad, sn_full_g, p in zip(g_w, g_w0, self.snapshot_gradient, self.model.parameters()):
            if grad is None: continue
            p.grad.data = grad.data + sn_full_g.data - snapshot_grad.data
        
        debugstring = "|g_vr - g_w|/|g_w|= {:.3f}".format((flatten(g_w) - flatten(\
            [p.grad.data for p in self.model.parameters()])).norm()/flatten(g_w).norm())
        self.logger.add_text('debug_updates', debugstring)

        self.optimizer.step()
        return loss

    def update_snapshot(self):
        # Accumulate grads from regular model (do in test mode?)
        self.model.zero_grad()
        for minibatch in self.dataloaders['train']:
            (self.loss(*minibatch, model = self.model)/len(self.dataloaders['train'])).backward()
        self.snapshot_gradient = [p.grad for p in self.model.parameters()]

        flat_g = flatten(self.snapshot_gradient)
        debugstring = "|full_g| {:.3f}".format(flat_g.norm())
        self.logger.add_text('debug_full_grads', debugstring)
        # # Update the weights of the snapshot network
        # with torch.no_grad():
        #     for snapshot_param, param in zip(self.snapshot_model.parameters(),self.model.parameters()):
        #         snapshot_param.data = param.data.clone()
        #         # snapshot_param.detach()
        #         snapshot_param.requires_grad=True
        self.snapshot_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        for param in self.snapshot_model.parameters():
            param.requires_grad = True

