import torch, dill
from torch import optim
from ..logging.lazyLogger import LazyLogger
from ..utils.utils import Eval
from ..utils.mytqdm import tqdm
import copy, os, random




class Trainer(object):
    """ Base trainer
        """
    def __init__(self, model, dataloaders, 
                opt_constr=optim.Adam, lr_sched = lambda e: 1, 
                log_dir=None, log_args={}, description='',
                extraInit=lambda:None):

        # Setup model, optimizer, and dataloaders
        self.model = model
        self.optimizer = opt_constr(self.model.parameters())
        # self.lr_scheduler = lr_sched(self.optimizer)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,lr_sched)
        self.dataloaders = dataloaders # A dictionary of dataloaders
        self.epoch = 0

        self.logger = LazyLogger(log_dir, **log_args)
        self.logger.add_text('ModelSpec','model: {}'.format(model))
        self.hypers = {'description':description}
        # Extra work to do (used for subclass)
        extraInit()
        # Log the hyper parameters
        self.logger.add_scalars('ModelSpec', self.hypers)

    def train_to(self, final_epoch=100):
        self.train(final_epoch-self.epoch)

    def train(self, num_epochs=100):
        """ The main training loop"""
        start_epoch = self.epoch
        for self.epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
            self.lr_scheduler.step(self.epoch)
            for i, minibatch in enumerate(self.dataloaders['train']):
                with self.logger as do_log:
                    if do_log: self.logStuff(i, minibatch)
                self.step(*minibatch)
                

    def step(self, *minibatch):
        self.optimizer.zero_grad()
        loss = self.loss(*minibatch)
        loss.backward()
        self.optimizer.step()
        return loss

    def loss(self, *minibatch):
        """ Takes in a minibatch of data and outputs the loss"""
        raise NotImplementedError
    
    def logStuff(self, i, minibatch):
        step = i+1 + (self.epoch+1)*len(self.dataloaders['train'])
        metrics = {'Minibatch_Loss':self.loss(*minibatch).cpu().data.numpy()}
        self.logger.add_scalars('metrics', metrics, step)
        self.logger.report()
    
    def getAverageLoss(self, loader):
        loss_sum, num_total = 0, 0
        with Eval(self.model), torch.no_grad():
            for minibatch in loader:
                mb_size = minibatch[1].size(0)
                loss_sum += self.loss(*minibatch).cpu().data.numpy()
                num_total += mb_size
        return loss_sum/num_total

    def state_dict(self):
        state = {
            'epoch':self.epoch,
            'model_state':self.model.state_dict(),
            'optim_state':self.optimizer.state_dict(),
            'logger_state':self.logger.state_dict(),
        }
        return state

    def load_state_dict(self,state):
        self.epoch = state['epoch']
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optim_state'])
        self.logger.load_state_dict(state['logger_state'])

    def save_checkpoint(self, save_path = None):
        if save_path is None: 
            checkpoint_save_dir = self.logger.log_dir + 'checkpoints/'
            save_path = checkpoint_save_dir + 'c.{}.ckpt'.format(self.epoch)
            #save_path_all = checkpoint_save_dir + 'c.{}.dump'.format(self.epoch)
        else:
            checkpoint_save_dir = os.path.dirname(save_path)
          # so that we use the same subset of data as before
        os.makedirs(checkpoint_save_dir, exist_ok=True)
        torch.save(self.state_dict(), save_path, pickle_module=dill)

    def load_checkpoint(self, load_path):
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            state = torch.load(load_path, pickle_module=dill)
            self.load_state_dict(state)
        else:
            print("=> no checkpoint found at '{}'".format(load_path))


