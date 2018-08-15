from torch import optim
from .lazyLogger import LazyLogger
from .utils import reconstructor
import copy, os, random

class Trainer(object):

    def __init__(self, model, dataloaders, 
                opt_constr=optim.Adam, lr_sched = lambda e: 1, 
                log_dir=None, log_args={},
                amnt_lab=1, amnt_dev=0, description='',
                extraInit=lambda:None, data_seed=0, rebuildable=True):

        # Setup model, optimizer, and dataloaders
        self.model = model
        self.optimizer = opt_constr(self.model.parameters())
        try: self.lr_scheduler = lr_sched(self.optimizer)
        except: self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,lr_sched)
        if isinstance(dataloaders,(tuple,list)):
            self.train_iter = dataloaders[0]
        else:  
            self.train_iter = dataloaders
        self.epoch = 0

        self.logger = LazyLogger(log_dir, **log_args)
        self.logger.add_text('ModelSpec','model: '+type(model).__name__)
        self.hypers = {'description':description}
        # Extra work to do (used for subclass)
        extraInit()
        # Log the hyper parameters
        for tag, value in self.hypers.items():
            self.logger.add_text('ModelSpec',tag+' = '+str(value))
        # Magic method for capturing a closure with this init function
        self.reconstructor = reconstructor() if rebuildable else None

    def train_to(self, final_epoch=100):
        self.train(final_epoch-self.epoch)

    def train(self, num_epochs=100):
        """ The main training loop called (also for subclasses)"""
        start_epoch = self.epoch
        for self.epoch in range(start_epoch, start_epoch + num_epochs):
            self.lr_scheduler.step(self.epoch)
            for i, train_data in enumerate(self.train_iter):
                self.step(*train_data)
                self.logger.maybe_do(self.logStuff,
                        *(i,self.epoch,start_epoch+num_epochs,train_data))

    def step(self, *data):
        self.optimizer.zero_grad()
        loss = self.loss(*data)
        loss.backward()
        self.optimizer.step()

    def loss(self, *data):
        """ Takes in a minibatch of data """
        raise NotImplementedError
    
    def logStuff(self, *args):
        raise NotImplementedError

    def state_dict(self):
        state = {
            'epoch':self.epoch,
            'model_state':self.model.state_dict(),
            'optim_state':self.optimizer.state_dict(),
            'logger_state':self.logger.state_dict(),
            'reconstructor':self.reconstructor,
        }
        return state

    def load_state_dict(self,state):
        self.epoch = state['epoch']
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optim_state'])
        self.logger.load_state_dict(state['logger_state'])
        self.reconstructor = state['reconstructor']

    def save_checkpoint(self, save_path = None):
        if save_path is None: 
            checkpoint_save_dir = self.logger.get_dir() + 'checkpoints/'
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


