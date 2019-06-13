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
        super().__init__(*args,**kwargs)
        self.hypers['l1'] = l1

    def genLoss(self,data):
        y,x = data # Here y is the output image and x is input
        fake_y = self.G(y)
        adversarial = hinge_loss_G(self.D(fake_y,x))
        return adversarial + self.hypers['l1']*L1(fake_y,y)

    def logStuff(self, i, minibatch=None):
        raise NotImplementedError












from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, recursively_update,imap,islice
from oil.tuning.study import train_trial
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.datasets import CIFAR10
from oil.architectures.img_classifiers import layer13s
from oil.architectures.img_gen import conditionalgan

def simpleGanTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'loader_config':{'amnt_dev':0,'lab_BS':64,'dataseed':0,'num_workers':1},
            'gen':conditionalgan.Generator,'disc':conditionalgan.Discriminator,
            'trainer':cGan,
            'trainer_config':{'n_disc':2},
            'opt_config':{'lr':2e-4,'betas':(.5,.999)},
            'num_epochs':400,
        }
        recursively_update(cfg,config)
        trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),gan_normalize=True)
        device = torch.device('cuda')
        G = cfg['gen'](num_classes=10).to(device)
        D = cfg['disc'](num_classes=10).to(device)

        dataloaders = {}
        dataloaders['train'], _ = getLabLoader(trainset,**cfg['loader_config'])
        imgs_only  = imap(lambda z: z[0], dataloaders['train'])
        dataloaders['dev'] = islice(imgs_only,5000//cfg['loader_config']['lab_BS'])
        dataloaders = {k: LoaderTo(v,device) for k,v in dataloaders.items()}
        opt_constr = lambda params: torch.optim.Adam(params,**cfg['opt_config'])
        lr_sched = cosLr(cfg['num_epochs'])
        trainer = cfg['trainer'](G,dataloaders,opt_constr=opt_constr,
                                lr_sched=lr_sched,D=D,**cfg['trainer_config'])
        return trainer
    return train_trial(makeTrainer,strict)

if __name__=='__main__':
    Trial = simpleGanTrial(strict=True)
    Trial({'num_epochs':4*(100,)})