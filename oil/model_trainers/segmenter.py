import torch
import torch.nn as nn
from oil.utils.utils import Eval, cosLr, islice
from oil.utils.metrics import confusion_from_logits, meanIoU, freqIoU, pixelAcc,meanAcc
from oil.model_trainers.trainer import Trainer
from oil.model_trainers.classifier import Classifier, Regressor
import torchvision.utils as vutils
import itertools
import math

class Segmenter(Classifier):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        viz_loader = (self.dataloaders['val'] or self.dataloaders['train'])
        self.viz_loader = islice(viz_loader,math.ceil(4/viz_loader.batch_size))
        self.overlay = False

    def metrics(self,loader):
        mb_confusion = lambda mb: confusion_from_logits(self.model(mb[0]),mb[1])
        full_confusion = self.evalAverageMetrics(loader,mb_confusion)
        metrics = {
            'pixelAcc':pixelAcc(full_confusion),
            'meanAcc':meanAcc(full_confusion),
            'mIoU':meanIoU(full_confusion),
            'fwIoU':freqIoU(full_confusion),
        }
        return metrics
    def logStuff(self, step, minibatch=None):
        seg2img = self.viz_loader.dataset.decode_segmap
        means = torch.tensor(self.viz_loader.dataset.means)[None,:,None,None]
        stds = torch.tensor(self.viz_loader.dataset.stds)[None,:,None,None]
        with torch.no_grad():
            xs,ys,gts = zip(*[(mb[0].cpu().data,
                torch.argmax(self.model(mb[0]),1).cpu().data,
                mb[1].cpu().data.squeeze(1)) for mb in self.viz_loader])
            imgs = [torch.cat(xs)*stds+means,seg2img(torch.cat(ys)),seg2img(torch.cat(gts))]
            if self.overlay: imgs = [overlay(imgs[0],imgs[1]),overlay(imgs[0],imgs[2])]
            img_grid = vutils.make_grid(torch.cat(imgs),nrow=len(imgs[0]),range=(0,1))
            self.logger.add_image('Segmentations', img_grid, step)
        super().logStuff(step,minibatch)

def overlay(rgb,segmap,alpha=.5):
    return rgb + alpha*(segmap-rgb)*(segmap.sum(1)!=0)[:,None].float()

class ImgRegressor(Regressor):
    pass







# Convenience function for that covers a common use case of training the model using
#   the cosLr schedule, and logging the outcome and returning the results
from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice
from oil.tuning.study import train_trial
from oil.datasetup.segmentation.voc import VOCSegmentation
from oil.architectures.img2img import DeepLab
import collections

def simpleSegmenterTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'dataset': VOCSegmentation,'network':DeepLab,'net_config': {},
            'loader_config': {'batch_size':2,'pin_memory':True,'num_workers':2,'shuffle':True},
            'opt_config':{'lr':.01, 'momentum':.9, 'weight_decay':1e-4,'nesterov':True},
            'num_epochs':100,'trainer_config':{'log_args':{'timeFrac':1/2}},
            }
        recursively_update(cfg,config)


        trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),image_set='train')
        valset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),image_set='val')
        device = torch.device('cuda')
        fullCNN = cfg['network'](num_classes=trainset.num_classes,**cfg['net_config'])
        # vgg16_backbone = VGG16(pretrained=True)
        # fullCNN.copy_params_from_vgg16(vgg16_backbone)
        fullCNN.to(device).train()
        #fullCNN.load_state_dict(torch.load(cfg['network'].download()))
        dataloaders = {}
        dataloaders['train'] = DataLoader(trainset,**cfg['loader_config'])
        dataloaders['val'] = islice(DataLoader(valset,**cfg['loader_config']),30)
        dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}

        #opt_constr = lambda params: torch.optim.SGD(params, **cfg['opt_config'])
        opt_constr = lambda params: torch.optim.Adam(params, lr=3e-3)
        lr_sched = cosLr()
        return Segmenter(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)
    

if __name__=='__main__':
    Trial = simpleSegmenterTrial(strict=True)
    Trial({'num_epochs':5})