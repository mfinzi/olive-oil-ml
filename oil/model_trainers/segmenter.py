import torch
import torch.nn as nn
from oil.utils.utils import Eval, cosLr
from oil.utils.metrics import confusion_from_logits, meanIoU, freqIoU, pixelAcc,meanAcc
from oil.model_trainers.trainer import Trainer
from oil.model_trainers.classifier import Classifier, Regressor

class Segmenter(Classifier):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def metrics(self,loader):
        mb_confusion = lambda mb: confusion_from_logits(self.model(mb[0]),mb[1])
        full_confusion = self.evalAverageMetrics(loader,mb_confusion)
        print("confusion_matrix: {}".format(full_confusion))
        metrics = {
            'pixelAcc':pixelAcc(full_confusion),
            'meanAcc':meanAcc(full_confusion),
            'mIoU':meanIoU(full_confusion),
            'fwIoU':freqIoU(full_confusion),
        }
        return metrics

class ImgRegressor(Regressor):
    pass







# Convenience function for that covers a common use case of training the model using
#   the cosLr schedule, and logging the outcome and returning the results
from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, recursively_update,islice
from oil.tuning.study import train_trial
from oil.datasetup.segmentation.voc import VOCSegmentation
from oil.architectures.img2img import FCN32s, VGG16
import collections

def simpleSegmenterTrial(strict=False):
    def makeTrainer(config):
        cfg = {
            'dataset': VOCSegmentation,'network':FCN32s,'net_config': {},
            'loader_config': {'batch_size':1,'pin_memory':True,'num_workers':1,'shuffle':True},
            'opt_config':{'lr':.1, 'momentum':.9, 'weight_decay':1e-4,'nesterov':True},
            'num_epochs':100,'trainer_config':{},
            }
        recursively_update(cfg,config)


        trainset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),image_set='train')
        valset = cfg['dataset']('~/datasets/{}/'.format(cfg['dataset']),image_set='val')
        device = torch.device('cuda')
        fullCNN = cfg['network'](num_classes=trainset.num_classes,**cfg['net_config'])
        vgg16_backbone = VGG16(pretrained=True)
        fullCNN.copy_params_from_vgg16(vgg16_backbone)
        fullCNN.to(device).train()
        #fullCNN.load_state_dict(torch.load(cfg['network'].download()))
        dataloaders = {}
        dataloaders['train'] = DataLoader(trainset,**cfg['loader_config'])
        dataloaders['val'] = islice(DataLoader(valset,shuffle=True),300)
        dataloaders = {k:LoaderTo(v,device) for k,v in dataloaders.items()}

        #opt_constr = lambda params: torch.optim.SGD(params, **cfg['opt_config'])
        opt_constr = lambda params: torch.optim.Adam(params, lr=1e-3)
        lr_sched = cosLr()
        return Segmenter(fullCNN,dataloaders,opt_constr,lr_sched,**cfg['trainer_config'])
    return train_trial(makeTrainer,strict)
    

if __name__=='__main__':
    Trial = simpleSegmenterTrial(strict=True)
    Trial({'num_epochs':5})