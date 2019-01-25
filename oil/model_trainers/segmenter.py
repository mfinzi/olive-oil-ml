import torch
import torch.nn as nn
from ..utils.utils import Eval, cosLr, loader_to
from .trainer import Trainer
from .classifier import Classifier, Regressor

class Segmenter(Classifier):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """
    # IoU, MaP, MP
    def evalAccuracy(self,loader,model=None):
        acc = lambda mb: model(mb[0]).max(1)[1].type_as(mb[1]).eq(mb[1]).float().mean()
        return self.evalAverageLoss(loader,model,acc)

    #metrics = {**Trainer.__metrics__,'Acc':evalAccuracy}

class ImgRegressor(Regressor):
    pass