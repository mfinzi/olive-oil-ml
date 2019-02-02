import torch
import torch.nn as nn
from ..utils.utils import Eval, cosLr, loader_to
from ..utils.metrics import confusion_from_logits, meanIoU, freqIoU, pixelAcc,meanAcc
from .trainer import Trainer
from .classifier import Classifier, Regressor

class Segmenter(Classifier):
    """ Trainer subclass. Implements loss (crossentropy), batchAccuracy
        and getAccuracy (full dataset) """

    def metrics(self,loader):
        mb_confusion = lambda mb: confusion_from_logits(self.model(mb[0]),mb[1])
        full_confusion = self.evalAverageMetrics(loader,mb_confusion)
        metrics = {
            'pixelAcc':pixelAcc(full_confusion),
            'meanAcc':meanAcc(full_confusion)
            'mIoU':meanIoU(full_confusion),
            'fwIoU':freqIoU(full_confusion)
        }
        return metrics

class ImgRegressor(Regressor):
    pass
