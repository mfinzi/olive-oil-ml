

from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, recursively_update
from oil.tuning.study import train_trial
from oil.datasetup.dataloaders import getLabLoader
from oil.datasetup.segmentation.voc import VOCSegmentation
from oil.architectures.img_classifiers import layer13s
import collections
import os

trainset = VOCSegmentation('~/datasets/PascalVOC2012')