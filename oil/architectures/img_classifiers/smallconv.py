
import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from ...utils.utils import Expression,export,Named
from ..parts import ResBlock, DeConv2d, FastDeconv, MaxBlurPool,BlurPool


def ConvBNrelu(in_channels,out_channels,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def ConvDrelu(in_channels,out_channels,stride=1):
    return nn.Sequential(
        FastDeconv(in_channels,out_channels,3,padding=1,stride=stride),
        nn.ReLU()
    )

def ConvGNrelu(in_channels,out_channels,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride),
        nn.GroupNorm(out_channels//16,out_channels),
        nn.ReLU()
    )
@export
class smallCNN(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, num_targets=10,in_channels=3,k=16):
        super().__init__()
        self.num_targets = num_targets
        self.net = nn.Sequential(
            ConvBNrelu(in_channels,k),
            ConvBNrelu(k,k),
            ConvBNrelu(k,2*k),
            nn.MaxPool2d(2),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_targets)
        )
    def forward(self,x):
        return self.net(x)

@export
class layer13s(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, num_targets=10,in_channels=3,k=128):
        super().__init__()
        self.num_targets = num_targets
        self.net = nn.Sequential(
            ConvBNrelu(in_channels,k),
            ConvBNrelu(k,k),
            ConvBNrelu(k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            #nn.Dropout2d(),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            #nn.Dropout2d(),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_targets)
        )
    def forward(self,x):
        return self.net(x)

@export
class layer13d(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, num_targets=10,in_channels=3,k=128):
        super().__init__()
        self.num_targets = num_targets
        self.net = nn.Sequential(
            ConvDrelu(in_channels,k),
            ConvDrelu(k,k),
            ConvDrelu(k,2*k),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            ConvDrelu(2*k,2*k),
            ConvDrelu(2*k,2*k),
            ConvDrelu(2*k,2*k),
            nn.MaxPool2d(2),
            nn.Dropout2d(),
            ConvDrelu(2*k,2*k),
            ConvDrelu(2*k,2*k),
            ConvDrelu(2*k,2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_targets)
        )
    def forward(self,x):
        return self.net(x)