import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from ...utils.utils import Expression


def ConvBNrelu(in_channels,out_channels,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1,stride=stride),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class smallCNN(nn.Module):
    """
    Very small CNN
    """
    def __init__(self, numClasses=10,k=16):
        super().__init__()
        self.numClasses = numClasses
        self.net = nn.Sequential(
            ConvBNrelu(3,k),
            ConvBNrelu(k,k),
            ConvBNrelu(k,2*k),
            nn.MaxPool2d(2),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            nn.MaxPool2d(2),
            ConvBNrelu(2*k,2*k),
            ConvBNrelu(2*k,2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,numClasses)
        )
    def forward(self,x):
        return self.net(x)
