import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from ...utils.utils import Expression,export,Named
from .CoordConv import CoordConv

def conv2d(in_channels,out_channels,kernel_size=3,coords=False,**kwargs):
    """ Wraps nn.Conv2d and CoordConv, padding is set to same
        and coords=True can be specified to get additional coordinate in_channels"""
    assert 'padding' not in kwargs, "assumed to be padding = same "
    same = kernel_size//2
    if coords: 
        return CoordConv(in_channels,out_channels,kernel_size,padding=same,**kwargs)
    else: 
        return nn.Conv2d(in_channels,out_channels,kernel_size,padding=same,**kwargs)

class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,ksize=3,stride=1,drop_rate=0,coords=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            conv2d(in_channels,out_channels,ksize,stride=stride,coords=coords),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            conv2d(in_channels,out_channels,ksize,stride=stride,coords=coords),
            nn.Dropout(p=drop_rate)
        )
        if in_channels != out_channels:
            self.shortcut = conv2d(in_channels,out_channels,1,stride=stride,coords=coords)
        elif stride!=1:
            self.shortcut = Expression(lambda x: F.interpolate(x,scale_factor=1/stride))
        else:
            self.shortcut = nn.Sequential()

    def forward(self,x):
        return self.shortcut(x) + self.net(x)

def ConvBNrelu(in_channels,out_channels,kernel_size=3,stride=1,coords=False):
    return nn.Sequential(
        conv2d(in_channels,out_channels,kernel_size,stride=stride,coords=coords),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

def FcBNrelu(in_channels,out_channels):
    return nn.Sequential(
        nn.Linear(in_channels,out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )


class DenseLayer(nn.Module):
    def __init__(self, inplanes, k=12, drop_rate=0,coords=True):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNrelu(inplanes,4*k,kernel_size=1,coords=coords),
            ConvBNrelu(4*k,k,kernel_size=3,coords=coords),
            nn.Dropout(p=drop_rate),
        )
    def forward(self, x):
        return torch.cat((x, self.net(x)), 1)

class DenseBlock(nn.Module):
    def __init__(self, inplanes,k=16,N=20,drop_rate=0,coords=True):
        super().__init__()
        layers = []
        for i in range(N):
            layers.append(DenseLayer(inplanes,k,drop_rate,coords))
            inplanes += k
        layers.append(ConvBNrelu(inplanes,inplanes//2))
        self.net = nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)
