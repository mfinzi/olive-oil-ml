import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from oil.architectures.parts import PointConvSetAbstraction, pConvBNrelu, pBottleneck#,PointConvDensitySetAbstraction
from oil.architectures.parts import pResBlock,Pass
import numpy as np
from torch.nn.utils import weight_norm
import math
from ..parts import conv2d
from ...utils.utils import Expression,export,Named



def logspace(a,b,k):
    return np.exp(np.linspace(np.log(a),np.log(b),k))

@export
class layer13pc(nn.Module,metaclass=Named):
    """
    pointconvnet
    """
    def __init__(self, num_classes=10,k=64,ksize=3,num_layers=4,**kwargs):
        super().__init__()
        self.num_classes = num_classes
        nbhd = int(np.round(ksize**2))
        ds_fracs = 256**(-1/num_layers)
        chs = np.round(logspace(k,4*k,num_layers+1)).astype(int)
        self.initial_conv = conv2d(3,k,1)
        self.net = nn.Sequential(
            *[pConvBNrelu(chs[i],chs[i+1],ds_frac=ds_fracs,nbhd=nbhd,**kwargs) for i in range(num_layers)],
            Expression(lambda u:u[-1].mean(-1)),
            nn.Linear(chs[-1],num_classes)
        )

    def forward(self,x):
        bs,c,h,w = x.shape
        coords = torch.stack(torch.meshgrid([torch.linspace(-1,1,h),torch.linspace(-1,1,w)]),dim=-1).view(h*w,2).unsqueeze(0).permute(0,2,1).repeat(bs,1,1).to(x.device)
        inp_as_points = self.initial_conv(x).view(bs,-1,h*w)
        return self.net((coords,inp_as_points))


@export
class resnetpc(nn.Module,metaclass=Named):
    """
    pointconvnet
    """
    def __init__(self, num_classes=10,k=8,ksize=3.66,num_layers=6,**kwargs):
        super().__init__()
        self.num_classes = num_classes
        nbhd = int(np.round(ksize**2))
        ds_fracs = 256**(-1/num_layers)#logspace(1,1/256,num_layers+1)
        chs = np.round(logspace(16,64*k,num_layers+1)).astype(int)
        #print(chs)
        self.initial_conv = conv2d(3,16,1)
        self.net = nn.Sequential(
            *[pBottleneck(chs[i],chs[i+1],ds_frac=ds_fracs,nbhd=nbhd,**kwargs) for i in range(num_layers)],
            Expression(lambda u:u[-1].mean(-1)),
            nn.Linear(chs[-1],num_classes)
        )

    def forward(self,x):
        bs,c,h,w = x.shape
        coords = torch.stack(torch.meshgrid([torch.linspace(-1,1,h),torch.linspace(-1,1,w)]),dim=-1).view(h*w,2).unsqueeze(0).permute(0,2,1).repeat(bs,1,1).to(x.device)
        inp_as_points = self.initial_conv(x).view(bs,-1,h*w)
        return self.net((coords,inp_as_points))

@export
class pWideResNet(nn.Module,metaclass=Named):
    def __init__(self, num_classes=10, depth=28, widen_factor=10, drop_rate=0.3,nbhd=9):
        super().__init__()
        self.in_planes = 16
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor
        nstages = [16, 16 * k, 32 * k, 64 * k]
        self.initial_conv = conv2d(3,nstages[0],1)
        self.net = nn.Sequential(
            self._wide_layer(pResBlock, nstages[1], n, drop_rate, stride=1,nbhd=nbhd),
            self._wide_layer(pResBlock, nstages[2], n, drop_rate, stride=2,nbhd=nbhd),
            self._wide_layer(pResBlock, nstages[3], n, drop_rate, stride=2,nbhd=nbhd),
            Pass(nn.BatchNorm1d(nstages[3])),#,momentum=0.9)),
            Pass(nn.ReLU()),
            Expression(lambda u:u[-1].mean(-1)),
            nn.Linear(nstages[3],num_classes)
        )
    def _wide_layer(self, block, planes, num_blocks, drop_rate, stride,nbhd):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, drop_rate, stride,nbhd=nbhd))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self,x):
        bs,c,h,w = x.shape
        a = np.sqrt(3)
        coords = torch.stack(torch.meshgrid([torch.linspace(-a,a,h),torch.linspace(-a,a,w)]),dim=-1).view(h*w,2).unsqueeze(0).permute(0,2,1).repeat(bs,1,1).to(x.device)
        inp_as_points = self.initial_conv(x).view(bs,-1,h*w)
        return self.net((coords,inp_as_points))

@export
class colorEquivariantLayer13pc(layer13pc):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,xyz_dim=5,knn_channels=2,**kwargs)
    def forward(self,x):
        bs,c,h,w = x.shape
        coords = torch.stack(torch.meshgrid([torch.linspace(-1,1,h),torch.linspace(-1,1,w)]),dim=-1).view(h*w,2).unsqueeze(0).permute(0,2,1).repeat(bs,1,1).to(x.device)
        coords_w_color = torch.cat([coords,x.view(bs,-1,h*w)],dim=1)
        inp_as_points = self.initial_conv(torch.randn_like(x)).view(bs,-1,h*w)
        return self.net((coords_w_color,inp_as_points))

@export
class colorEquivariantResnetpc(resnetpc):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,xyz_dim=5,knn_channels=2,**kwargs)
    def forward(self,x):
        bs,c,h,w = x.shape
        coords = torch.stack(torch.meshgrid([torch.linspace(-1,1,h),torch.linspace(-1,1,w)]),dim=-1).view(h*w,2).unsqueeze(0).permute(0,2,1).repeat(bs,1,1).to(x.device)
        coords_w_color = torch.cat([coords,x.view(bs,-1,h*w)],dim=1)
        inp_as_points = self.initial_conv(torch.randn_like(x)).view(bs,-1,h*w)
        return self.net((coords_w_color,inp_as_points))

@export
class layer13pc(nn.Module,metaclass=Named):
    """
    pointconvnet
    """
    def __init__(self, num_classes=10,k=64,ksize=3.66,num_layers=4,**kwargs):
        super().__init__()
        self.num_classes = num_classes
        nbhd = int(np.round(ksize**2))
        ds_fracs = 256**(-1/num_layers)
        chs = np.round(logspace(k,4*k,num_layers+1)).astype(int)
        self.initial_conv = conv2d(3,k,1)
        self.net = nn.Sequential(
            *[pConvBNrelu(chs[i],chs[i+1],ds_frac=ds_fracs,nbhd=nbhd,**kwargs) for i in range(num_layers)],
            Expression(lambda u:u[-1].mean(-1)),
            nn.Linear(chs[-1],num_classes)
        )

    def forward(self,x):
        bs,c,h,w = x.shape
        coords = torch.stack(torch.meshgrid([torch.linspace(-1,1,h),torch.linspace(-1,1,w)]),dim=-1).view(h*w,2).unsqueeze(0).permute(0,2,1).repeat(bs,1,1).to(x.device)
        inp_as_points = self.initial_conv(x).view(bs,-1,h*w)
        return self.net((coords,inp_as_points))

@export
class PointConv3d(layer13pc):
    def __init__(self,num_classes=40,k=64,ksize=np.sqrt(32),xyz_dim=3,**kwargs):
        super().__init__(num_classes=num_classes,k=k,ksize=ksize,xyz_dim=xyz_dim,**kwargs)
        self.k = k

    def forward(self,x):
        bs,c,n = x.shape
        assert c==3, "expected points living in 3d"
        noise_input = torch.randn(bs,self.k,n).to(x.device)
        return self.net((x,noise_input))