import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from oil.architectures.parts import PointConvSetAbstraction, pConvBNrelu, pBottleneck#,PointConvDensitySetAbstraction
from oil.architectures.parts import pResBlock,Pass, imgpConvBNrelu
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
    def __init__(self, num_classes=10,k=64,ksize=3,num_layers=4,total_ds=1/256,**kwargs):
        super().__init__()
        self.num_classes = num_classes
        nbhd = int(np.round(ksize**2))
        ds_fracs = total_ds**(1/num_layers)
        chs = np.round(logspace(k,4*k,num_layers+1)).astype(int)
        self.initial_conv = conv2d(3,k,1)
        self.net = nn.Sequential(
            *[pConvBNrelu(chs[i],chs[i+1],ds_frac=ds_fracs,nbhd=nbhd,**kwargs) for i in range(num_layers)],
            Expression(lambda u:u[-1].mean(-1)),
            nn.Linear(chs[-1],num_classes)
        )
        self.k=k

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
    def __init__(self, num_classes=10,k=8,ksize=3.66,num_layers=6,total_ds=1/256,**kwargs):
        super().__init__()
        self.num_classes = num_classes
        nbhd = int(np.round(ksize**2))
        ds_fracs = total_ds**(1/num_layers)#logspace(1,1/256,num_layers+1)
        chs = np.round(logspace(16,64*k,num_layers+1)).astype(int)
        #print(chs)
        self.initial_conv = conv2d(3,16,1)
        self.net = nn.Sequential(
            *[pBottleneck(chs[i],chs[i+1],ds_frac=ds_fracs,nbhd=nbhd,r=1,**kwargs) for i in range(num_layers)],
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
    def __init__(self, num_classes=10, depth=28, widen_factor=10, drop_rate=0.3,nbhd=9,xyz_dim=2):
        super().__init__()
        self.in_planes = 16
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor
        nstages = [16, 16 * k, 32 * k, 64 * k]
        self.initial_conv = conv2d(3,nstages[0],1)
        self.net = nn.Sequential(
            self._wide_layer(pResBlock, nstages[1], n, drop_rate, stride=1,nbhd=nbhd,xyz_dim=xyz_dim),
            self._wide_layer(pResBlock, nstages[2], n, drop_rate, stride=2,nbhd=nbhd,xyz_dim=xyz_dim),
            self._wide_layer(pResBlock, nstages[3], n, drop_rate, stride=2,nbhd=nbhd,xyz_dim=xyz_dim),
            Pass(nn.BatchNorm1d(nstages[3])),#,momentum=0.9)),
            Pass(nn.ReLU()),
            Expression(lambda u:u[-1].mean(-1)),
            nn.Linear(nstages[3],num_classes)
        )
    def _wide_layer(self, block, planes, num_blocks, drop_rate, stride,nbhd,xyz_dim):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, drop_rate, stride,nbhd=nbhd,xyz_dim=xyz_dim))
            self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self,x):
        bs,c,h,w = x.shape
        a = np.sqrt(3)
        coords = torch.stack(torch.meshgrid([torch.linspace(-a,a,h),torch.linspace(-a,a,w)]),dim=-1).view(h*w,2).unsqueeze(0).permute(0,2,1).repeat(bs,1,1).to(x.device)
        inp_as_points = self.initial_conv(x).view(bs,-1,h*w)
        return self.net((coords,inp_as_points))

@export
class pointWideResNet(pWideResNet):
    def __init__(self,num_classes=40,depth=16,widen_factor=6,nbhd=32,xyz_dim=3,drop_rate=0,**kwargs):
        super().__init__(num_classes=num_classes,depth=depth,widen_factor=widen_factor,
                        nbhd=nbhd,drop_rate=drop_rate,xyz_dim=xyz_dim,**kwargs)

    def forward(self,x):
        bs,c,n = x.shape
        assert c==3, "expected points living in 3d"
        noise_input = torch.randn(bs,16,n).to(x.device)
        return self.net((x,noise_input))

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
class PointConv3d(layer13pc):
    def __init__(self,num_classes=40,k=64,ksize=np.sqrt(32),xyz_dim=3,**kwargs):
        super().__init__(num_classes=num_classes,k=k,ksize=ksize,xyz_dim=xyz_dim,**kwargs)
        self.k = k

    def forward(self,x):
        bs,c,n = x.shape
        assert c==3, "expected points living in 3d"
        noise_input = torch.randn(bs,self.k,n).to(x.device)
        return self.net((x,noise_input))

@export
class PointRes3d(resnetpc):
    def __init__(self,num_classes=40,k=8,ksize=np.sqrt(32),xyz_dim=3,**kwargs):
        super().__init__(num_classes=num_classes,k=k,ksize=ksize,xyz_dim=xyz_dim,**kwargs)
        self.k = k

    def forward(self,x):
        bs,c,n = x.shape
        assert c==3, "expected points living in 3d"
        noise_input = torch.randn(bs,16,n).to(x.device)
        return self.net((x,noise_input))

@export
class mnistLayer13pc(layer13pc):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.initial_conv = nn.Conv1d(1,self.k,1)
    def forward(self,x):
        coords,vals = x
        return self.net((coords,self.initial_conv(vals)))


@export
class player13s(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, num_classes=10,k=128):
        super().__init__()
        self.num_classes = num_classes
        cbr = lambda c1,c2: imgpConvBNrelu(c1,c2,ds_frac=1,nbhd=9)
        self.net = nn.Sequential(
            cbr(3,k),
            cbr(k,k),
            cbr(k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            cbr(2*k,2*k),
            cbr(2*k,2*k),
            cbr(2*k,2*k),
            nn.MaxPool2d(2),#MaxBlurPool(2*k),
            cbr(2*k,2*k),
            cbr(2*k,2*k),
            cbr(2*k,2*k),
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_classes)
        )
    def forward(self,x):
        return self.net(x)


# @export
# def pConvBNrelu(in_channels,out_channels,**kwargs):
#     return nn.Sequential(
#         PointConv(in_channels,out_channels,**kwargs),
#         Pass(nn.BatchNorm1d(out_channels)),
#         Pass(nn.ReLU())
#     )