"""
    WideResNet model definition
    ported from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from ...utils.utils import export, Named


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=math.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, drop_rate, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=drop_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

@export
class WideResNet(nn.Module,metaclass=Named):
    def __init__(self, num_targets=10, depth=28, widen_factor=10, drop_rate=0.3,in_channels=3,initial_stride=1):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(in_channels, nstages[0])
        self.layer1 = self._wide_layer(WideBasic, nstages[1], n, drop_rate, stride=initial_stride)
        self.layer2 = self._wide_layer(WideBasic, nstages[2], n, drop_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nstages[3], n, drop_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nstages[3])#, momentum=0.9)
        self.linear = nn.Linear(nstages[3], num_targets)

    def _wide_layer(self, block, planes, num_blocks, drop_rate, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, drop_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = self.linear(out.mean(-1).mean(-1))
        return out
        
@export
class WideResNet28x10(WideResNet):
    def __init__(self,num_targets=10,drop_rate=.3,in_channels=3):
        super().__init__(num_targets,depth=28, widen_factor=10,drop_rate=drop_rate,in_channels=in_channels)

@export
class WideResNet28x10stl(WideResNet):
    def __init__(self,num_targets=10,drop_rate=.3,in_channels=3):
        super().__init__(num_targets,depth=28, widen_factor=10,drop_rate=drop_rate,in_channels=in_channels,initial_stride=2)