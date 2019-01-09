# ResNet generator and discriminator
from torch import nn
import torch.nn.functional as F
import numpy as np
#from .spectral_normalization import SpectralNorm
#from torch.nn.utils import spectral_norm
from ...utils.utils import Expression
from .ganBase import GanBase, add_spectral_norm, xavier_uniform_init



class Generator(GanBase):
    def __init__(self, z_dim=128,img_channels=3,k=128):
        super().__init__(z_dim,img_channels)
        self.k = k

        self.model = nn.Sequential(
            nn.Linear(z_dim, 4 * 4 * k),
            Expression(lambda z: z.view(-1,k,4,4)),
            ResBlockGenerator(k, k, stride=2),
            ResBlockGenerator(k, k, stride=2),
            ResBlockGenerator(k, k, stride=2),
            nn.BatchNorm2d(k),
            nn.ReLU(),
            nn.Conv2d(k, img_channels, 3, stride=1, padding=1),
            nn.Tanh())

        self.apply(xavier_uniform_init)
        #self.apply(add_spectral_norm)
    def forward(self, z):
        return self.model(z)

    
class Discriminator(nn.Module):
    def __init__(self,img_channels=3,k=128,out_size=1):
        super().__init__()
        self.img_channels = img_channels
        self.k = k
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(img_channels, k, stride=2),
                ResBlockDiscriminator(k, k, stride=2),
                ResBlockDiscriminator(k, k),
                ResBlockDiscriminator(k, k),
                nn.ReLU(),
                nn.AvgPool2d(8),
                Expression(lambda u: u.view(-1,k)),
                nn.Linear(k, out_size)
            )
        self.apply(xavier_uniform_init)
        self.apply(add_spectral_norm)

    def forward(self, x):
        return self.model(x)

class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, 1, 1, padding=0),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )
    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, padding=0),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)



