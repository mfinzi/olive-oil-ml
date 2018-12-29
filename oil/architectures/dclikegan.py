# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F
import numpy as np
#from .spectral_normalization import SpectralNorm
#from torch.nn.utils import spectral_norm
from ..utils.utils import Expression
from .ganBase import GanBase, add_spectral_norm, xavier_uniform_init


w_g = 4

class Generator(GanBase):
    def __init__(self, z_dim=128,img_channels=3):
        super().__init__(z_dim,img_channels)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1,1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, img_channels, 3, stride=1, padding=(1,1)),
            nn.Tanh())

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))

class Discriminator(nn.Module):
    def __init__(self,img_channels=3, out_size=1):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(img_channels, 64, 3, stride=1, padding=(1,1))

        self.conv2 = nn.Conv2d(64, 64, 4, stride=2, padding=(1,1))
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=(1,1))
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2, padding=(1,1))
        self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=(1,1))
        self.conv6 = nn.Conv2d(256, 256, 4, stride=2, padding=(1,1))
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=(1,1))


        self.fc = nn.Linear(w_g * w_g * 512, out_size)
        #self.apply(xavier_uniform_init)
        self.apply(add_spectral_norm)
    def forward(self, x):
        m = x
        m = nn.LeakyReLU(.1)(self.conv1(m))
        m = nn.LeakyReLU(.1)(self.conv2(m))
        m = nn.LeakyReLU(.1)(self.conv3(m))
        m = nn.LeakyReLU(.1)(self.conv4(m))
        m = nn.LeakyReLU(.1)(self.conv5(m))
        m = nn.LeakyReLU(.1)(self.conv6(m))
        m = nn.LeakyReLU(.1)(self.conv7(m))

        return self.fc(m.view(-1,w_g * w_g * 512))