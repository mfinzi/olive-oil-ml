import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm
from ...utils.utils import Expression,export,Named
#from ...datasetup.augLayers import GaussianNoise
# weight init is automatically done in the module initialization
# see https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py

def weight_init_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
        
    def forward(self, x):
        return self.func(x)


@export
class layer13(nn.Module,metaclass=Named):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, num_targets=10):
        super().__init__()
        self.numClasses = num_targets
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(0.5)
        
        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(0.5)
        
        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  weight_norm(nn.Linear(128, num_targets))
    
    def forward(self, x, getFeatureVec=False):
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop1(x)
        
        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop2(x)
        
        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)
        
        x = x.view(-1, 128)
        if getFeatureVec: return x
        else: return self.fc1(x)




class ConvSmall(nn.Module,metaclass=Named):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, numClasses=10):
        super().__init__()
        self.numClasses = numClasses
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 96, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(96)
        self.conv1b = weight_norm(nn.Conv2d(96, 96, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(96)
        self.conv1c = weight_norm(nn.Conv2d(96, 96, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(96)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(0.5)
        
        self.conv2a = weight_norm(nn.Conv2d(96, 192, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(192)
        self.conv2b = weight_norm(nn.Conv2d(192, 192, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(192)
        self.conv2c = weight_norm(nn.Conv2d(192, 192, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(192)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(0.5)
        
        self.conv3a = weight_norm(nn.Conv2d(192, 192, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(192)
        self.conv3b = weight_norm(nn.Conv2d(192, 192, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(192)
        self.conv3c = weight_norm(nn.Conv2d(192, 192, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(192)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  weight_norm(nn.Linear(192, numClasses))
    
    def forward(self, x, getFeatureVec=False):
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop1(x)
        
        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop2(x)
        
        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)
        
        x = x.view(-1, 192)
        if getFeatureVec: return x
        else: return self.fc1(x)



class ConvSmallNWN(nn.Module,metaclass=Named):
    """
    CNN from Mean Teacher paper
    """
    
    def __init__(self, numClasses=10):
        super().__init__()
        self.numClasses = numClasses
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = nn.Conv2d(3, 96, 3, padding=1)
        self.bn1a = nn.BatchNorm2d(96)
        self.conv1b = nn.Conv2d(96, 96, 3, padding=1)
        self.bn1b = nn.BatchNorm2d(96)
        self.conv1c = nn.Conv2d(96, 96, 3, padding=1)
        self.bn1c = nn.BatchNorm2d(96)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(0.5)
        
        self.conv2a = nn.Conv2d(96, 192, 3, padding=1)
        self.bn2a = nn.BatchNorm2d(192)
        self.conv2b = nn.Conv2d(192, 192, 3, padding=1)
        self.bn2b = nn.BatchNorm2d(192)
        self.conv2c = nn.Conv2d(192, 192, 3, padding=1)
        self.bn2c = nn.BatchNorm2d(192)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(0.5)
        
        self.conv3a = nn.Conv2d(192, 192, 3, padding=0)
        self.bn3a = nn.BatchNorm2d(192)
        self.conv3b = nn.Conv2d(192, 192, 1, padding=0)
        self.bn3b = nn.BatchNorm2d(192)
        self.conv3c = nn.Conv2d(192, 192, 1, padding=0)
        self.bn3c = nn.BatchNorm2d(192)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  nn.Linear(192, numClasses)
    
    def forward(self, x, getFeatureVec=False):
        x = self.activation(self.bn1a(self.conv1a(x)))
        x = self.activation(self.bn1b(self.conv1b(x)))
        x = self.activation(self.bn1c(self.conv1c(x)))
        x = self.mp1(x)
        x = self.drop1(x)
        
        x = self.activation(self.bn2a(self.conv2a(x)))
        x = self.activation(self.bn2b(self.conv2b(x)))
        x = self.activation(self.bn2c(self.conv2c(x)))
        x = self.mp2(x)
        x = self.drop2(x)
        
        x = self.activation(self.bn3a(self.conv3a(x)))
        x = self.activation(self.bn3b(self.conv3b(x)))
        x = self.activation(self.bn3c(self.conv3c(x)))
        x = self.ap3(x)
        
        x = x.view(-1, 192)
        if getFeatureVec: return x
        else: return self.fc1(x)