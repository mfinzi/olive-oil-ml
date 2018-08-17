import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import numbers
import torch.nn.functional as F
import numpy as np
from math import ceil, floor

class GaussianNoise(nn.Module):
    """ Layer that adds pixelwise gaussian noise to input (during train)"""
    def __init__(self, std):
        super().__init__()
        self.std = std
    
    def forward(self, x):
        if self.training:
            zeros_ = torch.zeros(x.size()).cuda()
            n = Variable(torch.normal(zeros_, std=self.std).cuda())
            return x + n
        else:
            return x


class RandomHorizontalFlip(nn.Module):
    """ Wraps RandomHorizontalFlip augmentation into a layer (during train)"""
    def __init__(self):
        super().__init__()

    def randomFlip(self, x):
        bs, _, w, h = x.size()

        # Build affine matrices for random translation of each image
        affineMatrices = np.zeros((bs,2,3))
        affineMatrices[:,0,0] = 2*np.random.randint(2,size=bs)-1
        affineMatrices[:,1,1] = 1
        affineMatrices = torch.from_numpy(affineMatrices).float().cuda()

        flowgrid = F.affine_grid(affineMatrices, size = x.size())
        x_out = F.grid_sample(x, flowgrid)
        return x_out

    def forward(self, x):
        if self.training:
            return self.randomFlip(x)
        else:
            return x


# class RandomCrop(nn.Module):
#     """ Wraps RandomCrop augmentation into a layer (during train)"""
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.cropper = transforms.RandomCrop(*args, **kwargs)
    
#     def forward(self, x):
#         if self.training:
#             return self.cropper(x)
#         else:
#             return x

class RandomTranslate(nn.Module):
    """ Wraps RandomCrop augmentation into a layer (during train)
        translations are randomly sampled from integers between
        -max_dist, max_dist inclusive for both x and y, independently
        for each image. Note that zero padding here is after normalization."""
    def __init__(self, max_trans):
        super().__init__()
        self.max_trans = max_trans

    def randomTranslate(self, x):
        bs, _, w, h = x.size()

        # Build affine matrices for random translation of each image
        affineMatrices = np.zeros((bs,2,3))
        affineMatrices[:,0,0] = 1
        affineMatrices[:,1,1] = 1
        affineMatrices[:,0,2] = -2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/w
        affineMatrices[:,1,2] = 2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/h
        affineMatrices = torch.from_numpy(affineMatrices).float().cuda()

        flowgrid = F.affine_grid(affineMatrices, size = x.size())
        x_out = F.grid_sample(x, flowgrid)
        return x_out

    def forward(self, x):
        if self.training:
            return self.randomTranslate(x)
        else:
            return x

    def __repr__(self):
        return self.__class__.__name__ + '(max_trans={0})'.format(self.max_trans)



## A pytorch transforms.transform object, not a layer. Does not support backprop
class LinearTransformationGPU(object):
    def __init__(self, transformation_matrix):
        if transformation_matrix.size(0) != transformation_matrix.size(1):
            raise ValueError("transformation_matrix should be square. Got " +
                             "[{} x {}] rectangular matrix.".format(*transformation_matrix.size()))
        self.transformation_matrix = transformation_matrix.cuda()

    def __call__(self, tensor):
        if tensor.size(0) * tensor.size(1) * tensor.size(2) != self.transformation_matrix.size(0):
            raise ValueError("tensor and transformation matrix have incompatible shape." +
                             "[{} x {} x {}] != ".format(*tensor.size()) +
                             "{}".format(self.transformation_matrix.size(0)))
        flat_tensor = tensor.view(1, -1).cuda()
        transformed_tensor = torch.mm(flat_tensor, self.transformation_matrix)
        tensor = transformed_tensor.view(tensor.size())
        return tensor