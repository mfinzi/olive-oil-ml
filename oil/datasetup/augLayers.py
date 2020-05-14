import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.nn as nn
import numbers
import torch.nn.functional as F
import numpy as np
from math import ceil, floor
from ..utils.utils import log_uniform

    
class RandomErasing(nn.Module):
    '''
    Augmentation module that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    ave_area_frac: average fraction of img area that is erased
    '''
    def __init__(self, p = 1, af=1/4, ar=3,max_scale=3):
        self.p = p
        self.area_frac = af
        self.max_ratio = ar
        self.max_scale=max_scale
        super().__init__()

    def forward(self, x):
        if self.training:
            return self.random_erase(x)
        else:
            return x

    def random_erase(self, img):
        bs,c,h,w = img.shape
        area = h*w
        target_areas = log_uniform(1/self.max_scale, self.max_scale,size=bs)*self.area_frac*area
        aspect_ratios = log_uniform(1/self.max_ratio, self.max_ratio,size=bs)

        do_erase = np.random.random(bs)<self.p
        cut_hs = np.sqrt(target_areas * aspect_ratios)*do_erase
        cut_ws = np.sqrt(target_areas / aspect_ratios)*do_erase
        cut_i = np.random.randint(h,size=bs)
        cut_j = np.random.randint(h,size=bs)

        i,j = np.mgrid[:h,:w]
        ui = (cut_i+cut_hs/2)[:,None,None]
        li = (cut_i-cut_hs/2)[:,None,None]
        uj = (cut_j+cut_ws/2)[:,None,None]
        lj = (cut_j-cut_ws/2)[:,None,None]
        no_erase_mask = ~((li<i)&(i<ui)&(lj<j)&(j<uj))[:,None,:,:]
        no_erase_tensor = torch.from_numpy(no_erase_mask.astype(np.float32)).to(img.device)
        return img*no_erase_tensor

class Cutout(RandomErasing):
    """ A simplificaiton to the square case with deterministic size: cutout (works a bit worse)"""
    def __init__(self,area_frac=.2):
        super().__init__(p=1,ave_area_frac=area_frac,
                        max_aspect_ratio=1,max_scale=1)


class GaussianNoise(nn.Module):
    """ Layer that adds pixelwise gaussian noise to input (during train)"""
    def __init__(self, std):
        super().__init__()
        self.std = std
    
    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x)*self.std
        else:
            return x

class PointcloudScale(nn.Module):
    def __init__(self, r=1.25):
        super().__init__()
        self.r=r
    def forward(self, x):
        if self.training:
            return x + self.r**torch.rand(x.shape[0]).to(x.device)[:,None,None] # one scale per example
        else:
            return x

class RandomZrotation(nn.Module):
    def __init__(self,max_angle=np.pi):
        super().__init__()
        self.max_angle = max_angle
    def forward(self,x):
        if self.training:
            # this presumes z axis is coordinate 2?
            # assumes x has shape B3N
            bs,c,n = x.shape; assert c==3
            angles = (2*torch.rand(bs)-1)*self.max_angle
            R = torch.zeros(bs,3,3)
            R[:,2,2] = 1
            R[:,0,0] = R[:,1,1] = angles.cos()
            R[:,0,1] = R[:,1,0] = angles.sin()
            R[:,1,0] *=-1
            return R.to(x.device)@x
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
        affineMatrices = torch.from_numpy(affineMatrices).float().to(x.device)

        flowgrid = F.affine_grid(affineMatrices, size = x.size(),align_corners=True)
        x_out = F.grid_sample(x, flowgrid,align_corners=True)
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
        affineMatrices = torch.from_numpy(affineMatrices).float().to(x.device)

        flowgrid = F.affine_grid(affineMatrices, size = x.size(),align_corners=True)
        x_out = F.grid_sample(x, flowgrid,align_corners=True)
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