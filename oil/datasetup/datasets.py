import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as ds
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import augLayers
from ..utils.utils import Named
from . import camvid
from .celeba import CelebA
class EasyIMGDataset(Dataset,metaclass=Named):
    ignored_index = -100
    class_weights = None
    def __init__(self,*args,gan_normalize=False,flow=False,download=True,**kwargs):
        transform = kwargs.pop('transform',None)
        if not transform: transform = self.default_transform(gan_normalize,flow)
        super().__init__(*args,transform=transform,download=download,**kwargs)
    
    def default_transform(self,gan_normalize=False,flow=False):
        if flow:
            return transforms.ToTensor()
        elif gan_normalize: 
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            normalize = transforms.Normalize(self.means, self.stds)
        transform = transforms.Compose([transforms.ToTensor(),normalize])
        return transform

    # def compute_default_transform(self):
    #     raise NotImplementedError

    def default_aug_layers(self):
        return nn.Sequential()

# class InMemoryDataset(EasyIMGDataset):
#     def __init__(self,*args,**kwargs):
#         super().__init__(*args,**kwargs)
#         self.data = F.to_tensor(self.data)
#     def to(self,device):
#         self.data.to(device)
#         self.targets.to(device)
#         return self


class CIFAR10(EasyIMGDataset,ds.CIFAR10):
    means = (0.4914, 0.4822, 0.4465)
    stds = (.247,.243,.261)
    num_classes=10
    def default_aug_layers(self):
        return nn.Sequential(
        augLayers.RandomTranslate(4),
        augLayers.RandomHorizontalFlip(),
        )

class CIFAR100(EasyIMGDataset,ds.CIFAR100):
    means = (0.5071, 0.4867, 0.4408)
    stds = (0.2675, 0.2565, 0.2761)
    num_classes=100
    def default_aug_layers(self):
        return nn.Sequential(
        augLayers.RandomTranslate(4),
        augLayers.RandomHorizontalFlip(),
        )

class SVHN(EasyIMGDataset,ds.SVHN):
    #TODO: Find real mean and std
    means = (0.5, 0.5, 0.5)
    stds = (0.25, 0.25, 0.25)
    num_classes=10
    def default_aug_layers(self):
        return nn.Sequential(
        augLayers.RandomTranslate(4),
        augLayers.RandomHorizontalFlip(),
        )

# class SegmentationDataset(EasyIMGDataset):
#     def __init__(self,*args,joint_transform=True,split='train',**kwargs):
#         if joint_transform is True:
#             joint_transform = self.default_joint_transform() if \
#                 split=='train' else None
#         super().__init__(*args,joint_transform=joint_transform,
#                                 split=split,**kwargs)

#     def default_joint_transform(self):
#         """ Currently translating x and y is more easily
#             expressed as a joint transformation rather than layer """
#         raise NotImplementedError
    
# class CamVid(camvid.CamVid):
#     @classmethod
#     def default_joint_transform(self):
#         return transforms.Compose([
#                 JointRandomCrop(224),
#                 JointRandomHorizontalFlip()
#                 ])




# def CIFAR10ZCA():
#     """ Note, currently broken and doesn't support data aug """
#     transform_dev = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((.0904,.0868,.0468), (1,1,1))])
#     transform_train = transform_dev
#     pathToDataset = '/scratch/datasets/cifar10/'
#     trainset = ds.CIFAR10(pathToDataset, download=True, transform=transform_train)
#     testset = ds.CIFAR10(pathToDataset, train=False, download=True, transform=transform_dev)
#     try: ZCAt_mat = torch.load("ZCAtranspose.np")
#     except: ZCAt_mat = constructCifar10ZCA(trainset)
#     trainset.train_data = np.dot(trainset.train_data.reshape(-1,32*32*3), ZCAt_mat).reshape(-1,32,32,3)
#     testset.test_data = np.dot(testset.test_data.reshape(-1,32*32*3), ZCAt_mat).reshape(-1,32,32,3)

# def constructCifar10ZCA(trainset):
#     print("Constructing ZCA matrix for Cifar10")
#     X = trainset.train_data.reshape(-1,32*32*3)
#     cov = np.cov(X, rowvar=False)
#     # Singular Value Decomposition. X = U * np.diag(S) * V
#     U,S,V = np.linalg.svd(cov)
#         # U: [M x M] eigenvectors of sigma.
#         # S: [M x 1] eigenvalues of sigma.
#         # V: [M x M] transpose of U
#     # Whitening constant: prevents division by zero
#     epsilon = 1e-6
#     # ZCA Whitening matrix: U * Lambda * U'
#     ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
#     torch.save(ZCAMatrix.T, "ZCAtranspose.np")
#     return ZCAMatrix.T
    