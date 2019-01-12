import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as ds
import torch.nn as nn
import numpy as np
from . import augLayers


class Named(type):
    def __str__(self):
        return self.__name__

class EasyIMGDataset(object,metaclass=Named):
    def __init__(self,*args,gan_normalize=False,download=True,**kwargs):
        transform = kwargs.pop('transform',None)
        if not transform: transform = self.default_transform(gan_normalize)
        super().__init__(*args,transform=transform,download=download,**kwargs)

    def default_transform(self,gan_normalize=False):
        if gan_normalize: 
            normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            normalize = transforms.Normalize(self.means, self.stds)
        transform = transforms.Compose([transforms.ToTensor(),normalize])
        return transform

    def compute_default_transform(self):
        raise NotImplementedError

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
    stds = (0.5, 0.5, 0.5)
    num_classes=10
    def default_aug_layers(self):
        return nn.Sequential(
        augLayers.RandomTranslate(4),
        augLayers.RandomHorizontalFlip(),
        )







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
    