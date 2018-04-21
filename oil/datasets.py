import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as ds
import torch.nn as nn

from oil.networkparts import layer13
import oil.augLayers as augLayers




def CIFAR10(aug=True, ZCA=False):
    if ZCA:
        transform_dev = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.0904,.0868,.0468), (1,1,1))])
    else:
        transform_dev = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (.247,.243,.261))])
    if aug:
        transform_train = transforms.Compose(
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transform_dev])
    else: transform_train = transform_dev
    pathToDataset = '/scratch/datasets/cifar10/'
    trainset = ds.CIFAR10(pathToDataset, download=True, transform=transform_train)
    devset = ds.CIFAR10(pathToDataset, train=False, download=True, transform=transform_dev)
    testset = None
    datasets = (trainset, devset, testset)
    if ZCA:
        ZCAt_mat = torch.load("ZCAtrans_np")
        trainset.train_data = np.dot(trainset.train_data.reshape(-1,32*32*3), ZCAt_mat).reshape(-1,32,32,3)
        devset.test_data = np.dot(devset.test_data.reshape(-1,32*32*3), ZCAt_mat).reshape(-1,32,32,3)
        assert testset == None
    return datasets


def C10augLayers():
    layers = nn.Sequential(
        augLayers.RandomTranslate(4),
        augLayers.RandomHorizontalFlip(),
        )
    return layers



    # def CIFAR10(aug=True, ZCA=False):
    # img_size = 32
    # transform_base = transforms.Compose(
    #     [transforms.Resize(img_size),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (1,1,1))])
    # if ZCA:
    #     ZCAt_mat = torch.load("ZCAtranspose")
    #     ZCA_transform = augLayers.LinearTransformationGPU(ZCAt_mat)
    #     transform_dev = transforms.Compose([transform_base, ZCA_transform])
    # else:
    #     transform_dev = transforms.Compose([transform_base,
    #         transforms.Normalize((0,0,0), (.247,.243,.261))])
    #     # previously (0.2023, 0.1994, 0.2010)
    # if aug:
    #     transform_train = transforms.Compose(
    #         [transforms.RandomCrop(32, padding=4),
    #         transforms.RandomHorizontalFlip(),
    #         transform_dev])
    # else:
    #     transform_train = transform_dev
    # pathToDataset = '/scratch/datasets/cifar10/'
    # trainset = ds.CIFAR10(pathToDataset, download=True, transform=transform_train)
    # devset = ds.CIFAR10(pathToDataset, train=False, download=True, transform=transform_dev)
    # testset = None
    # datasets = (trainset, devset, testset)
    # return datasets