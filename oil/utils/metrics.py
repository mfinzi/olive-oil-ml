import torch
import torch.nn as nn
import numpy as np
import os
import subprocess
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from scipy.linalg import norm,sqrtm
from ..utils.utils import Eval, Expression
#from torch.nn.functional import adaptive_avg_pool2d
#from .pytorch-fid.fid_score import calculate_frechet_distance
#from .pytorch-fid.inception import InceptionV3

# GAN Metrics

# TODO: cache logits for existing datasets
#       should be possible if we can serialize dataloaders
def get_inception():
    """ grabs the pytorch pretrained inception_v3 with resized inputs """
    inception = inception_v3(pretrained=True,transform_input=False)
    upsample = Expression(lambda x: nn.functional.interpolate(x,size=(299,299),mode='bilinear'))
    model = nn.Sequential(upsample,inception).cuda().eval()
    return model

def get_logits(model,loader):
    """ Extracts logits from a model, dataloader returns a numpy array of size (N, K)
        where K is the number of classes """
    with torch.no_grad(), Eval(model): 
        model_logits = lambda mb: model(mb).cpu().data.numpy()
        logits = np.concatenate([model_logits(minibatch) for minibatch in loader],axis=0)
    return logits

def FID_from_logits(logits1,logits2):
    """Computes the FID between logits1 and logits2
        Inputs: [logits1 (N,C)] [logits2 (N,C)] """
    mu1 = np.mean(logits1,axis=0)
    mu2 = np.mean(logits2,axis=0)
    sigma1 = np.cov(logits1, rowvar=False)
    sigma2 = np.cov(logits2, rowvar=False)

    tr = np.trace(sigma1 + sigma2 - 2*sqrtm(sigma1@sigma2))
    distance = norm(mu1-mu2)**2 + tr
    return distance

def IS_from_logits(logits):
    """ Computes the Inception score (IS) from logits of the dataset of size N with C classes.
        Inputs: [logits (N,C)], Outputs: [IS (scalar)]"""
    # E_z[KL(Pyz||Py)] = \mean_z [\sum_y (Pyz log(Pyz) - Pyz log(Py))]
    Pyz = np.exp(logits).transpose() # Take softmax (up to a normalization constant)
    Pyz /= Pyz.sum(0)[None,:]        # divide by normalization constant
    Py = np.broadcast_to(Pyz.mean(-1)[:,None],Pyz.shape)        # Average over z
    logIS = entropy(Pyz,Py).mean()   # Average over z
    return np.exp(logIS)

cachedLogits = {} 
def FID(loader1,loader2):
    """ Computes the Frechet Inception Distance  (FID) between the two image dataloaders
        using pytorch pretrained inception_v3. Requires >2048 imgs for comparison
        Dataloader should be an iterable of minibatched images, assumed to already
        be normalized with mean 0, std 1 (per color)
        """
    model = get_inception()
    logits1 = get_logits(model,loader1)
    if loader2 not in cachedLogits:
        cachedLogits[loader2] = get_logits(model,loader2)
    logits2 = cachedLogits[loader2]
    return FID_from_logits(logits1,logits2)
    
def IS(loader):
    """Computes the Inception score of a dataloader using pytorch pretrained inception_v3"""
    model = get_inception()
    logits = get_logits(model,loader)
    return IS_from_logits(logits)

   
def FID_and_IS(loader1,loader2):
    """Computes FID and IS score for loader1 against target loader2 """
    model = get_inception()
    logits1 = get_logits(model,loader1)
    if loader2 not in cachedLogits:
        cachedLogits[loader2] = get_logits(model,loader2)
    logits2 = cachedLogits[loader2]
    return FID_from_logits(logits1,logits2),IS_from_logits(logits1)

#TODO: Implement Kernel Inception Distance (KID) from (https://openreview.net/pdf?id=r1lUOzWCW)


def get_official_FID(loader,dataset='cifar10'):
    #TODO: make function not ass and check that it still works
    dir = os.path.expanduser("~/olive-oil-ml/oil/utils/")
    path = dir+"temp"
    loader.write_imgs(path)
    if dataset not in ('cifar10',):
        raise NotImplementedError
    score = subprocess.check_output(dir+"TTUR/fid.py "+dir+"fid_stats_{}_train.npz {}.npy"
                .format(dataset,path),shell=True)
    return score


# Semantic Segmentation Metrics
# Adapted from https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
def confusion_from_logits(logits,y_gt):
    bs, num_classes, _, _ = logits.shape
    pred_image = logits.max(1)[1].type_as(y_gt).cpu().data.numpy()
    #print(pred_image)
    gt_image = y_gt.cpu().data.numpy()
    return confusion_matrix(pred_image,gt_image,num_classes)

def confusion_matrix(pred_image,gt_image,num_classes):
    """Computes the confusion matrix from two numpy class images (integer values)
        ignoring classes that are negative"""
    mask = (gt_image >= 0) & (gt_image < num_classes)
    #print(gt_image[mask])
    label = num_classes * gt_image[mask].astype(int) + pred_image[mask]
    count = np.bincount(label, minlength=num_classes**2)
    confusion_matrix = count.reshape(num_classes, num_classes)
    return confusion_matrix # confusing shape, maybe transpose

def meanIoU(confusion_matrix):
    MIoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    MIoU = np.nanmean(MIoU)
    return MIoU

def freqIoU(confusion_matrix):
    freq = np.sum(confusion_matrix, axis=1) / np.sum(confusion_matrix)
    iu = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def pixelAcc(confusion_matrix):
    return np.diag(confusion_matrix).sum() / confusion_matrix.sum()

def meanAcc(confusion_matrix):
    return np.nanmean(np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1))


# def boundary_mIoU(confusion_matrix,epsilon)