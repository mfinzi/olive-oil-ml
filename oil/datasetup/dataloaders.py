import numpy as np
from ..utils.utils import FixedNumpySeed
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler, SubsetRandomSampler


# def getUnlabLoader(trainset, ul_BS, **kwargs):
#     """ Returns a dataloader for the full dataset, with cyclic reshuffling """
#     indices = np.arange(len(trainset))
#     unlabSampler = ShuffleCycleSubsetSampler(indices)
#     unlabLoader = DataLoader(trainset,sampler=unlabSampler,batch_size=ul_BS,**kwargs)
#     return unlabLoader

def getLabLoader(trainset, lab_BS, amnt_lab=1, amnt_dev=0, dataseed=0, balanced=True,**kwargs):
    """ returns a dataloader of class balanced subset of the full dataset,
        and a (possibly empty) dataloader reserved for devset
        amntLabeled and amntDev can be a fraction or an integer.
        If fraction amntLabeled specifies fraction of entire dataset to
        use as labeled, whereas fraction amntDev is fraction of labeled
        dataset to reserve as a devset  """
    numLabeled = amnt_lab
    if amnt_lab <= 1: 
        numLabeled *= len(trainset)
    numDev = amnt_dev
    if amnt_dev <= 1:
        numDev *= numLabeled
    with FixedNumpySeed(dataseed):
        get_indices = classBalancedSampleIndices if (balanced and trainset.balanced) else randomSampleIndices
        labIndices, devIndices = get_indices(trainset, int(numLabeled), int(numDev))

    labSampler = SubsetRandomSampler(labIndices)
    labLoader = DataLoader(trainset,sampler=labSampler,batch_size=lab_BS,**kwargs)
    if numLabeled == 0: labLoader = EmptyLoader()

    devSampler = SequentialSubsetSampler(devIndices) # No shuffling on dev
    devLoader = DataLoader(trainset,sampler=devSampler,batch_size=lab_BS)
    return labLoader, devLoader

def classBalancedSampleIndices(trainset, numLabeled, numDev):
    """ Generates a subset of indices of y (of size numLabeled) so that
        each class is equally represented """
    y = np.array([target for img,target in trainset])
    uniqueVals = np.unique(y)
    numDev = (numDev // len(uniqueVals))*len(uniqueVals)
    numLabeled = ((numLabeled-numDev)// len(uniqueVals))*len(uniqueVals)
    
    classIndices = [np.where(y==val) for val in uniqueVals]
    devIndices = np.empty(numDev, dtype=np.int64)
    labIndices = np.empty(numLabeled, dtype=np.int64)
    
    dev_m = numDev // len(uniqueVals)
    lab_m = numLabeled // len(uniqueVals); assert lab_m>0, "Note: dev is subtracted from train"
    total_m = lab_m + dev_m
    for i in range(len(uniqueVals)):
        sampledclassIndices = np.random.choice(classIndices[i][0],total_m,replace=False)
        labIndices[i*lab_m:i*lab_m+lab_m] = sampledclassIndices[:lab_m]
        devIndices[i*dev_m:i*dev_m+dev_m] = sampledclassIndices[lab_m:]
        
    print("Creating Train, Dev split \
        with {} Train and {} Dev".format(numLabeled, numDev))
    return labIndices, devIndices

def randomSampleIndices(trainset,numLabeled,numDev):
    numLabeled = (numLabeled-numDev)
    indices = np.random.permutation(len(trainset))
    return indices[:numLabeled],indices[numLabeled:numLabeled+numDev]

#TODO: Needs some rework to function properly in the semisupervised case

# class ShuffleCycleSubsetSampler(Sampler):
#     """A cycle version of SubsetRandomSampler with
#         reordering on calls to __iter__
#         contains current permutation & index as a state """
#     def __init__(self, indices):
#         self.indices = indices
    
#     def __iter__(self):
#         return self._gen()

#     def _gen(self):
#         i = len(self.indices)
#         while True:
#             if i >= len(self.indices):
#                 perm = np.random.permutation(self.indices)
#                 i=0
#             yield perm[i]
#             i+=1
    
#     def __len__(self):
#         return len(self.indices)

class SequentialSubsetSampler(Sampler):
    """Samples sequentially from specified indices, does not cycle """
    def __init__(self, indices):
        self.indices = indices
    def __iter__(self):
        return iter(self.indices)
    def __len__(self):
        return len(self.indices)

class EmptyLoader(object):
    """A dataloader that loads None tuples, with zero length for convenience"""
    def __next__(self):
        return (None,None)
    def __len__(self):
        return 0
    def __iter__(self):
        return self


# def getUandLloaders(trainset, amntLabeled, lab_BS, ul_BS, **kwargs):
#     """ Returns two cycling dataloaders where the first one only operates on a subset
#         of the dataset. AmntLabeled can either be a fraction or an integer """
#     numLabeled = amntLabeled
#     if amntLabeled <= 1: 
#         numLabeled *= len(trainset)
    
#     indices = np.random.permutation(len(trainset))
#     labIndices = indices[:numLabeled]

#     labSampler = ShuffleCycleSubsetSampler(labIndices)
#     labLoader = DataLoader(trainset,sampler=labSampler,batch_size=lab_BS,**kwargs)
#     if amntLabeled == 0: labLoader = EmptyLoader()

#     # Includes the labeled samples in the unlabeled data
#     unlabSampler = ShuffleCycleSubsetSampler(indices)
#     unlabLoader = DataLoader(trainset,sampler=unlabSampler,batch_size=ul_BS,**kwargs)
        
#     return unlabLoader, labLoader

# def getLoadersBalanced(trainset, amntLabeled, lab_BS, ul_BS, **kwargs):
#     """ Variant of getUandLloaders"""
#     numLabeled = amntLabeled
#     if amntLabeled <= 1: 
#         numLabeled *= len(trainset)
    
#     indices = np.random.permutation(len(trainset))
#     labIndices = classBalancedSampleIndices(trainset, numLabeled)

#     labSampler = ShuffleCycleSubsetSampler(labIndices)
#     labLoader = DataLoader(trainset,sampler=labSampler,batch_size=lab_BS,**kwargs)
#     if amntLabeled == 0: labLoader = EmptyLoader()

#     # Includes the labeled samples in the unlabeled data
#     unlabSampler = ShuffleCycleSubsetSampler(indices)
#     unlabLoader = DataLoader(trainset,sampler=unlabSampler,batch_size=ul_BS,**kwargs)
        
#     return unlabLoader, labLoader
