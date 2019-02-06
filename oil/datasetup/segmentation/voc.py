import torch, torchvision
#import torchvision.datasets as ds
import oil.datasetup.segmentation._replacement as ds # until torchvision update is pushed to conda
import oil.datasetup.segmentation.joint_transforms as tr
from oil.utils.utils import Named

class VOCSegmentation(ds.VOCSegmentation,metaclass=Named):

    def __init__(self,*args,download=True,image_set='train',**kwargs):
        super().__init__(*args,download=download,image_set=image_set,**kwargs)

    class_weights = None
    ignored_index = 255
    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)
    num_classes=21

    transforms_train = torchvision.transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=513, crop_size=513),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=means, std=stds),
            tr.ToTensor()])
    transforms_test = torchvision.transforms.Compose([
            tr.FixScaleCrop(crop_size=513),
            tr.Normalize(mean=means, std=stds),
            tr.ToTensor()])

    def __getitem__(self,index):
        img,target =super().__getitem__(index)
        if self.image_set=='train':
            img,target = self.transforms_train((img,target))
        elif self.image_set=='val':
            img,target = self.transforms_test((img,target))
        else: raise NotImplementedError("Unknown split {}".format(self.image_set))
        return img,target.long()