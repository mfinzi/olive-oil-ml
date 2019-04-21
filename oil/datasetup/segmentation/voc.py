import torch, torchvision
#import torchvision.datasets as ds
import oil.datasetup.segmentation._replacement as ds # until torchvision update is pushed to conda
import oil.datasetup.segmentation.joint_transforms as tr
from oil.utils.utils import Named
import numpy as np

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

    label_colors = np.array([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128], [192,192,192]])
    label_ids = list(range(num_classes))+[255]

    def decode_segmap(self,labels):
        rgbimg = labels.new(labels.shape+(3,)).float()
        for id,color in zip(self.label_ids,self.label_colors):
            rgbimg[labels==id] = torch.from_numpy(color).float()/255
        return rgbimg.permute((0,3,1,2))

    def __getitem__(self,index):
        img,target =super().__getitem__(index)
        if self.image_set=='train':
            img,target = self.transforms_train((img,target))
        elif self.image_set=='val':
            img,target = self.transforms_test((img,target))
        else: raise NotImplementedError("Unknown split {}".format(self.image_set))
        return img,target.long()



def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb