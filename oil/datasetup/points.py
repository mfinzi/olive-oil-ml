import torch, torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as ds
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import h5py
import os
from torch.utils.data import Dataset
from oil.datasetup import augLayers
from oil.utils.utils import Named, export, Expression
from oil.architectures.parts import FarthestSubsample
import torch_geometric
warnings.filterwarnings('ignore')

#ModelNet40 code adapted from 
#https://github.com/DylanWusee/pointconv_pytorch/blob/master/data_utils/ModelNetDataLoader.py

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

def _load_data_file(name):
    f = h5py.File(name)
    data = f["data"][:]
    label = f["label"][:]
    return data, label

def load_data(dir,classification = False):
    data_train0, label_train0,Seglabel_train0  = load_h5(dir + 'ply_data_train0.h5')
    data_train1, label_train1,Seglabel_train1 = load_h5(dir + 'ply_data_train1.h5')
    data_train2, label_train2,Seglabel_train2 = load_h5(dir + 'ply_data_train2.h5')
    data_train3, label_train3,Seglabel_train3 = load_h5(dir + 'ply_data_train3.h5')
    data_train4, label_train4,Seglabel_train4 = load_h5(dir + 'ply_data_train4.h5')
    data_test0, label_test0,Seglabel_test0 = load_h5(dir + 'ply_data_test0.h5')
    data_test1, label_test1,Seglabel_test1 = load_h5(dir + 'ply_data_test1.h5')
    train_data = np.concatenate([data_train0,data_train1,data_train2,data_train3,data_train4])
    train_label = np.concatenate([label_train0,label_train1,label_train2,label_train3,label_train4])
    train_Seglabel = np.concatenate([Seglabel_train0,Seglabel_train1,Seglabel_train2,Seglabel_train3,Seglabel_train4])
    test_data = np.concatenate([data_test0,data_test1])
    test_label = np.concatenate([label_test0,label_test1])
    test_Seglabel = np.concatenate([Seglabel_test0,Seglabel_test1])

    if classification:
        return train_data, train_label, test_data, test_label
    else:
        return train_data, train_Seglabel, test_data, test_Seglabel


@export
class ModelNet40(Dataset,metaclass=Named):
    ignored_index = -100
    class_weights = None
    balanced=False
    num_classes=40
    classes=['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
        'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot',
        'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor',
        'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase',
        'wardrobe', 'xbox']
    default_root_dir = '~/datasets/ModelNet40/'
    def __init__(self,root_dir=default_root_dir,train=True,transform=None,size=1024):
        super().__init__()
        #self.transform = torchvision.transforms.ToTensor() if transform is None else transform
        train_x,train_y,test_x,test_y = load_data(os.path.expanduser(root_dir),classification=True)
        self.coords = train_x if train else test_x
        # SWAP y and z so that z (gravity direction) is in component 3
        self.coords[...,2] += self.coords[...,1]
        self.coords[...,1] = self.coords[...,2]-self.coords[...,1]
        self.coords[...,2] -= self.coords[...,1]
        # N x m x 3
        self.labels = train_y if train else test_y
        self.coords_std = np.std(train_x,axis=(0,1))
        self.coords /= self.coords_std
        self.coords = self.coords.transpose((0,2,1)) # B x n x c -> B x c x n
        self.size=size
        #pt_coords = torch.from_numpy(self.coords)
        #self.coords = FarthestSubsample(ds_frac=size/2048)((pt_coords,pt_coords))[0].numpy()

    def __getitem__(self,index):
        return torch.from_numpy(self.coords[index]).float(), int(self.labels[index])
    def __len__(self):
        return len(self.labels)
    def default_aug_layers(self):
        subsample = Expression(lambda x: x[:,:,np.random.permutation(x.shape[-1])[:self.size]])
        return nn.Sequential(subsample,augLayers.RandomZrotation(),augLayers.GaussianNoise(.01))#,augLayers.PointcloudScale())#

@export
class MNISTSuperpixels(torch_geometric.datasets.MNISTSuperpixels,metaclass=Named):
    ignored_index = -100
    class_weights = None
    balanced = False
    num_classes = 10
    # def __init__(self,*args,**kwargs):
    #     super().__init__(*args,**kwargs)
    # coord scale is 0-25, std of unif [0-25] is 
    def __getitem__(self,index):
        datapoint = super().__getitem__(int(index))
        coords = (datapoint.pos.T-13.5)/5 # 2 x M array of coordinates
        bchannel = (datapoint.x.T-.1307)/0.3081 # 1 x M array of blackwhite info
        label = int(datapoint.y.item())
        return ((coords,bchannel),label)
    def default_aug_layers(self):
        return nn.Sequential()

from oil.utils.utils import FixedNumpySeed
import torchvision
@export
class RotMNIST(torchvision.datasets.MNIST,metaclass=Named):
    ignored_index = -100
    class_weights = None
    balanced = False
    num_classes = 10
    def __init__(self,*args,dataseed=0,**kwargs):
        super().__init__(*args,download=True,**kwargs)
        xy = (np.mgrid[:28,:28]-13.5)/5
        disk_cutout = xy[0]**2 +xy[1]**2 < 7
        self.img_coords = torch.from_numpy(xy[:,disk_cutout]).float()
        self.cutout_data = self.data[:,disk_cutout].unsqueeze(1)
        with FixedNumpySeed(dataseed):
            angles = torch.rand(len(self.data))*2*np.pi
        R = torch.zeros(len(self.data),2,2)
        R[:,0,0] = R[:,1,1] = angles.cos()
        R[:,0,1] = R[:,1,0] = angles.sin()
        R[:,1,0] *=-1
        self.img_coords = R@self.img_coords

    def __getitem__(self,index):
        index = int(index)
        pixel_vals = (self.cutout_data[index]-.1307)/0.3081
        return ((self.img_coords[index],pixel_vals),self.targets[index].item())
    def __len__(self):
        return len(self.data)
    def default_aug_layers(self):
        return nn.Sequential()

if __name__=='__main__':
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import cv2
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    i = 0
    # a = load_data(os.path.expanduser('~/datasets/ModelNet40/'))[0]
    # a[...,2] += a[...,1]
    # a[...,1] = a[...,2]-a[...,1]
    # a[...,2] -= a[...,1]
    D = ModelNet40()
    def update_plot(e):
        global i
        if e.key == "right": i+=1
        elif e.key == "left": i-=1
        else:return
        ax.cla()
        xyz,label = D[i]#.T
        x,y,z = xyz.numpy()*D.coords_std[:,None]
        # d[2] += d[1]
        # d[1] = d[2]-d[1]
        # d[2] -= d[1]
        ax.scatter(x,y,z,c=z)
        ax.text2D(0.05, 0.95, D.classes[label], transform=ax.transAxes)
        #ax.contour3D(d[0],d[2],d[1],cmap='viridis',edgecolor='none')
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event',update_plot)
    plt.show()