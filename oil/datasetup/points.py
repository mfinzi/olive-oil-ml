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
from oil.utils.utils import Named, export
warnings.filterwarnings('ignore')

#ModelNet40 code adapted from 
#https://github.com/DylanWusee/pointconv_pytorch/blob/master/data_utils/ModelNetDataLoader.py

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    seg = []
    return (data, label, seg)

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


class RandomZrotation(nn.Module):
    def __init__(self,max_angle=np.pi):
        super().__init__()
        self.max_angle = max_angle
    def forward(self,x):
        # this presumes z axis is coordinate 2?
        # assumes x has shape B3N
        bs,c,n = x.shape; assert c==3
        angles = (2*torch.rand(bs)-1)*self.max_angle
        R = torch.zeros(bs,3,3)
        R[:,1,1] = 1
        R[:,0,0] = R[:,2,2] = angles.cos()
        R[:,0,2] = R[:,2,0] = angles.sin()
        R[:,2,0] *=-1
        return R.to(x.device)@x



@export
class ModelNet40(Dataset,metaclass=Named):
    num_classes=40
    def __init__(self,root_dir,train=True,transform=None):
        super().__init__()
        self.transform = torch.ToTensor() if transform is None else transform
        train_x,train_y,test_x,test_y = load_data(
            os.path.expanduser('~/datasets/ModelNet40/'),classification=True)
        self.coords = train_x if train else test_x
        # N x m x 3
        self.labels = train_y if train else test_y
        self.coords = self.coords.transpose((0,2,1)) # B x n x c -> B x c x n

    def __getitem__(self,index):
        return self.transform(self.coords[index]), self.labels[index]
    def __len__(self):
        return len(self.coords)
    def default_aug_layers(self):
        return RandomZrotation()

if __name__=='__main__':
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import cv2
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    i = 0
    a = load_data(os.path.expanduser('~/datasets/ModelNet40/'))
    def update_plot(e):
        global i
        if e.key == "right": i+=1
        elif e.key == "left": i-=1
        else:return
        ax.cla()
        d = a[0][i].T
        ax.scatter(d[0],d[2],d[1],c=d[1])
        #ax.contour3D(d[0],d[2],d[1],cmap='viridis',edgecolor='none')
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        fig.canvas.draw()
    fig.canvas.mpl_connect('key_press_event',update_plot)
    plt.show()