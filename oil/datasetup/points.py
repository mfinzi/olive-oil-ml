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
    def __init__(self,root_dir,train=True,transform=None,size=1024):
        super().__init__()
        #self.transform = torchvision.transforms.ToTensor() if transform is None else transform
        train_x,train_y,test_x,test_y = load_data(
            os.path.expanduser('~/datasets/ModelNet40/'),classification=True)
        self.coords = train_x if train else test_x
        # N x m x 3
        self.labels = train_y if train else test_y
        self.coords /= np.std(train_x,axis=(0,1))
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
        return nn.Sequential(subsample,augLayers.RandomZrotation(),augLayers.GaussianNoise(.02))#,augLayers.PointcloudScale())#


class MNISTSuperpixels(torch_geometric.datasets.MNISTSuperpixels):
    ignored_index = -100
    class_weights = None
    balanced = True
    num_classes = 10
    # def __init__(self,*args,**kwargs):
    #     super().__init__(*args,**kwargs)

    def __getitem__(self,index):
        datapoint = super().__getitem__(index)
        coords = datapoint.pos.T # 2 x M array of coordinates
        bchannel = datapoint.x.T # 1 x M array of blackwhite info
        label = datapoint.y
        return ((coords,bchannel),label)



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