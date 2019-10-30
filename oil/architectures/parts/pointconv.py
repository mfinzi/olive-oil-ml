"""
Utility function for PointConv
Originally from : https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/utils.py
Modify by Wenxuan Wu
Date: September 2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np
from sklearn.neighbors.kde import KernelDensity
from ...utils.utils import Expression,export,Named
from ..parts import conv2d

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def sample_and_group(npoint, nsample, xyz, points, density_scale = None):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, nsample, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    if N==S:
        new_xyz = xyz
    else:
        fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx) # 
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    # Added: remove effective coordconv, don't add coords to channels
    new_points = index_points(points, idx)
    #
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density

def subsample_and_neighbors(npoint,nsample,xyz):
    B, N, C = xyz.shape
    S = npoint
    if N==S:
        new_xyz = xyz
    else:
        fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx) # 
    neighbor_idx = knn_point(nsample, xyz, new_xyz)
    return new_xyz,neighbor_idx

def pthash(xyz):
    return hash(tuple(xyz.cpu().data.numpy().reshape(-1))+(xyz.device,))

class FarthestSubsample(nn.Module):
    def __init__(self,ds_frac=0.5,knn_channels=None):
        super().__init__()
        self.ds_frac = ds_frac
        self.subsample_lookup = {}
        self.knn_channels = knn_channels
    def forward(self,x,coords_only=False):
        # BCN representation assumed
        coords,values = x
        if self.ds_frac==1:
            if coords_only: return coords
            else: return x
        coords = coords.permute(0, 2, 1)
        values = values.permute(0, 2, 1)
        num_downsampled_points = int(np.round(coords.shape[1]*self.ds_frac))
        key = pthash(coords[:,:,:self.knn_channels])
        if key not in self.subsample_lookup:# or True:
            #print("Cache miss")
            self.subsample_lookup[key] = farthest_point_sample(coords, num_downsampled_points)
        fps_idx = self.subsample_lookup[key]
        new_coords = index_points(coords,fps_idx).permute(0, 2, 1)
        if coords_only: return new_coords
        new_values = index_points(values,fps_idx).permute(0, 2, 1)
        return new_coords,new_values

def subsample(npoint,nsample,xyz):
    B, N, C = xyz.shape
    S = npoint
    if N==S:
        new_xyz = xyz
    else:
        fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx) # 
    return new_xyz

def compute_density(xyz, bandwidth):
    '''
    xyz: input points position data, [B, N, C]
    '''
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    sqrdists = square_distance(xyz, xyz)
    gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
    xyz_density = gaussion_density.mean(dim = -1)

    return xyz_density

class DensityNet(nn.Module):
    def __init__(self, hidden_unit = [8, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList() 

        self.mlp_convs.append(nn.Conv1d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv1d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm1d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv1d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm1d(1))

    def forward(self, xyz_density):
        B, N = xyz_density.shape 
        density_scale = xyz_density.unsqueeze(1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale =  bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale) + 0.5
            else:
                density_scale = F.relu(density_scale)
        
        return density_scale

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)))

        return weights

import numbers

@export
class PointConvSetAbstraction(nn.Module):
    def __init__(self, ds_frac, nsample, in_channel, mlp, group_all,xyz_dim=3,knn_channels=None):
        super().__init__()
        cicm_co = 16
        self.knn_channels = knn_channels
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(xyz_dim, cicm_co)
        self.linear = nn.Linear(cicm_co * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.group_all = group_all

        self.neighbor_lookup = {}
        if isinstance(ds_frac,numbers.Number):
            self.subsample = FarthestSubsample(ds_frac,knn_channels=knn_channels)
        else:
            self.subsample = ds_frac

    def forward(self, inp):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        
        new_xyz = self.subsample(inp,coords_only=True)
        xyz, points = inp
        B = xyz.shape[0]
        xyz = xyz.permute(0, 2, 1)
        new_xyz = new_xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)
        num_ds_points = new_xyz.shape[1]
        key = pthash(xyz[:,:,:self.knn_channels])
        #print(xyz.shape,new_xyz.shape)
        #assert False
        if key not in self.neighbor_lookup:# or True:
            #print("Cache miss")
            self.neighbor_lookup[key] = knn_point(min(self.nsample,xyz.shape[1]),
                    xyz[:,:,:self.knn_channels], new_xyz[:,:,:self.knn_channels])
        neighbor_idx = self.neighbor_lookup[key]
        neighbor_xyz = index_points(xyz, neighbor_idx) # [B, npoint, nsample, C]
        B, N, C = xyz.shape
        neighbor_xyz_offsets = neighbor_xyz - new_xyz.view(B, num_ds_points, 1, C)
        new_points = index_points(points, neighbor_idx)
        #new_xyz, new_points, grouped_xyz_norm, _ = sample_and_group(self.npoint, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        grouped_xyz = neighbor_xyz_offsets.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2),
                        other = weights.permute(0, 3, 2, 1)).view(B, num_ds_points, -1)
        new_points = self.linear(new_points).permute(0,2,1)
        new_xyz = new_xyz.permute(0, 2, 1)
        return (new_xyz, new_points)

@export
class both(nn.Module):
    def __init__(self,module1,module2):
        super().__init__()
        self.module1 = module1
        self.module2 = module2
    def forward(self,inp):
        x,z = inp
        return self.module1(x),self.module2(z)

@export
class Pass(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.module = module
    def forward(self,x):
        c,y = x
        return c, self.module(y)

def PointConv(in_channels,out_channels,nbhd=9,ds_frac=1,bandwidth=0.1,xyz_dim=2,knn_channels=None):
    mlp_channels = [out_channels//4,out_channels//2,out_channels]
    # return PointConvDensitySetAbstraction(
    #         npoint=npoint, nsample=nbhd, in_channel=in_channels,
    #         mlp=mlp_channels, bandwidth = bandwidth, group_all=False,xyz_dim=2)
    return PointConvSetAbstraction(
            ds_frac=ds_frac, nsample=nbhd, in_channel=in_channels,
            mlp=mlp_channels, group_all=False,xyz_dim=xyz_dim,knn_channels=knn_channels)

@export
def pConvBNrelu(in_channels,out_channels,**kwargs):
    return nn.Sequential(
        PointConv(in_channels,out_channels,**kwargs),
        Pass(nn.BatchNorm1d(out_channels)),
        Pass(nn.ReLU())
    )

@export
class pBottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,nbhd=3.66**2,ds_frac=0.5,drop_rate=0,r=4,gn=False,**kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = (lambda c: nn.GroupNorm(c//16,c)) if gn else nn.BatchNorm1d
        self.pointconv = PointConv(in_channels//r,out_channels,nbhd=nbhd,ds_frac=ds_frac,**kwargs)
        self.net = nn.Sequential(
            Pass(norm_layer(in_channels)),
            Pass(nn.ReLU()),
            Pass(nn.Conv1d(in_channels,in_channels//r,1)),
            Pass(norm_layer(in_channels//r)),
            Pass(nn.ReLU()),
            self.pointconv,
            Pass(norm_layer(out_channels)),
            Pass(nn.ReLU()),
            Pass(nn.Conv1d(out_channels,out_channels,1)),
        )

    def forward(self,x):
        coords,values = x
        #print(values.shape)
        new_coords,new_values  = self.net(x)
        new_values[:,:self.in_channels] += self.pointconv.subsample(x)[1] # subsampled old values
        #print(shortcut.shape)
        #print(new_coords.shape,new_values.shape)
        return new_coords,new_values

class FarthestSubsample(nn.Module):
    def __init__(self,ds_frac=0.5,knn_channels=None):
        super().__init__()
        self.ds_frac = ds_frac
        self.subsample_lookup = {}
        self.knn_channels = knn_channels
    def forward(self,x,coords_only=False):
        # BCN representation assumed
        coords,values = x
        coords = coords.permute(0, 2, 1)
        values = values.permute(0, 2, 1)
        num_downsampled_points = int(np.round(coords.shape[1]*self.ds_frac))
        key = pthash(coords[:,:,:self.knn_channels])
        if key not in self.subsample_lookup:# or True:
            #print("Cache miss")
            self.subsample_lookup[key] = farthest_point_sample(coords, num_downsampled_points)
        fps_idx = self.subsample_lookup[key]
        new_coords = index_points(coords,fps_idx).permute(0, 2, 1)
        if coords_only: return new_coords
        new_values = index_points(values,fps_idx).permute(0, 2, 1)
        return new_coords,new_values

def imagelike_nn_downsample(x,coords_only=False):
    coords,values = x
    bs,c,N = values.shape
    h = w = int(np.sqrt(N))
    ds_coords = torch.nn.functional.interpolate(coords.view(bs,2,h,w),scale_factor=0.5)
    ds_values = torch.nn.functional.interpolate(values.view(bs,c,h,w),scale_factor=0.5)
    if coords_only: return ds_coords.view(bs,2,-1)
    return ds_coords.view(bs,2,-1), ds_values.view(bs,c,-1)



@export
class pResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,drop_rate=0,stride=1,nbhd=3**2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        ds = 1 if stride==1 else imagelike_nn_downsample
        self.net = nn.Sequential(
            Pass(nn.BatchNorm1d(in_channels)),
            Pass(nn.ReLU()),
            PointConv(in_channels,out_channels,nbhd=nbhd,ds_frac=1),
            Pass(nn.Dropout(p=drop_rate)),
            Pass(nn.BatchNorm1d(out_channels)),
            Pass(nn.ReLU()),
            PointConv(out_channels,out_channels,nbhd=nbhd,ds_frac=ds),
        )
        self.shortcut = nn.Sequential()
        if in_channels!=out_channels:
            self.shortcut.add_module('conv',Pass(nn.Conv1d(in_channels,out_channels,1)))
        if stride!=1:
            self.shortcut.add_module('ds',Expression(lambda a: imagelike_nn_downsample(a)))

    def forward(self,x):
        res_coords,res_values = self.net(x)
        skip_coords,skip_values = self.shortcut(x)
        return res_coords,res_values+skip_values

# @export
# class PointConvDensitySetAbstraction(nn.Module):
#     def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all,xyz_dim=3):
#         super(PointConvDensitySetAbstraction, self).__init__()
#         cicm_co = 16
#         self.npoint = npoint
#         self.nsample = nsample
#         self.mlp_convs = nn.ModuleList()
#         self.mlp_bns = nn.ModuleList()
#         last_channel = in_channel
#         for out_channel in mlp:
#             self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
#             self.mlp_bns.append(nn.BatchNorm2d(out_channel))
#             last_channel = out_channel
#         #ci*cm = co*16
#         self.weightnet = WeightNet(xyz_dim, cicm_co)
#         self.linear = nn.Linear(cicm_co * mlp[-1], mlp[-1])
#         self.bn_linear = nn.BatchNorm1d(mlp[-1])
#         self.densitynet = DensityNet()
#         self.group_all = group_all
#         self.bandwidth = bandwidth

#     def forward(self, inp):
#         """
#         Input:
#             xyz: input points position data, [B, C, N]
#             points: input points data, [B, D, N]
#         Return:
#             new_xyz: sampled points position data, [B, C, S]
#             new_points_concat: sample points feature data, [B, D', S]
#         """
#         xyz, points = inp
#         B = xyz.shape[0]
#         N = xyz.shape[2]
#         xyz = xyz.permute(0, 2, 1)
#         if points is not None:
#             points = points.permute(0, 2, 1)

#         xyz_density = compute_density(xyz, self.bandwidth)
#         #import ipdb; ipdb.set_trace()
#         density_scale = self.densitynet(xyz_density)

#         new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(self.npoint, self.nsample, xyz, points, density_scale.view(B, N, 1))
#         # new_xyz: sampled points position data, [B, npoint, C]
#         # new_points: sampled points data, [B, npoint, nsample, C+D]
#         new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
#         for i, conv in enumerate(self.mlp_convs):
#             bn = self.mlp_bns[i]
#             new_points =  F.relu(bn(conv(new_points)))

#         grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
#         grouped_xyz = grouped_xyz * grouped_density.permute(0, 3, 2, 1)
#         weights = self.weightnet(grouped_xyz)

#         new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint or N, -1)
#         new_points = self.linear(new_points)
#         new_points = self.bn_linear(new_points.permute(0, 2, 1))
#         new_points = F.relu(new_points)
#         new_xyz = new_xyz.permute(0, 2, 1)

#         return (new_xyz, new_points)