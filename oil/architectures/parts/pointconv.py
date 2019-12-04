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
import string
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

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
    if torch.any(group_idx>10000):
        print("greater than 10k :(")
        print(xyz.shape)
        print(new_xyz.shape)
        print(xyz[0])
        print(new_xyz[0])
        raise Exception
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

@export
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
        #key = pthash(coords[:,:,:self.knn_channels])
        #if key not in self.subsample_lookup:# or True:
            #print("Cache miss")
        #    self.subsample_lookup[key] = farthest_point_sample(coords, num_downsampled_points)
        fps_idx = farthest_point_sample(coords[:,:,:self.knn_channels], num_downsampled_points)#self.subsample_lookup[key]
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

    def __init__(self, in_channel, out_channel, hidden_unit = (8, 8)):
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

class Group(object,metaclass=Named):
    def embed(self):

        """ Compute a euclidean embedding of group element g that can be processed with
            a neural network"""
        raise NotImplementedError

    def extract_embedded_group_elem(self,p1,p2):
        """ Computes an embedded form of the group element that maps p1->p2
            ie. g p1 = p2. p1 should have shape (*,c) and p2 shape (*,c)"""
        raise NotImplementedError

    def sample_origin_stabilizer(self,group_elems):
        """ Samples a stabilizer of the origin for each input, apply it
            to the input group elements, and concatenate an embedding of it
            to the output."""
        raise NotImplementedError

class LieGroup(Group):
    @staticmethod
    def exp(A):
        raise NotImplementedError
    @staticmethod
    def log(A):
        raise NotImplementedError
    # @staticmethod
    # def embed(g):
    #     logg = g.log(g)
    #     return logg.view(*logg.shape[:-2],-1)

class GroupElem(object):
    def __matmul__(self,g):
        raise NotImplementedError

class T(Group):
    def __init__(self,dim):
        self.embed_dim=dim
    @staticmethod
    def extract_embedded_group_elem(p1,p2):
        return p1-p2, 0, 0
    @staticmethod
    def sample_origin_stabilizer(group_elems):
        """ Translation group has no stabilizers"""
        return group_elems


# Hodge star on R3
def cross_matrix(k):
    """Application of hodge star on R3, mapping Λ^1 R3 -> Λ^2 R3"""
    K = torch.zeros(*k.shape,3).to(k.device)
    K[...,0,1] = -k[...,2]
    K[...,0,2] = k[...,1]
    K[...,1,0] = k[...,2]
    K[...,1,2] = -k[...,0]
    K[...,2,0] = -k[...,1]
    K[...,2,1] = k[...,0]
    return K

def uncross_matrix(K):
    """Application of hodge star on R3, mapping Λ^2 R3 -> Λ^1 R3"""
    k = torch.zeros(*K.shape[:-1]).to(K.device)
    k[...,0] = K[...,2,1]
    k[...,1] = K[...,0,2]
    k[...,2] = K[...,1,0]
    return k
@export
class SO3(LieGroup):
    """ Use the rodriguez formula representation. I could just use
        the lie algebra representation, but this has the problem of not
        dealing with wraparounds well: \theta = 2pi,0."""
    embed_dim = 3
    @staticmethod
    def sample_origin_stabilizer(deltas):
        #k = torch.randn_like(deltas) # (bs,n,nbhd,3)
        #k = torch.randn(*deltas.shape[:-2],1,3).to(deltas.device)
        #k /= k.norm(dim=-1,keepdim=True)
        k = torch.zeros_like(deltas)
        k[...,2]=1.
        #thetas = (torch.rand(*deltas.shape[:-1],1,1).to(deltas.device)*2-1)*np.pi
        thetas = (torch.rand(*deltas.shape[:-2],1,1,1).to(deltas.device)*2-1)*np.pi
        K = cross_matrix(k)
        R = SO3.exp(K,thetas)
        # Embed rotated deltas, and the rotation
        embedding = torch.zeros(*deltas.shape[:-1],3+SO3.embed_dim).to(deltas.device)
        embedding[...,:3] = (R@deltas.unsqueeze(-1)).squeeze(-1)
        embedding[...,3:] = .1*k*thetas.squeeze(-1)#SO3.embed(R)
        return embedding
    @staticmethod
    def embed(R):
        return uncross_matrix(SO3.log(R))
    @staticmethod
    def exp(K,theta=None):
        """ Rodriguez's formula"""
        if theta is None:
            theta = torch.sqrt((K*K).sum(-2,keepdim=True).sum(-1,keepdim=True)/2)
            K = K/theta.clamp(min=1e-5)
        sin,cos = theta.sin(),theta.cos()
        I = torch.eye(3).to(K.device)
        Rs = I + sin*K + (1-cos)*(K@K)
        return Rs
    @staticmethod
    def log(R):
        otherdims = len(R.shape)-2
        placeholders = string.ascii_lowercase[1:otherdims+1]
        trR = torch.einsum(f'{placeholders}aa -> {placeholders}',R)
        theta = torch.acos((trR-1)/2)
        pm = theta.sign()
        theta *= pm
        logR = theta[...,None,None]*(R-R.transpose(-1,-2))/(2*theta.sin().clamp(min=1e-5)[...,None,None])
        return logR
    @staticmethod
    def extract_group_elem(p1,p2):
        normed_p1 = p1/p1.norm(dim=-1,keepdim=True)
        normed_p2 = p2/p2.norm(dim=-1,keepdim=True)
        # p1 cross p2 gives k
        orthogonal_vector = (cross_matrix(normed_p1)@normed_p2.unsqueeze(-1)).squeeze()
        angle = torch.acos((normed_p1*normed_p2).sum(-1).sqrt())[...,None,None]
        R = SO3.exp(K,angle)
        return R, R@p1, p2#TODO: check that it is not negative of the angle

@export
class SO2(LieGroup):
    embed_dim = 2
    @staticmethod
    def sample_origin_stabilizer(deltas):
        thetas = (torch.rand(*deltas.shape[:-2],1).to(deltas.device)*2-1)*np.pi
        R = torch.zeros(*deltas.shape,SO2.embed_dim).to(deltas.device)
        sin,cos = thetas.sin(),thetas.cos()
        R[...,0,0] = cos
        R[...,1,1] = cos
        R[...,0,1] = -sin
        R[...,1,0] = sin
        embedding = torch.zeros(*deltas.shape[:-1],2+SO2.embed_dim).to(deltas.device)
        embedding[...,:2] = (R@deltas.unsqueeze(-1)).squeeze(-1)
        embedding[...,2] = .1*cos
        embedding[...,3] = .1*sin
        return embedding

@export
class SE2(LieGroup):
    embed_dim = 9
    act_dim = 3
    @staticmethod
    def log(g):
        sin = g[...,1,0]
        cos = g[...,0,0]
        theta = torch.atan2(sin,cos)
        vxy = g[...,:2,2]
        Vinv = torch.zeros_like(g[...,:2,:2])
        Vinv[...,0,1] = theta/2
        Vinv[...,1,0] = -theta/2
        sincterm = 1-theta*theta/4 # use taylor series expansion for numeric stability
        notailer = (theta.abs()>1e-3) # if theta is large enough we can use exact
        sincterm[notailer] = (.5*theta*sin/(1-cos))[notailer]
        Vinv[...,0,0] = sincterm
        Vinv[...,1,1] = sincterm
        p = (Vinv@vxy.unsqueeze(-1)).squeeze(-1)
        a = torch.zeros_like(g)
        a[...,:2,2] = p
        a[...,0,1] = theta
        a[...,1,0] = -theta
        return a
    @staticmethod
    def exp(a):
        g = torch.zeros_like(a)
        theta = a[...,0,1]
        tx = a[...,0,2]
        ty = a[...,1,2]
        assert torch.allclose(theta,-a[...,1,0]), "element not in lie algebra?"
        assert torch.allclose(a[...,2,2],0*a[...,0,0]), "element not in lie algebra?"
        assert torch.allclose(a[...,1,1],0*a[...,0,0]), "element not in lie algebra?"
        assert torch.allclose(a[...,0,0],0*a[...,0,0]), "element not in lie algebra?"
        sin = theta.sin()
        cos = theta.cos()
        sinc = 1-theta*theta/6 + (1/120)*theta**4 # use taylor series expansion for numeric stability
        notailer = (theta.abs()>1e-3) # if theta is large enough we can use exact
        sinc[notailer] = (sin/theta)[notailer]
        cosc = -theta/2 + (1/24)*theta**3 # use taylor series expansion for numeric stability
        cosc[notailer] = ((1-cos)/theta)[notailer] # if theta is large enough we can use exact

        g[...,0,0] = cos
        g[...,1,1] = cos
        g[...,0,1] = -sin
        g[...,1,0] = sin
        g[...,2,2] = 1
        g[...,0,2] = sinc*tx-cosc*ty
        g[...,1,2] = cosc*tx+sinc*ty
        return g

    @staticmethod
    def lifted_samples(p,num_samples):
        """assumes p has shape (bs,2)"""
        bs,d = p.shape
        # Sample stabilizer of the origin
        thetas = (torch.rand(bs,num_samples).to(p.device)*2-1)*np.pi
        R = torch.zeros(bs,num_samples,d+1,d+1).to(p.device)
        sin,cos = thetas.sin(),thetas.cos()
        R[...,0,0] = cos
        R[...,1,1] = cos
        R[...,0,1] = -sin
        R[...,1,0] = sin
        R[...,2,2] = 1
        # Get T(p)
        T = torch.zeros_like(R)
        T[...,0,0]=1
        T[...,1,1]=1
        T[...,2,2]=1
        T[...,:2,2] = p[:,None,:]
        return SE2.log(R@T)


@export
class RGBscale(LieGroup):
    embed_dim=1
    @staticmethod
    def sample_origin_stabilizer(deltas):
        logr = torch.randn(*deltas.shape[:-2],1).to(deltas.device)
        embedding = torch.zeros(*deltas.shape[:-1],5+1).to(deltas.device)
        embedding[...,:2] = torch.exp(logr)*deltas
        embedding[...,2] = logr
        return embedding

@export
class Trivial(LieGroup):
    embed_dim = 0
    @staticmethod
    def sample_origin_stabilizer(deltas):
        return deltas

@export
class Coordinates(nn.Module,metaclass=Named):
    __name__ = "Coordinates"
    def __init__(self):
        super().__init__()
        self.embed_dim=0
    def forward(self,x):
        return x
#     def __str__(self):
#         return str(type(self))
#     def __repr__(self):
#         return repr(type(self))
# C = Coordinates()
@export
class LogPolar(Coordinates):
    def __init__(self,include_xy=False):
        super().__init__()
        self.include_xy = include_xy
        self.embed_dim = (2 if include_xy else 0)

    def forward(self,xy):
        r = xy.norm(dim=-1).unsqueeze(-1)
        theta = torch.atan2(xy[...,1],xy[...,0]).unsqueeze(-1)
        features = (r.log(),theta)
        if self.include_xy: features += (xy,)
        return torch.cat(features,dim=-1)

@export
class LogCylindrical(Coordinates):
    def __init__(self,include_xy=False):
        super().__init__()
        self.include_xy = include_xy
        self.embed_dim = (3 if include_xy else 0)

    def forward(self,xy):
        r = xy[...,:2].norm(dim=-1).unsqueeze(-1)
        theta = torch.atan2(xy[...,1],xy[...,0]).unsqueeze(-1)
        z = xy[...,2].unsqueeze(-1)
        features = (r.log(),theta,z)
        if self.include_xy: features += (xy,)
        return torch.cat(features,dim=-1)

@export
class LearnableCoordmap(Coordinates):
    def __init__(self,outdim=2,indim=2):
        super().__init__()
        self.embed_dim = outdim
        self.net = nn.Sequential(
            nn.Linear(indim,24),
            nn.ReLU(),
            nn.Linear(24,24),
            nn.ReLU(),
            nn.Linear(24,outdim)
        )
    def forward(self,xy):
        return torch.cat((xy,self.net(xy)),dim=-1)

class PointConvBase(nn.Module):
    def __init__(self,chin,chout,nbhd=32,xyz_dim=3,knn_channels=None):
        super().__init__()
        self.chin = chin
        self.cicm_co = 16
        self.xyz_dim = xyz_dim
        self.knn_channels = knn_channels
        self.nbhd = nbhd
        self.weightnet = WeightNet(xyz_dim, self.cicm_co,hidden_unit = (32, 32))
        self.linear = nn.Linear(self.cicm_co * chin, chout)

    def extract_neighborhood(self,inp_xyz,inp_vals,query_xyz):
        neighbor_idx = knn_point(min(self.nbhd,inp_xyz.shape[1]),
                    inp_xyz[:,:,:self.knn_channels], query_xyz[:,:,:self.knn_channels])#self.neighbor_lookup[key]
        nidx = neighbor_idx.cpu().data.numpy()
        neighbor_xyz = index_points(inp_xyz, neighbor_idx) # [B, npoint, nsample, C]
        nidx2 = neighbor_idx.cpu().data.numpy()
        neighbor_values = index_points(inp_vals, neighbor_idx)
        return neighbor_xyz, neighbor_values # (bs,n,nbhd,c)

    def compute_deltas(self,output_xyz,neighbor_xyz):
        return neighbor_xyz - output_xyz.unsqueeze(2)

    def point_convolve(self,embedded_group_elems,nbhd_vals):
        """ embedded_group_elems: (bs,n,nbhd,gc)
            vals: (bs,n,nbhd,c)"""
        bs,n,nbhd,c = nbhd_vals.shape
        # has shape (bs,n,nbhd,8)
        penult_kernel_weights = self.weightnet(embedded_group_elems.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # (bs,n,nbhd,c) -> (bs,n,c,nbhd)                                              #should be c*8
        partial_convolved_vals = torch.matmul(nbhd_vals.permute(0, 1, 3, 2),penult_kernel_weights).view(bs, n, -1) 
        convolved_vals = self.linear(partial_convolved_vals)/min(n,nbhd)
        return convolved_vals

    def get_embedded_group_elems(self,nbhd_xyz,output_xyz):
        return nbhd_xyz - output_xyz
    def forward(self,inp,query_xyz):
        """inputs and outputs [xyz (bs,n,3)], [vals (bs,n,c)]"""
        #xyz (bs,n,c), new_xyz (bs,m,c) neighbor_xyz (bs,m,nbhd,c)
        inp_xyz,inp_vals = inp
        nbhd_xyz,nbhd_vals = self.extract_neighborhood(inp_xyz,inp_vals,query_xyz)
        deltas = self.get_embedded_group_elems(nbhd_xyz,query_xyz.unsqueeze(2))
        convolved_vals = self.point_convolve(deltas,nbhd_vals)
        return convolved_vals

class GroupPointConv(PointConvBase):
    def __init__(self,*args,group=Trivial,**kwargs):
        super().__init__(*args,**kwargs)
        self.group = group
        self.weightnet = WeightNet(self.xyz_dim+self.group.embed_dim, self.cicm_co,hidden_unit = (32, 32))
    def extract_neighborhood(self,inp_xyz,inp_vals,query_xyz):
        raise NotImplementedError
    def get_embedded_group_elems(self,nbhd_xyz,output_xyz):
        G = self.group
        output_A = output_xyz.view(*output_xyz.shape[:-1],G.act_dim,G.act_dim)
        nbhd_A = nbhd_xyz.view(*nbhd_xyz.shape[:-1],G.act_dim,G.act_dim)
        commutator = output_A@nbhd_A - nbhd_A@output_A 
        embedded_group_elems = output_xyz - nbhd_xyz -0.5*commutator.view(*nbhd_xyz.shape)
        return embedded_group_elems

@export
class LearnedSubsample(nn.Module):
    def __init__(self,ds_frac=0.5,nbhd=24,knn_channels=None,xyz_dim=3,chin=64,**kwargs):
        super().__init__()
        self.ds_frac = ds_frac
        self.knn_channels = knn_channels
        self.mapping = PointConvBase(chin,xyz_dim,nbhd,xyz_dim,knn_channels,**kwargs)
    def forward(self,x,coords_only=False):
        # BCN representation assumed
        coords,values = x
        if self.ds_frac==1:
            if coords_only: return coords
            else: return x
        coords = coords
        values = values
        num_downsampled_points = int(np.round(coords.shape[1]*self.ds_frac))
        #key = pthash(coords[:,:,:self.knn_channels])
        #if key not in self.subsample_lookup:# or True:
            #print("Cache miss")
        #    self.subsample_lookup[key] = farthest_point_sample(coords, num_downsampled_points)
        fps_idx = farthest_point_sample(coords, num_downsampled_points)#self.subsample_lookup[key]

        new_coords = index_points(coords,fps_idx)
        new_values = index_points(values,fps_idx)
        self.inp_coords = new_coords[0].cpu().data.numpy()
        #print(new_values.shape,new_coords.shape)
        offset = .03*self.mapping(x,new_coords)
        self.offset = offset[0].cpu().data.numpy()
        new_coords = new_coords + offset
        self.out_coords = new_coords[0].cpu().data.numpy()
        if coords_only: return new_coords
        
        return new_coords,new_values
    # def log_data(self,logger,step,name):
    #     #print("log_data called")
    #     fig = plt.figure()
    #     ax = plt.axes(projection='3d')
    #     x,y,z = self.inp_coords.T
    #     dx,dy,dz = self.offset.T
    #     ax.cla()
    #     ax.scatter(x,y,z,c=z)
    #     ax.quiver(x,y,z,dx,dy,dz)#,length = mag)
    #     #ax.scatter(xp,yp,zp,c=zp)
    #     fig.canvas.draw()
    #     plt.show()




# class GroupPointConv(PointConvBase):
#     def __init__(self,*args,group=Trivial,**kwargs):
#         super().__init__(*args,**kwargs)
#         self.group = group
#         self.weightnet = WeightNet(self.xyz_dim+self.group.embed_dim, self.cicm_co,hidden_unit = (32, 32))

#     def get_embedded_group_elems(self,nbhd_xyz,output_xyz):
#         deltas = nbhd_xyz - output_xyz
#         embedded_group_elems = self.group.sample_origin_stabilizer(deltas)
#         return embedded_group_elems


# class GroupPointConvV2(PointConvBase):
#     def __init__(self,*args,group=Trivial,**kwargs):
#         super().__init__(*args,**kwargs)
#         self.group = group
#         self.weightnet = WeightNet(self.group.embed_dim, self.cicm_co,hidden_unit = (32, 32))

class CoordinatePointConv(PointConvBase):
    def __init__(self,*args,coordmap=Coordinates(),**kwargs):
        super().__init__(*args,**kwargs)
        self.coordmap = coordmap
        self.weightnet = WeightNet(self.xyz_dim+self.coordmap.embed_dim, self.cicm_co,hidden_unit = (32, 32))
    def get_embedded_group_elems(self,nbhd_xyz,output_xyz):
        return self.coordmap(nbhd_xyz) - self.coordmap(output_xyz)

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
@export
class PointConv(nn.Module):
    def __init__(self,in_channels,out_channels,nbhd=9,ds_frac=1,xyz_dim=2,knn_channels=None,**kwargs):
        super().__init__()
        self.basepointconv = PointConvBase(in_channels,out_channels,nbhd=nbhd,xyz_dim=xyz_dim,
                                            knn_channels=knn_channels,**kwargs)
        self.subsample = LearnedSubsample(ds_frac,knn_channels=knn_channels,nbhd=nbhd,
                                        xyz_dim=xyz_dim,chin=in_channels,**kwargs)
    def forward(self,inp):
        xyz,vals = inp
        bnc_inp = (xyz.permute(0,2,1),vals.permute(0,2,1)) 
        query_xyz = self.subsample(bnc_inp,coords_only=True)
        #print(query_xyz.shape,bnc_inp[0].shape,bnc_inp[1].shape)
        return query_xyz.permute(0,2,1),self.basepointconv(bnc_inp,query_xyz).permute(0,2,1)

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
        norm_layer = nn.BatchNorm1d
        self.pointconv = PointConv(in_channels//r,out_channels,nbhd=nbhd,ds_frac=ds_frac,**kwargs)
        self.net = nn.Sequential(
            Pass(norm_layer(in_channels)),
            Pass(nn.ReLU()),
            Pass(nn.Conv1d(in_channels,in_channels//r,1)),
            Pass(norm_layer(in_channels//r)),
            Pass(nn.ReLU()),
            self.pointconv,
            #Pass(norm_layer(out_channels)),
            #Pass(nn.ReLU()),
            #Pass(nn.Conv1d(out_channels,out_channels,1)),
        )

    def forward(self,x):
        #coords,values = x
        #print(values.shape)
        new_coords,new_values  = self.net(x)
        new_values[:,:self.in_channels] += self.pointconv.subsample(x)[1] # subsampled old values
        #print(shortcut.shape)
        #print(new_coords.shape,new_values.shape)
        return new_coords,new_values



def imagelike_nn_downsample(x,coords_only=False):
    coords,values = x
    bs,c,N = values.shape
    h = w = int(np.sqrt(N))
    ds_coords = torch.nn.functional.interpolate(coords.view(bs,2,h,w),scale_factor=0.5)
    ds_values = torch.nn.functional.interpolate(values.view(bs,c,h,w),scale_factor=0.5)
    if coords_only: return ds_coords.view(bs,2,-1)
    return ds_coords.view(bs,2,-1), ds_values.view(bs,c,-1)

def concat_coords(x):
    bs,c,h,w = x.shape
    coords = torch.stack(torch.meshgrid([torch.linspace(-1,1,h),torch.linspace(-1,1,w)]),dim=-1).view(h*w,2).unsqueeze(0).permute(0,2,1).repeat(bs,1,1).to(x.device)
    inp_as_points = x.view(bs,c,h*w)
    return (coords,inp_as_points)
def uncat_coords(x):
    bs,c,n = x[1].shape
    h = w = int(np.sqrt(n))
    return x[1].view(bs,c,h,w)

@export
def imgpConvBNrelu(in_channels,out_channels,**kwargs):
    return nn.Sequential(
        Expression(concat_coords),
        PointConv(in_channels,out_channels,**kwargs),
        Expression(uncat_coords),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

@export
class pResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,drop_rate=0,stride=1,nbhd=3**2,xyz_dim=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if xyz_dim==2:
            ds = 1 if stride==1 else imagelike_nn_downsample#
        elif xyz_dim==3:
            ds = 1 if stride==1 else .25
            
        self.net = nn.Sequential(
            Pass(nn.BatchNorm1d(in_channels)),
            Pass(nn.ReLU()),
            PointConv(in_channels,out_channels,nbhd=nbhd,
            ds_frac=1 if xyz_dim==2 else np.sqrt(ds),xyz_dim=xyz_dim),
            Pass(nn.Dropout(p=drop_rate)),
            Pass(nn.BatchNorm1d(out_channels)),
            Pass(nn.ReLU()),
            PointConv(out_channels,out_channels,nbhd=nbhd,
            ds_frac=ds if xyz_dim==2 else np.sqrt(ds),xyz_dim=xyz_dim),
        )
        self.shortcut = nn.Sequential()
        if in_channels!=out_channels:
            self.shortcut.add_module('conv',Pass(nn.Conv1d(in_channels,out_channels,1)))
        if stride!=1:
            if xyz_dim==2:
                self.shortcut.add_module('ds',Expression(lambda a: imagelike_nn_downsample(a)))
            elif xyz_dim==3:
                self.shortcut.add_module('ds',FarthestSubsample(ds_frac=ds))

    def forward(self,x):
        res_coords,res_values = self.net(x)
        skip_coords,skip_values = self.shortcut(x)
        return res_coords,res_values+skip_values
