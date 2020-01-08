import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

import numpy as np
from torch.nn.utils import weight_norm
import math
from ..parts import conv2d
from ...utils.utils import Expression,export,Named


def RestrictedAttention(Q,K,V,P=0):
    """ Self attention mechanism, O = softmax(QK^T/sqrt(d) + P)V
        Q matrix has shape (bs,n,d)
        K matrix has shape (bs,n,r,d)  where r is the number of points in nbhd
        V matrix has shape (bs,n,r,d)
        P matrix has shape (bs,n,r)
        O output has shape (bs,n,d) """
    n,d = Q.shape[-2:] # (bs,n,1,d)@(bs,n,d,r) -> (bs,n,1,r) -> (bs,n,r)
    att_scores = (Q.unsqueeze(2)@K.permute(0,1,3,2)).squeeze(2)/np.sqrt(d) # (bs,n,r)
    weighting = torch.softmax(att_scores + P,axis=-1) # (bs,n,r)
    # (bs,n,1,r)@(bs,n,r,d) -> (bs,n,1,d) -> (bs,n,d)
    weighted_values = (weighting.unsqueeze(2)@V).squeeze(2) 
    return weighted_values

def fold_heads_into_batchdim(x,num_heads):
    """ Converts x of shape (bs,*,d) -> (num_heads*bs,*,d//num_heads)"""
    d = x.shape[-1]
    bs = x.shape[0]
    M = len(x.shape)
    heads_at_front = x.view(*x.shape[:-1],d//num_heads,num_heads).permute(M,*range(M))
    return heads_at_front.reshape(bs*num_heads,*x.shape[1:-1],d//num_heads)

def fold_heads_outof_batchdim(x,num_heads):
    """ Converts x of shape (num_heads*bs,*,d//num_heads) -> (bs,*,d)"""
    bs = x.shape[0]//num_heads
    d = x.shape[-1]*num_heads
    M = len(x.shape)
    return x.view(num_heads,bs,*x.shape[1:]).permute(*range(1,M+1),0).reshape(bs,*x.shape[1:-1],d)

class FFNPositionalNetwork(nn.Module):
    def __init__(self,ch,nbhd_extractor,num_heads):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,ch),nn.ReLU(),nn.Linear(ch,num_heads))
        self.nbhd_extractor = nbhd_extractor # (bs,n,d) -> (bs,n,r,d)
    def forward(self,x):
         # assumes x is an image with shape (bs,h,w,c)
        bs,c,h,w = x.shape                                              # (1,h*w,2)
        coords = torch.stack(torch.meshgrid([torch.linspace(-3,3,h),torch.linspace(-3,3,w)]),dim=-1).view(h*w,2).unsqueeze(0)
        relative_positional_enc = self.nbhd_extractor(coords) - coords.unsqueeze(2) #(p'-p), (1,h*w,r,2)
        positional_scores = self.net(relative_positional_enc.cuda()) # (1,h*w,r,2) -> =(1,h*w,r,nh)
        return positional_scores.repeat(bs,1,1,1) #(bs,h*w,r,nh)

class RestrictedSelfAttention(nn.Module):
    def __init__(self,ch_in,nbhd_extractor,num_heads=8):
        super().__init__()
        self.WQ = nn.Linear(ch_in,ch_in)
        self.WK = nn.Linear(ch_in,ch_in)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        self.WV = nn.Linear(ch_in,ch_in)
        self.WO = nn.Linear(ch_in,ch_in)            #TODO: unfix 2
        self.nbhd_extractor = nbhd_extractor
        self.num_heads = num_heads

    def forward(self,X,P=0):
        #print(X.shape)
        #print(P.shape)
        """Expects X shape (bs,n,d), P of shape (bs,n,r,nh)"""
        Queries = fold_heads_into_batchdim(self.WQ(X),self.num_heads) #(nh*bs,n,d//nh)
        nbhd_Keys = fold_heads_into_batchdim(self.nbhd_extractor(self.WK(X)),self.num_heads) #(nh*bs,n,r,d//nh)
        nbhd_Vals = fold_heads_into_batchdim(self.nbhd_extractor(self.WV(X)),self.num_heads) #(nh*bs,n,r,d//nh)
        nbhd_Pos = fold_heads_into_batchdim(P,self.num_heads).squeeze(-1)                  #(nh*bs,n,r)
        folded_attended_vals = RestrictedAttention(Queries,nbhd_Keys,nbhd_Vals,nbhd_Pos)
        attended_vals = fold_heads_outof_batchdim(folded_attended_vals,self.num_heads)  #(nh*bs,n,d//nh) ->(bs,n,d)
        return self.WO(attended_vals)

# Plan:
# 1) implement standalone self-attention conv replacement layer, verify that it works on cifar10
#       - (investigate replacing bottleneck block with a transformer block (w/ FFN)) #2 relus vs 1
#       - replace positional encoding style with that used in https://arxiv.org/pdf/1904.11491.pdf
#       - There are two options: multi-head self attention or just convolution
# 2) square 7 x 7 block -> nearest 50 neighbors precomputed on the images (could use kd-tree because 2d)
#       - optimize implementation efficiency
#       - tune number of neighbors
# 3) investigate non grid pooling mechanisms. Candidates:
#       a) random subsampling
#       b) randomly placed w/positional (or other) attention
#       c) bottom up superpixel segmentation/aggregation
#       d) neural net parametrizes a density p(x,y) proportional to e^{-f(x,y)}
#               - use HMC or NUTS to sample and use attention to aggregate
#               - restrict to subset of original points, subsample or use attention to aggregate
# 4) investigate predicting subsample factor sigmoid (0,1) and penalizing by the computation time
#       - So that this factor has pos signal, linearly iterpolate perf at more/less points
#       - subsample factors to be shared across elements in the minibatch to ease batching
#       - design heuristic so that no scheduling of the cost factor is necessary, eg: 
# 5) Alternatively, control downsampling factor to maximize the derivative of train/val loss wrt time = dl/di di/dt
#       - can use 



def extract_image_patches(x, kernel, stride=1, dilation=1):
    """Assumes input has shape (bs,c,h,w) output has shape"""
    # Do TF 'SAME' Padding
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
    
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    # has shape [bs,h,w,c,k,k]
    return patches


def square_nbhd_extractor(diameter):
    # diam is the diameter of the square nbhd, ie nbhd is (diam,diam)
    def diam_square_nbhd_extractor(x):
        # (bs,n,d) -> (bs,n,r,d)
        r = diameter**2
        bs,n,d = x.shape # unpack x into an image
        h = w = int(np.sqrt(n)) # assume img is square atm
        patches = extract_image_patches(x.permute(0,2,1).reshape(bs,h,w,d),diameter) #(bs,h,w,d,k,k)
        return patches.view(bs,n,d,r).permute(0,1,3,2)
    return diam_square_nbhd_extractor



class AttConvReplacement(nn.Module):
    def __init__(self, channels, ksize, num_heads=8):
        super().__init__()
        self.nbhd_extractor = square_nbhd_extractor(ksize)
        self.position_network = FFNPositionalNetwork(channels*8,self.nbhd_extractor,num_heads)
        self.mha = RestrictedSelfAttention(channels,self.nbhd_extractor,num_heads)
    def forward(self,X):
        P = self.position_network(X)
        bs,c,h,w = X.shape
        X_as_points = X.permute(0,2,3,1).view(bs,h*w,c)
        # (bs,n,c) - > (bs,c,h,w)
        return self.mha(X_as_points,P).permute(0,2,1).view(X.shape)

class PositionOnlyAtt(nn.Module):
    def __init__(self, ch, ksize):
        super().__init__()
        self.nbhd_extractor = square_nbhd_extractor(ksize)
        self.position_network = nn.Sequential(nn.Linear(2,ch),nn.ReLU(),nn.Linear(ch,ch))
    def forward(self,x):
         # assumes x is an image with shape (bs,h,w,c)
        bs,c,h,w = x.shape                                              # (1,h*w,2)
        coords = torch.stack(torch.meshgrid([torch.linspace(-3,3,h),torch.linspace(-3,3,w)]),dim=-1).view(h*w,2).unsqueeze(0)
        relative_positional_enc = self.nbhd_extractor(coords) - coords.unsqueeze(2) #(p'-p), (1,h*w,r,2)
        P = self.position_network(relative_positional_enc.cuda()) # (1,h*w,r,2) -> (1,h*w,r,nh)
        weighting = fold_heads_into_batchdim(torch.softmax(P,axis=2).repeat(bs,1,1,1),c).squeeze(-1)#(bs,n,r,c)->(bs*c,n,r)
        X_as_points = x.permute(0,2,3,1).view(bs,h*w,c) #(bs,c,h,w) -> (bs,h*w,c)
        V = fold_heads_into_batchdim(self.nbhd_extractor(X_as_points)).squeeze(-1)  # (bs,n,c) -> (bs,n,r,c) -> (bs*c,n,r)
        # (bs*c,n,r)*(bs*c,n,r) -> (bs,n,1,d) -> (bs,n,d)
        return fold_heads_outof_batchdim((weighting*V).sum(-1).unsqueeze(-1),c).permute(0,2,1).view(x.shape)

class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        v_out_h, v_out_w = v_out.split(self.out_channels // 2, dim=1)
        v_out = torch.cat((v_out_h + self.rel_h, v_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

@export
class AttResBlock(nn.Module):
    def __init__(self,k=64,ksize=7,drop_rate=0,stride=1,gn=False,num_heads=8):
        super().__init__()
        norm_layer = (lambda c: nn.GroupNorm(c//16,c)) if gn else nn.BatchNorm2d
        self.net = nn.Sequential(
            norm_layer(k),
            nn.ReLU(),
            conv2d(k,k,1),
            norm_layer(k),
            nn.ReLU(),
            AttentionConv(k,k,kernel_size=ksize, padding=ksize//2, groups=8),#conv2d(k,k,3),#AttConvReplacement(k,ksize,num_heads),
            nn.Dropout(p=drop_rate),
        )

    def forward(self,x):
        return x + self.net(x)

@export
class layer13a(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, num_targets=10,k=64,ksize=7,num_heads=8):
        super().__init__()
        self.num_targets = num_targets
        self.net = nn.Sequential(
            conv2d(3,k,1),#AttConvReplacement(3,k,ksize),
            *[AttResBlock(k,ksize,num_heads=num_heads) for i in range(3)],
            nn.AvgPool2d(2),
            conv2d(k,2*k,1),
            *[AttResBlock(2*k,ksize,num_heads=num_heads) for i in range(3)],
            nn.AvgPool2d(2),
            *[AttResBlock(2*k,ksize,num_heads=num_heads) for i in range(3)],
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_targets)
        )
    def forward(self,x):
        return self.net(x)

# Version 2, more transformer style oriented

def FFN(k):
    ## assumes bs, *, k
    return nn.Sequential(nn.Conv2d(k,4*k,1),nn.ReLU(),nn.Conv2d(4*k,k,1))

class AddAndNorm(nn.Module):
    def __init__(self,block,ch_in,dropout=0):
        super().__init__()
        self.block = block
        self.layerNorm = nn.BatchNorm2d(ch_in)#nn.LayerNorm(normalized_shape=(ch_in,))
        self.dropout = nn.Dropout(dropout)
    def forward(self,X):
        """Expects X shape (bs,n,d)"""
        # Non Standard (pre-residual) placement of layernorm, see: https://openreview.net/pdf?id=B1x8anVFPr
        return X+self.dropout(self.block(self.layerNorm(X)))


class TransformerBlock(nn.Module):
    def __init__(self,hidden_dim,ksize=5,num_heads=8,dropout=0):
        super().__init__()
        MHA = AttConvReplacement(hidden_dim, ksize, num_heads)
        FF = FFN(hidden_dim)
        self.net = nn.Sequential(AddAndNorm(MHA,hidden_dim,dropout),AddAndNorm(FF,hidden_dim,dropout))
    def forward(self,X):
        """Expects X shape (bs,n,hidden_dim)"""
        return self.net(X)

@export
class layer13at(nn.Module,metaclass=Named):
    """
    Very small CNN
    """
    def __init__(self, num_targets=10,k=64,ksize=5,num_heads=8):
        super().__init__()
        self.num_targets = num_targets
        self.net = nn.Sequential(
            conv2d(3,k,1),#AttConvReplacement(3,k,ksize),
            *[TransformerBlock(k,ksize,num_heads) for _ in range(2)],
            nn.AvgPool2d(2),
            conv2d(k,2*k,1),
             *[TransformerBlock(2*k,ksize,num_heads) for _ in range(2)],
            nn.AvgPool2d(2),
             *[TransformerBlock(2*k,ksize,num_heads) for _ in range(2)],
            Expression(lambda u:u.mean(-1).mean(-1)),
            nn.Linear(2*k,num_targets)
        )
    def forward(self,x):
        return self.net(x)
