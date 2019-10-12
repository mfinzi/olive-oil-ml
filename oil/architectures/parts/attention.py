import torch
import torch.nn.functional as F
import torch.nn as nn
from ...utils.utils import Expression,export,Named

def Attention(Q,K,V):
    """ Assumes Q,K,V have shape (bs,N,d)"""
    bs,N,d = K.shape
    Kt = K.transpose(-1,-2)
    similarity = Q@Kt/np.sqrt(d)
    return F.softmax(similarity,dim=-1)@V

class SelfAttentionHead(nn.Module):
    
    def __init__(self,inp_channels, outp_channels):
        super().__init__()
        self.WQ = nn.Linear(inp_channels,outp_channels)
        self.WK = nn.Linear(inp_channels,outp_channels)
        self.WV = nn.Linear(inp_channels,outp_channels)
    def forward(self,X):
        """ Assumes X has shape (bs,N,d)"""
        return Attention(self.WQ(X),self.WK(X),self.WV(X))

class MultiHeadAtt(nn.Module):
    def __init__(self,inp_channels,num_heads):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(inp_channels,inp_channels/num_heads)
                                                                for _ in range(num_heads)])
        self.WO = nn.Linear(inp_channels,inp_channels)
    def forward(self,X):
        return self.WO(torch.cat([head(X) for head in self.heads],dim=-1))



