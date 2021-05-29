import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import numpy as np
from ...utils.utils import Expression,export,Named

@export
class GanBase(nn.Module,metaclass=Named):

    def __init__(self,z_dim,img_channels,num_classes=None):
        self.z_dim = z_dim
        self.img_channels = img_channels
        super().__init__()

    @property
    def device(self):
        try: return self._device
        except AttributeError:
            self._device = next(self.parameters()).device
            return self._device

    def sample_z(self, n=1):
        return torch.randn(n, self.z_dim).to(self.device)

    def sample(self, n=1):
        return self(self.sample_z(n))


def add_spectral_norm(module):
    if isinstance(module,  (nn.ConvTranspose1d,
                            nn.ConvTranspose2d,
                            nn.ConvTranspose3d,
                            )):
        spectral_norm(module,dim = 1)
        #print("SN on conv layer: ",module)
    elif isinstance(module, (nn.Linear,
                            nn.Conv1d,
                            nn.Conv2d,
                            nn.Conv3d)):
        spectral_norm(module,dim = 0)
        #print("SN on linear layer: ",module)

def xavier_uniform_init(module):
    if isinstance(module,  (nn.ConvTranspose1d,
                            nn.ConvTranspose2d,
                            nn.ConvTranspose3d,
                            nn.Conv1d,
                            nn.Conv2d,
                            nn.Conv3d)):
        if module.kernel_size==(1,1):
            nn.init.xavier_uniform_(module.weight.data,np.sqrt(2))
        else:
            nn.init.xavier_uniform_(module.weight.data,1)
        #print("Xavier init on conv layer: ",module)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data,1)
        #print("Xavier init on linear layer: ",module)