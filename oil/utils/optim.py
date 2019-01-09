import torch
from torchcontrib.optim import SWA

## Inverse LR wieghting automatic SWA
class AutoSWA(SWA):
    def __init__(self,*args,swa_start=0,swa_freq=1000,**kwargs):
        super().__init__(*args,swa_start=swa_start,swa_freq=swa_freq,**kwargs)

    def update_swa_group(self,group):
        coeff_new = 1/group["lr"]
        group["n_avg"] += coeff_new
        for p in group['params']:
            param_state = self.state[p]
            if 'swa_buffer' not in param_state:
                param_state['swa_buffer'] = torch.zeros_like(p.data)
            buf = param_state['swa_buffer']
            diff = (p.data - buf)*coeff_new/group["n_avg"]
            buf.add_(diff)
