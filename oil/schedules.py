import numpy as np
import torch.optim as optim

# class CosineAnnealer(optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, cycle_length, cycle_mult,
#                  with_restart=False, last_epoch=-1):
#         #self.optim_constructor = optim_constr
#         self.cycle_length = cycle_length # the base cycle length
#         self.cycle_mult = cycle_mult
#         self.with_restart = with_restart
#         super().__init__(optimizer, last_epoch)
    
#     def get_lr(self):
#         isIncreasing = self.cos_scale(self.last_epoch) > self.cos_scale(self.last_epoch-1)
#         if isIncreasing and self.with_restart:
#             self.restart_optimizer()
#         return [base_lr*self.cos_scale(self.last_epoch) for base_lr in self.base_lrs]

#     def cos_scale(self, epoch):
#         r = self.cycle_mult + 1e-6
#         L = self.cycle_length #base
#         current_cycle = np.floor(np.log(1+(r-1)*epoch/L)/np.log(r))
#         current_cycle_length = L*r**current_cycle
#         cycle_iter = epoch - L*(r**current_cycle - 1)/(r-1)
#         cos_scale = .5*(1 + np.cos(np.pi*cycle_iter/current_cycle_length))
#         return cos_scale

#     def restart_optimizer(self):
#         self.optimizer.reset_state()
#         print("Learning rate restart, optimizer reinitialized")


# class AdamR(optim.Adam):
#     def reset_state(self):
#         for group in self.param_groups:
#             for p in group['params']:
#                 self.state[p] = {}

def cosLr(cycle_length, cycle_mult=1):
    def lrSched(epoch):
        r = cycle_mult + 1e-8
        L = cycle_length #base
        current_cycle = np.floor(np.log(1+(r-1)*epoch/L)/np.log(r))
        current_cycle_length = L*r**current_cycle
        cycle_iter = epoch - L*(r**current_cycle - 1)/(r-1)
        cos_scale = .5*(1 + np.cos(np.pi*cycle_iter/current_cycle_length))
        return cos_scale
    return lrSched

def sigmoidConsRamp(rampup_length):
    def weightSched(epoch):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    return weightSched