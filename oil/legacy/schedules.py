import numpy as np
import torch.optim as optim
import torch
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

class LRSchedulerWithInherit(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, interval=1, last_epoch=-1):
        self.interval = interval
        super().__init__(optimizer, lr_lambda)

    def step(self, epoch=None):
        #self.optimizer.update()
        assert (self.interval==1 or epoch is not None), "Epoch arg needed for intervals"
        if epoch%self.interval==0: self.optimizer.inherit()
        super().step(epoch)

class ASGD(optim.SGD):
    def step(self, closure=None):
        super().step(closure)
        self.update()
    
    def update(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                weight = 1#/group['lr']
                self.updateAverageBuffers(p, group, weight)

    def updateAverageBuffers(self, p, group,w):
        param_state = self.state[p]
        if 'param_average' not in param_state:
            param_state['weight_sum'] = 0
            param_state['param_average'] = torch.zeros_like(p.data)
            param_state['mom_average'] = torch.zeros_like(p.data)
        wsum = param_state['weight_sum']
        wsum += w
        pave = param_state['param_average']
        pave.mul_(wsum/(w+wsum)).add_(w/(w+wsum), p.data)
        mave = param_state['mom_average']
        mave.mul_(wsum/(w+wsum)).add_(w/(w+wsum), param_state['momentum_buffer'])

    def inherit(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_state = self.state[p]
                    p.data.zero_()
                    p.data += param_state['param_average'].data
                    param_state['momentum_buffer'].zero_()
                    param_state['momentum_buffer'] += param_state['mom_average'].data
                    param_state['weight_sum'] = 0


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

def triangle(numEpochs):
    es = numEpochs-1
    thresh1 =(3/8)*es; val1 =1
    thresh2 = (6/8)*es; val2 = 10
    val3 = 1
    def lr_lambda(e):
        dist1 = e/thresh1
        dist2 = (e-thresh1)/(thresh2-thresh1)
        dist3 = (e-thresh2)/(es-thresh2)
        term1 = (dist1>=0)*(dist1<1)*(dist1*val2 +(1-dist1)*val1)
        term2 = (dist2>=0)*(dist2<1)*(dist2*val3 +(1-dist2)*val2)
        term3 = (dist3>=0)*(dist3<1)*((1-dist3)*val3)
        return term1+term2+term3
    return lr_lambda

def triangle2(numEpochs):
    es = numEpochs-1
    thresh1 =(2/8)*es; val1 =8
    thresh2 = (4/8)*es; val2 = 3
    val3 = .5
    def lr_lambda(e):
        dist1 = e/thresh1
        dist2 = (e-thresh1)/(thresh2-thresh1)
        dist3 = (e-thresh2)/(es-thresh2)
        term1 = (dist1>=0)*(dist1<1)*(dist1*val2 +(1-dist1)*val1)
        term2 = (dist2>=0)*(dist2<1)*(dist2*val3 +(1-dist2)*val2)
        term3 = (dist3>=0)*(dist3<1)*((1-dist3)*val3)
        return term1+term2+term3
    return lr_lambda

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