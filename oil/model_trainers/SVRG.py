import torch
from torch.autograd import Variable
from torch.optim.optimizer import Optimizer, required


class SVRG(Optimizer):
    r"""Implements stochastic variance reduced gradient or SVRG (optionally with momentum).
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        snapshot_params (iterable): iterable of parameters for the snapshot model or dicts
            defining parameter groups in the same way as params
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        update_frequency (int, optional): determines after how many epochs the snapshot should be updated (default: 1)
    """
    def __init__(self, params, snapshot_params=None, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, update_frequency=1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        if update_frequency < 0.0:
            raise ValueError("Invalid update_frequency value: {}".format(update_frequency))
        if not isinstance(update_frequency, int):
            raise TypeError("Invalid update_frequency type: {}".format(type(update_frequency)))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        super(SVRG, self).__init__(params, defaults)

        # Store the update_frequency parameter
        self.update_frequency = update_frequency
        # Define a variable that will count the epochs that will be used with update frequency
        self.count = 0

        # Making param groups out of the snapshot parameters
        snapshot_param_group = list(snapshot_params)
        if len(snapshot_param_group) == 0:
            raise ValueError("optimizer got an empty snapshot parameter list")
        if not isinstance(snapshot_param_group[0], dict):
            snapshot_param_group = [{'snapshot_params': snapshot_param_group}]

        for param_group in snapshot_param_group:
            snapshot_params = param_group['snapshot_params']
            if isinstance(snapshot_params, Variable):
                snapshot_param_group['snapshot_params'] = [snapshot_params]
            elif isinstance(snapshot_params, set):
                raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                                'the ordering of tensors in sets will change between runs. Please use a list instead.')
            else:
                param_group['snapshot_params'] = list(snapshot_params)

        # Add the snapshot_params and the average gradient to the parameter groups
        for idx, group in enumerate(self.param_groups):
            group['snapshot_params'] = snapshot_param_group[idx]['snapshot_params']
            group['average_gradient'] = list()
            for p in group['params']:
                group['average_gradient'].append(torch.zeros_like(p.data))

    def step(self, closure=None):
        r"""Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss, snapshot_loss = None, None
        if closure is not None:
            loss, snapshot_loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError("SVRG doesn't support sparse gradients")
                snapshot_params = group['snapshot_params'][idx]
                average_gradient = group['average_gradient'][idx]
                # gradient data
                d_p = p.grad.data
                # subtract the average gradient
                d_p.add_(-1, average_gradient)
                # add the snapshot gradient
                if snapshot_params.grad is not None:
                    d_p.add_(snapshot_params.grad.data)
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

    def update_snapshot(self, dataloader, closure):
        r"""Updates the parameter snapshot and the average gradient
        Arguments:
            dataloader : A dataloader used to get the training samples.
            closure (callable): A closure that reevaluates the snapshot model
                and returns the loss.
        """
        if closure is None:
            raise RuntimeError("A closure has to be given")
        if dataloader is None:
            raise RuntimeError("Dataloader has to be given")
        if self.count % self.update_frequency != 0:
            self.count = self.count + 1
            return
        else:
            self.count = 1
        # Zero the gradient of the parameters
        self.zero_grad()
        # Take a snapshot of the latest parameters
        for group in self.param_groups:
            for p, sp in zip(group['params'], group['snapshot_params']):
                sp.data.copy_(p.data)
        # Iterate over all the dataset to compute the average gradient
        for i, (data, target) in enumerate(dataloader):
            closure(data, target)
            for group in self.param_groups:
                for idx, p in enumerate(group['snapshot_params']):
                    if p.grad is None:
                        continue
                    if i == 0:
                        group['average_gradient'][idx].zero_()
                    group['average_gradient'][idx].add_(1/len(dataloader), p.grad.data)

    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`Variable` s."""
        for group in self.param_groups:
            for p, sp in zip(group['params'], group['snapshot_params']):
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
                if sp.grad is not None:
                    sp.grad.detach_()
                    sp.grad.zero_()