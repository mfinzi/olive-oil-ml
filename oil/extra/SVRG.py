import torch
from copy import deepcopy
from torch.optim import SGD

class SVRG(SGD):
    r"""SVRG optimizer. Computes updates (w/ momentum) where the gradients
        $\tilde{g}(w)$ are replaced by $\tilde{g}(w) + g(w_0) - \tilde{g}(w_0)$
        where $w_0$ is a copy of the weights made in update_snapshot, and $g$ is the
        full batch gradient. """

    def update_snapshot(self, full_backward_closure):
        r""" Updates the snapshot parameters and gradients using the current
            parameters (and gradients calculated by set_full_gradients).
            `set_full_gradients` should compute the full gradients over the dataset
            into the param.grad's."""
        self.zero_grad()
        full_backward_closure()
        for group in self.param_groups:
            for p in group['params']:
                #if p.grad is None: continue
                param_state = self.state[p]
                param_state['snapshot_params'] = torch.zeros_like(p.data)#deepcopy(p.data)
                param_state['snapshot_params'].add_(p.data)
                if p.grad is None: continue
                param_state['snapshot_grads'] = torch.zeros_like(p.grad)#.clone().detach()
                param_state['snapshot_grads'].add_(p.grad.data.clone().detach())
                #print(p)
    
    def step(self, backward_closure):
        r""" Takes in closure that computes the loss on the same minibatch again
            (using the parameters of the model). IMPORTANT: This closure should
            not call zero_grads() since the gradients need to be accumulated """
        # Add snapshot grads, copy original parameters, set params to snapshot params
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['orig_params'] = torch.zeros_like(p.data)#deepcopy(p.data)
                param_state['orig_params'].add_(p.data)
                param_state['orig_grads'] = torch.zeros_like(p.grad.data)#deepcopy(p.data)
                param_state['orig_grads'].add_(p.grad.data)

                p.data.zero_()
                p.data.add_(param_state['snapshot_params'])
                p.grad.data.zero_()

                # print(self.state[param_state['snapshot_params']])
                # assert w_id == id(self.state[param_state['snapshot_params']]), "Different ids"

                # THis line is the problem, likely the grads are overwritten
                #p.data = deepcopy(param_state['snapshot_params'])
                
                # out = -1*(\tilde{g(w)} + g(w_0))
        # Compute the stochastic gradient at the snapshot params
        self.zero_grad()
        backward_closure()
        # out = -1*( \tilde{g}(w) + g(w_0) - \tilde{g}(w_0) )
        # Set the params back to the original values, and fix the grads
        #self.zero_grad()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                # Subtract the stochastic anchor grads from orig_grads
                #param_state['orig_grads'].add_(-1*p.grad.data)
                # Return to the original parameters
                if p.grad is not None:
                    stochastic_anchor_grad = torch.zeros_like(p.grad.data)
                    stochastic_anchor_grad.add_(p.grad.data)
                p.data.zero_()
                p.data.add_(param_state['orig_params'])
                
                #p.data = deepcopy(param_state['orig_params']) # Problematic line again
                if p.grad is None: continue
                # Return to orignal grads
                p.grad.data.zero_()
                p.grad.data.add_(param_state['orig_grads']-stochastic_anchor_grad)
                p.grad.data.add_(param_state['snapshot_grads'])
                #p.grad.data.mul_(-1)
                # out = \tilde{g}(w) + g(w_0) - \tilde{g}(w_0)
        # Perform the regular sgd step with the modified gradients
        return super().step()
        


        
        