import operator
import torch
import warnings
import os
from itertools import chain
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.cuda._utils import _get_device_index


def try_multigpu_parallelize(model,bs,lr=None):
    scalelr = (lr is not None)
    if os.environ.copy().get("WORLD_SIZE",0)!=0:
        assert torch.cuda.is_available(), "No GPUs found"
        ngpus = torch.cuda.device_count() # For Adam, only the bs is scaled up
        print(f"Discovered and training with {ngpus} GPUs, bs ->\
         {ngpus}*bs{f', lr -> {ngpus}*lr' if scalelr else ''}.")
        torch.distributed.init_process_group(backend="nccl")
        DDP_model = nn.parallel.DistributedDataParallel(model)#,find_unused_parameters=True) #for 1.0.0
        return (DDP_model, bs*ngpus, lr*ngpus) if scalelr else (DDP_model, bs*ngpus)
    else:
        return (model, bs, lr) if scalelr else (model,bs)

def _check_balance(device_ids):
    imbalance_warn = """
    There is an imbalance between your GPUs. You may want to exclude GPU {} which
    has less than 75% of the memory or cores of GPU {}. You can do so by setting
    the device_ids argument to DataParallel, or by setting the CUDA_VISIBLE_DEVICES
    environment variable."""
    device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))
    dev_props = [torch.cuda.get_device_properties(i) for i in device_ids]

    def warn_imbalance(get_prop):
        values = [get_prop(props) for props in dev_props]
        min_pos, min_val = min(enumerate(values), key=operator.itemgetter(1))
        max_pos, max_val = max(enumerate(values), key=operator.itemgetter(1))
        if min_val / max_val < 0.75:
            warnings.warn(imbalance_warn.format(device_ids[min_pos], device_ids[max_pos]))
            return True
        return False

    if warn_imbalance(lambda props: props.total_memory):
        return
    if warn_imbalance(lambda props: props.multi_processor_count):
        return

class MyDataParallel(nn.DataParallel):

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            attr = getattr(self.module, name)
            if callable(attr):
                funcname = name
                def parallel_closure(*inputs,**kwargs):
                    if not self.device_ids:
                        return self.module(*inputs, **kwargs)

                    for t in chain(self.module.parameters(), self.module.buffers()):
                        if t.device != self.src_device_obj:
                            raise RuntimeError("module must have its parameters and buffers "
                                            "on device {} (device_ids[0]) but found one of "
                                            "them on device: {}".format(self.src_device_obj, t.device))

                    inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
                    if len(self.device_ids) == 1:
                        return getattr(self.module,funcname)(*inputs[0], **kwargs[0])
                    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
                    outputs = self.parallel_apply([getattr(module,funcname) for module in replicas], inputs, kwargs)
                    return self.gather(outputs, self.output_device)
                return parallel_closure
            else:
                return attr

import torch.cuda.comm
import torch.distributed as dist

if dist.is_available():
    from torch.distributed.distributed_c10d import _get_default_group

from torch.cuda._utils import _get_device_index


def _find_tensors(obj):
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []
class MyDistributedDataParallel(nn.parallel.DistributedDataParallel):

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            attr = getattr(self.module, name)
            if callable(attr):
                #print("got to callable function")
                funcname = name
                def parallel_closure(*inputs,**kwargs):
                    if self.require_forward_param_sync:
                        self._sync_params()

                    if self.device_ids:
                        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
                        if len(self.device_ids) == 1:
                            output =  getattr(self.module,funcname)(*inputs[0], **kwargs[0])
                        else:
                            outputs = self.parallel_apply([getattr(module,funcname) for module in self._module_copies[:len(inputs)]], inputs, kwargs)
                            output = self.gather(outputs, self.output_device)
                    else:
                        output = getattr(self.module,funcname)(*inputs, **kwargs)

                    if torch.is_grad_enabled() and self.require_backward_grad_sync:
                        #print("grad enabled, got here")
                        self.require_forward_param_sync = True
                        # We'll return the output object verbatim since it is a freeform
                        # object. We need to find any tensors in this object, though,
                        # because we need to figure out which parameters were used during
                        # this forward pass, to ensure we short circuit reduction for any
                        # unused parameters. Only if `find_unused_parameters` is set.
                        if self.find_unused_parameters:
                            self.reducer.prepare_for_backward(list(_find_tensors(output)))
                        else:
                            self.reducer.prepare_for_backward([])
                    else:
                        self.require_forward_param_sync = False

                    return output
                return parallel_closure
            else:
                return attr
