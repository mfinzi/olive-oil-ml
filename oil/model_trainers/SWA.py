import torch


def apply_SWA(module):
    param_list = module.parameters()
    print(next(param_list)['lr'])

class SWA(object):
    def __init__(self, name='weight'):
        pass
    def __call__(self,module,inputs):
        if module.training:
            weight_average = getattr(module,self.name+'_wa')
            current_weight = module._parameters[name]

    @staticmethod
    def apply(module,name):
        # Check that SWA hook doesn't already exist
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook,SWA) and hook.name==name:
                raise RuntimeError("Cannot register two SWA hooks on \
                                   the same parameter {}".format(name))
        fn = SWA(name)
        weight = module._parameters[name]
        module.register_buffer(name+"_wa",torch.zeros_like(weight))
        module.register_forward_pre_hook(fn)
