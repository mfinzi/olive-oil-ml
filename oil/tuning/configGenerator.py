
import numpy as np
import numbers
from ..utils.utils import log_uniform,ReadOnlyDict
from collections import Iterable
# class SearchVariation(object):
#     def __init__(self,sample_func):
#         self.sample_func = sample_func
#     def sample(self,config):
#         out = self.sample_func(config)
#         if isinstance(out,SearchVariation): raise KeyError
#         return out


# def sampleFrom(func):
#     return SearchVariation(func)
def logUniform(low,high):
    return lambda _: log_uniform(low,high)
def uniform(low,high):
    return lambda _:np.random.uniform(low,high)

class NoGetItLambdaDict(dict):
    """ Regular dict, but refuses to __getitem__ pretending
        the element is not there and throws a KeyError
        if the value is a non string iterable or a lambda """
    def __getitem__(self, key):
        value = super().__getitem__(key)
        if isinstance(value,Iterable) and not isinstance(value,(str,bytes)):
            raise LookupError("You shouldn't try to retrieve iterable from this dict")
        if callable(value) and value.__name__ == "<lambda>":
            raise LookupError("You shouldn't try to retrieve lambda from this dict")
        return value
        
    # pop = __readonly__
    # popitem = __readonly__

def sample_config(config_spec):
    """ Generates configs from the config spec.
        It will apply lambdas that depend on the config and sample from any
        iterables, make sure that no elements in the generated config are meant to 
        be iterable or lambdas, strings are allowed."""
    cfg_all = NoGetItLambdaDict(config_spec)
    more_work=True
    i=0
    while more_work:
        cfg_all, more_work = _sample_config(cfg_all,cfg_all)
        i+=1
        if i>10: raise RecursionError("config dependency unresolvable")
    return dict(cfg_all)#ReadOnlyDict(cfg_all)

def _sample_config(config_spec,cfg_all):
    cfg = NoGetItLambdaDict()
    more_work = False
    for k,v in config_spec.items():
        if isinstance(v,dict):
            new_dict,extra_work = _sample_config(v,cfg_all)
            cfg[k] = new_dict
            more_work |= extra_work
        elif isinstance(v,Iterable) and not isinstance(v,(str,bytes)):
            cfg[k] = np.random.choice(v)
        elif callable(v) and v.__name__ == "<lambda>":
            try:cfg[k] = v(cfg_all)
            except (KeyError, LookupError):
                cfg[k] = v # is used isntead of the variable it returns
                more_work = True
        else: cfg[k] = v
    return cfg, more_work

# def grid_search(config_spec):
#     """ Generates configs from the a grid search on the config spec.
#     """
#     cfg_all = NoGetItLambdaDict(config_spec)
#     more_work=True
#     i=0
#     while more_work:
#         cfg_all, more_work = _sample_config(cfg_all,cfg_all)
#         i+=1
#         if i>10: raise RecursionError("config dependency unresolvable")
#     return dict(cfg_all)#ReadOnlyDict(cfg_all)

# def _sample_config(config_spec,cfg_all):
#     cfg = NoGetItLambdaDict()
#     more_work = False
#     for k,v in config_spec.items():
#         if isinstance(v,dict):
#             new_dict,extra_work = _sample_config(v,cfg_all)
#             cfg[k] = new_dict
#             more_work |= extra_work
#         elif isinstance(v,Iterable) and not isinstance(v,(str,bytes)):
#             cfg[k] = np.random.choice(v)
#         elif callable(v) and v.__name__ == "<lambda>":
#             try:cfg[k] = v(cfg_all)
#             except (KeyError, LookupError):
#                 cfg[k] = v # is used isntead of the variable it returns
#                 more_work = True
#         else: cfg[k] = v
#     return cfg, more_work

def flatten_dict(d):
    """ Flattens a dictionary, ignoring outer keys. Only
        numbers and strings allowed, others will be converted
        to a string. """
    out = {}
    for k,v in d.items():
        if isinstance(v,dict):
            out.update(flatten_dict(v))
        elif isinstance(v,(numbers.Number,str,bytes)):
            out[k] = v
        else:
            out[k] = str(v)
    return out