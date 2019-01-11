
import numpy as np
from ..utils.utils import log_uniform

class SearchVariation(object):
    def __init__(self,sample_func):
        self.sample_func = sample_func
    def sample(self,config):
        out = self.sample_func(config)
        if isinstance(out,SearchVariation): raise KeyError
        return out


def sampleFrom(func):
    return SearchVariation(func)
def logUniform(low,high):
    return sampleFrom(lambda _: log_uniform(low,high))
def uniform(low,high):
    return sampleFrom(lambda _:np.random.uniform(low,high))

#TODO: better solution to config dependecy resolution via
# cycle detection in the dependency graph
def sample_config(config_spec):
    cfg_all = {}
    more_work=True
    i=0
    while more_work:
        cfg_all, more_work,latest_exception = _sample_config(config_spec,cfg_all)
        i+=1
        if i>10: raise latest_exception# or RecursionError("config dependency unresolvable")
    return cfg_all

def _sample_config(config_spec,cfg_all=None):
    cfg = {}
    more_work = False
    latest_exception = None
    for k,v in config_spec.items():
        if isinstance(v,list):
            cfg[k] = np.random.choice(v)
        elif isinstance(v,dict):
            new_dict,extra_work,latest_exception = _sample_config(v,cfg_all)
            cfg[k] = new_dict
            more_work |= extra_work
        elif isinstance(v,SearchVariation):
            try:cfg[k] = v.sample(cfg_all)
            except Exception as e: 
                #TODO: Handle non KeyError when SearchVariation
                cfg[k] = v # is used isntead of the variable it returns
                more_work = True
                latest_exception = e
        else: cfg[k] = v
    return cfg, more_work, latest_exception