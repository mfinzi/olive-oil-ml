import numpy as np
import torch
import numbers
import time
import torch.nn as nn
import inspect
import copy
import os
import dill

class map_with_len(object):
    def __init__(self, func, iter_with_len):
        self._func = func
        self._iter = iter_with_len
    def __iter__(self):
        return map(self._func, self._iter)
    def __len__(self):
        return len(self._iter)

def loader_to(device):
    """Returns a function that sends dataloader output
         to the specified device"""
    def minibatch_map(mb):
        try: return mb.to(device)
        except AttributeError: 
            return type(mb)(map(lambda x:x.to(device),mb))
    return lambda loader: map_with_len(minibatch_map, loader)

## Depreciated methods for performing to_gpu
# def to_var_gpu(x, **kwargs):
#     """ recursively map the elements to variables on gpu """
#     if x is None: 
#         return x
#     if type(x) in [list, tuple]:
#         curry = lambda x: to_var_gpu(x, **kwargs)
#         return type(x)(map(curry, x))
#     else: 
#         return Variable(x, **kwargs).cuda()

# def to_gpu(x, double=False, cpu=False):
#     if x is None:
#         return x
#     if type(x) in [list, tuple]:
#         curry = lambda x: to_gpu(x, double, cpu)
#         return type(x)(map(curry, x))
#     else:
#         if double and x.dtype==torch.float32: x = x.double()
#         if cpu: return x.cpu()
#         return x.to(torch.device("cuda"))

class Eval(object):
    def __init__(self, model):
        self.model = model
    def __enter__(self):
        self.training_state = self.model.training
        self.model.train(False)
    def __exit__(self, *args):
        self.model.train(self.training_state)

class FixedNumpySeed(object):
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.rng_state = np.random.get_state()
        np.random.seed(self.seed)
    def __exit__(self, *args):
        np.random.set_state(self.rng_state)

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

# Now part of LazyLogger.py in logging
# class logTimer(object):
#         def __init__(self, mins = 1, timeFrac = 1/10):
#             self.avgLogTime = 0
#             self.numLogs = 0
#             self.lastLogTime = 0
#             self.mins = mins
#             self.timeFrac = timeFrac
#             self.performedLog = False
#         def __enter__(self):
#             timeSinceLog = time.time() - self.lastLogTime
#             self.performedLog = (timeSinceLog>60*self.mins) \
#                                 and (self.avgLogTime<=self.timeFrac*timeSinceLog)
#             if self.performedLog: self.lastLogTime = time.time()
#             return self.performedLog
#         def __exit__(self, *args):
#             if self.performedLog:
#                 timeSpentLogging = time.time()-self.lastLogTime
#                 n = self.numLogs
#                 self.avgLogTime = timeSpentLogging/(n+1) + self.avgLogTime*n/(n+1)
#                 self.numLogs += 1
#                 self.lastLogTime = time.time()

def to_lambda(x):
    """ Turns constants into constant functions """
    if isinstance(x, numbers.Number):
        return lambda e: x
    else:
        return x

def prettyPrintLog(logDict, *epochIts):
    formatStr = "[%3d/%d][%6d/%d] "
    valuesTuple = epochIts
    for key, val in logDict.items():
        formatStr += (key+": %.3f ")
        valuesTuple += (val,)
    print(formatStr % valuesTuple)

def logOneMinusSoftmax(x):
    """ numerically more stable version of log(1-softmax(x)) """
    max_vals, _ = torch.max(x, 1)
    shifted_x = x - max_vals.unsqueeze(1).expand_as(x)
    exp_x = torch.exp(shifted_x)
    sum_exp_x = exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
    k = x.size()[1]
    batch_size = x.size()[0]
    sum_except_matrix = (torch.ones(k,k) - torch.eye(k)).cuda()
    resized_sum_except_m = sum_except_matrix.squeeze(0).expand(batch_size,k,k)
    sum_except_exp_x = torch.bmm(resized_sum_except_m, exp_x.unsqueeze(2)).squeeze()
    return torch.log(sum_except_exp_x) - torch.log(sum_exp_x)


def init_args():
    frame = inspect.currentframe()
    outer_frames = inspect.getouterframes(frame)
    caller_frame = outer_frames[1][0]
    args =inspect.getargvalues(caller_frame)[-1]
    args.pop("self")
    return args

# Coded by Massimiliano Tomassoli, 2012.
def genCur(func, unique = True, minArgs = None):
    """ Generates a 'curried' version of a function. """
    def g(*myArgs, **myKwArgs):
        def f(*args, **kwArgs):
            if args or kwArgs:                  # some more args!
                # Allocates data to assign to the next 'f'.
                newArgs = myArgs + args
                newKwArgs = dict.copy(myKwArgs)
 
                # If unique is True, we don't want repeated keyword arguments.
                if unique and not kwArgs.keys().isdisjoint(newKwArgs):
                    raise ValueError("Repeated kw arg while unique = True")
 
                # Adds/updates keyword arguments.
                newKwArgs.update(kwArgs)
 
                # Checks whether it's time to evaluate func.
                numArgsIn = len(newArgs) + len(newKwArgs)
                totalArgs = len(inspect.getfullargspec(func).args)
                namedArgs = 0 if func.__defaults__ is None else len(func.__defaults__)
                numArgsRequired = totalArgs - namedArgs
                if (minArgs is not None and minArgs <= numArgsIn) \
                    or (minArgs is None and numArgsRequired <= len(newArgs)):
                    #print(newArgs)
                    #print(newKwArgs)
                    return func(*newArgs, **newKwArgs)  # time to evaluate func
                else:
                    return g(*newArgs, **newKwArgs)     # returns a new 'f'
            else:                               # the evaluation was forced
                return func(*myArgs, **myKwArgs)
        return f
    return g

def curry(f,minArgs = None):
    return genCur(f, True, minArgs)

def cur(f, minArgs = None):
    return genCur(f, True, minArgs)
 
def curr(f, minArgs = None):
    return genCur(f, False, minArgs)

# Wraps a generator so that calling __iter__ multiple
#    times produces distinct non-empty generators  
class multiGen(object):
    def __init__(self, generator_constructor):
        self._gen = generator_constructor
    def __iter__(self):
        return self._gen()
    def __len__(self):
        return len(self._gen())

def dillcopy(obj):
    return dill.loads(dill.dumps(obj))

## Super hacky method to that returns a method that constructs the same object
## as the object constructed when this is called in an __init__ method of a base
## class
def reconstructor():
    frame = inspect.currentframe()
    outer_frames = inspect.getouterframes(frame)
    subclass_depth=-1
    while inspect.getframeinfo(outer_frames[subclass_depth+2][0])[2]=='__init__':
        subclass_caller_frame = outer_frames[subclass_depth+2][0]
        subclass_depth +=1
        
    assert subclass_depth >=0, "Not called in an __init__ method"
    #print("subclass depth = {}".format(subclass_depth))
    argnames,varargname,keywordname,localss = inspect.getargvalues(subclass_caller_frame)
    args_in = {k:v for k,v in localss.items() if k in argnames}
    cls = args_in.pop("self").__class__
    args_in_copy = dillcopy(args_in)
    
    args = dillcopy(localss[varargname]) if varargname is not None else None
    kwargs = dillcopy(localss[keywordname]) if keywordname is not None else {}
    kwargs.update(args_in_copy)
    
    if args is not None: return lambda **newKwArgs: cls(*args,**dict(kwargs,**newKwArgs))
    else: return lambda **newKwArgs: cls(**dict(kwargs,**newKwArgs))

def make_like(reconstructible):
    if isinstance(reconstructible, str):
        load_path = reconstructible
        if os.path.isfile(load_path):
            state = torch.load(load_path, pickle_module=dill)
            return state['reconstructor'](rebuildable=False)
        else:
            print("=> no checkpoint found at '{}'".format(load_path))
    else: #Then it is a live object
        return reconstructible.reconstructor(rebuildable=False)

def full_load(reconstructible):
    if isinstance(reconstructible, str):
        load_path = reconstructible
        if os.path.isfile(load_path):
            state = torch.load(load_path, pickle_module=dill)
            model = state['reconstructor'](rebuildable=False)
            model.load_state(state)
            return 
        else:
            print("=> no checkpoint found at '{}'".format(load_path))
    else:
        model = reconstructible.reconstructor(rebuildable=False)
        model.load_state(model.get_state())
