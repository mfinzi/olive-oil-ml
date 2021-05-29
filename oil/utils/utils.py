import numpy as np
import torch
import numbers
import time
import torch.nn as nn
import inspect
import copy
import os
import dill
import itertools
import sys
import torch.utils.data
import collections
import random


class Named(type):
    def __str__(self):
        return self.__name__
    def __repr__(self):
        return self.__name__

def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn

def log_uniform(low,high,size=None):
    logX = np.random.uniform(np.log(low),np.log(high),size)
    return np.exp(logX)

class ReadOnlyDict(dict):
    def __readonly__(self, *args, **kwargs):
        raise RuntimeError("Cannot modify ReadOnlyDict")
    __setitem__ = __readonly__
    __delitem__ = __readonly__
    pop = __readonly__
    popitem = __readonly__
    clear = __readonly__
    update = __readonly__
    setdefault = __readonly__
    del __readonly__

# class map_with_len(object):
#     def __init__(self, func, iter_with_len):
#         self._func = func
#         self._iter = iter_with_len
#     def __iter__(self):
#         return map(self._func, self._iter)
#     def __len__(self):
#         return len(self._iter)

# class imap(torch.utils.data.DataLoader):
#     def __init__(self,func,loader):
#         try: loader_func = loader._func
#         except AttributeError: loader_func=lambda x:x
#         self.__dict__ = loader.__dict__
#         self._dl = loader
#         self._func = lambda x: func(loader_func(x))
#     def __iter__(self):
#         return map(self._func,self._dl.__iter__())
# def imap(func,loader):
#     class _imap(loader.__class__):
#         def __init__(self,loader):
#             self.__dict__ = loader.__dict__
#         def __iter__(self):
#             return map(func,super().__iter__())
#     return _imap(loader)

class Wrapper(object):
    # Special methods are dispatched by what is defined in the class rather
    # than the instance, so it bypasses __getattr__, as a result for a
    # wrapper that makes use of any of these methods, we must dynamically dispatch
    # the special methods at the instance level (using getattr)
    def __init__(self, obj):
        self._wrapped_obj = obj
    def __getattr__(self, attr):
        if attr =='_wrapped_obj': raise AttributeError
        if attr == '__dict__': assert False
        #if attr not in self.__dict__: raise AttributeError
        return getattr(self._wrapped_obj, attr)
smethods =    '''__bool__ __int__ __float__ __complex__ __index__
                 __len__ __getitem__ __setitem__ __delitem__ __contains__
                 __iter__ __next__ __reversed__
                 __call__ __enter__ __exit__
                 __str__ __repr__  __bytes__ __format__
                 __eq__ __ne__ __lt__ __le__ __gt__ __ge__ __hash__
                 __add__ __mul__ __sub__ __truediv__ __floordiv__ __mod__
                 __and__ __or__ __xor__ __invert__ __lshift__ __rshift__
                 __pos__ __neg__ __abs__ __pow__ __divmod__
                 __round__ __ceil__ __floor__ __trunc__
                 __radd__ __rmul__ __rsub__ __rtruediv__ __rfloordiv__ __rmod__
                 __rand__ __ror__ __rxor__ __rlshift__ __rrshift__
                 __rpow__ __rdivmod__ __getitem__ 
                 __get__ __set__ __delete__
                 __dir__ __sizeof__'''.split()
for sm in smethods:
    setattr(Wrapper, sm, lambda self, *args, sm=sm: Wrapper.__getattr__(self,sm)(*args))

class dmap(Wrapper):
    def __init__(self,func,dataset):
        super().__init__(dataset)
        self._func = func
    def __getitem__(self,i):
        return self._func(super().__getitem__(i))

class imap(Wrapper):
    def __init__(self,func,loader):
        super().__init__(loader)
        self._func = func
    def __iter__(self):
        return map(self._func,super().__iter__())

class islice(Wrapper):
    def __init__(self,loader,*args,**kwargs):
        super().__init__(loader)
        self._args = args
        self._kwargs = kwargs
    def __iter__(self):
        return iter(itertools.islice(super().__iter__(),*self._args,**self._kwargs))

## Wraps a dataloader and cycles repeatedly
class icycle(Wrapper):
    def __init__(self,dataloader):
        super().__init__(dataloader)
    def __iter__(self):
        while True:
            for data in super().__iter__():
                yield data
    def __len__(self):
        return 10**10

# ## Wraps a dataloader and cycles repeatedly
# class icycle(object):
#     def __init__(self,dataloader):
#         self.dataloader = dataloader
#     def __iter__(self):
#         while True:
#             for data in self.dataloader:
#                 yield data
#     def __len__(self):
#         return 10**10

# class imap(object):
#     def __init__(self,func,loader):
#         self.func = func
#         self.loader = loader
#     def __iter__(self):
#         return map(self.func,self.loader.__iter__())
#     def __getattr__(self,name):
#         if name==
#         return self.loader.__getattribute__(name)
#     def __setattr__(self,name,value):
#         if name not in ['func','loader']:
#             self.loader.__setattr__(name,value)
#         else: super().__setattr__(name,value)

def minibatch_to(mb,device=None,dtype=None):
	try: return mb.to(device=device,dtype=dtype)
	except AttributeError:
		if isinstance(mb,dict):
			return type(mb)(((k,minibatch_to(v,device,dtype)) for k,v in mb.items()))
		else:
			return type(mb)(minibatch_to(elem,device,dtype) for elem in mb)
import functools
def LoaderTo(loader,device=None,dtype=None):
    return imap(functools.partial(minibatch_to,device=device,dtype=dtype),loader)

# class LoaderTo(torch.utils.data.DataLoader):
#     def __init__(self,loader, device):
#         self.__dict__ = loader.__dict__
#         self._device = device

#     def __iter__(self):
#         def minibatch_map(mb):
#             try: return mb.to(self._device)
#             except AttributeError: 
#                 return type(mb)(minibatch_map(elem) for elem in mb)#map(lambda x:x.to(self._device),mb))
#         return map(minibatch_map,super().__iter__())

# class islice(object):
#     def __init__(self,dataloader,k):
#         """ Wraps a dataloader, but only takes the first k elements with iter,
#             if shuffling is enabled, this may be different from different 
#             calls to iter """
#         self._k = k
#         self.loader = dataloader
#     def __iter__(self):
#         return iter(itertools.islice(self.loader),self._k)
#     def __getattr__(self,name):
#         return self.loader.__getattribute__(name)
#     def __setattr__(self,name,value):
#         if name not in ['_k','loader']:
#             self.loader.__setattr__(name,value)
#         else: super().__setattr__(name,value)

# class islice(torch.utils.data.DataLoader):
#     def __init__(self,dataloader,k):
#         """ Wraps a dataloader, but only takes the first k elements with iter,
#             if shuffling is enabled, this may be different from different 
#             calls to iter """
#         self.__dict__= dataloader.__dict__
#         self.dl = dataloader
#         self._k = k
#     def __iter__(self):
#         return iter(itertools.islice(self.dl,self._k))

def to_device_layer(device):
    def minibatch_map(mb):
        try: return mb.to(device)
        except AttributeError: 
            return type(mb)(map(lambda x:x.to(device),mb))
    return Expression(minibatch_map)

# def loader_to(device):
#     """Returns a function that sends dataloader output
#          to the specified device"""
#     def minibatch_map(mb):
#         try: return mb.to(device)
#         except AttributeError: 
#             return type(mb)(map(lambda x:x.to(device),mb))
#     return lambda loader: map_with_len(minibatch_map, loader)

# # Wraps a generator so that calling __iter__ multiple
# #    times produces distinct non-empty generators  
class reusable(object):
    def __init__(self, generator_constructor):
        self._gen = generator_constructor
    def __iter__(self):
        return self._gen()
    # def __len__(self):
    #     return len(self._gen())

# class islice(object):
#     def __init__(self,dataloader,k):
#         self.dataloader = dataloader
#         self.k = k
#     def __iter__(self):
#         return iter(itertools.islice(self.dataloader,self.k))
#     def __len__(self):
#         return self.k

class izip(object):
    def __init__(self,*iters):
        self.iters = iters
    def __iter__(self):
        return iter(zip(*self.iters))
    def __len__(self):
        return min(len(it) for it in self.iters)


class Eval(object):
    def __init__(self, model, on=True):
        self.model = model
        self.on = on
    def __enter__(self):
        self.training_state = self.model.training
        self.model.train(not self.on)
    def __exit__(self, *args):
        self.model.train(self.training_state)

class FixedNumpySeed(object):
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.np_rng_state = np.random.get_state()
        np.random.seed(self.seed)
        self.rand_rng_state = random.getstate()
        random.seed(self.seed)
    def __exit__(self, *args):
        np.random.set_state(self.np_rng_state)
        random.setstate(self.rand_rng_state)

class FixedPytorchSeed(object):
    def __init__(self, seed):
        self.seed = seed
    def __enter__(self):
        self.pt_rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.seed)
    def __exit__(self, *args):
        torch.random.set_rng_state(self.pt_rng_state)
	
class Expression(nn.Module):
    def __init__(self, func):
        super(Expression, self).__init__()
        self.func = func
        
    def forward(self, *args,**kwargs):
        return self.func(*args,**kwargs)



def cosLr(num_epochs,cycle_mult=1):
    if isinstance(num_epochs, collections.abc.Iterable):
        num_epochs = sum(num_epochs)
    def lrSched(epoch):
        r = cycle_mult + 1e-8
        L = num_epochs#cycle_length #base
        current_cycle = np.floor(np.log(1+(r-1)*epoch/L)/np.log(r))
        current_cycle_length = L*r**current_cycle
        cycle_iter = epoch - L*(r**current_cycle - 1)/(r-1) #(cap lr from going too low)
        cos_scale = max(.5*(1 + np.cos(np.pi*cycle_iter/current_cycle_length)),1e-3)
        return cos_scale
    return lrSched

def recursively_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursively_update(d.get(k, type(v)()), v)
        else:
            d[k] = v
    return d

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

import sys
import select
def maybe_get_input():
    """ Returns None if no enter has been pressed, otherwise the line"""
    i,o,e = select.select([sys.stdin],[],[],0.0001)
    for s in i:
        if s == sys.stdin:
            return sys.stdin.readline()
    return None
