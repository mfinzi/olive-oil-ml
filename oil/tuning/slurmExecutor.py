
import dill
from ast import literal_eval
import sys,os,stat
import time
import tempfile
import atexit
import subprocess
from concurrent import futures
from functools import partial
import itertools
import torch
from oil.tuning.localGpuExecutor import LocalGpuExecutor

def kwargs_to_list(kwargs):
    return ["%s%s"%(('--'+k+'=',v) if len(k)>1
               else ('-'+k+' ',v)) for k,v in kwargs.items()]
               
def tmp_file_name(suffix=".sh"):
    t = tempfile.mktemp(dir='.',suffix=suffix)
    atexit.register(os.unlink, t)
    return t

class SlurmExecutor(futures.ThreadPoolExecutor):
    def __init__(self,*args,slurm_cfg={},clone_session=True,**kwargs):
        self.slurm_cfg = slurm_cfg
        # Dump the python session
        if clone_session:
            self.session_file_name = tmp_file_name(".pkl")
            dill.dump_session(self.session_file_name)
        else:
            self.session_file_name = 'no_session'
        super().__init__(*args,**kwargs)

    def submit(self,fn,*args,**kwargs):
        def todo():
            with open(tmp_file_name(), 'wb+') as funcfile:
                dill.dump((fn,args,kwargs),funcfile)
            with open(tmp_file_name(), "wb+") as sh_script:
                sh_script.write(os.fsencode('#!/bin/sh\n{} {} {} {}'\
                        .format(sys.executable,os.path.realpath(__file__),
                        funcfile.name,self.session_file_name)))
                os.fchmod(sh_script.fileno(),stat.S_IRWXU|stat.S_IRWXG|stat.S_IROTH|stat.S_IXOTH)
            cfg_args = kwargs_to_list(self.slurm_cfg)
            subprocess.call(['srun',*cfg_args,sh_script.name])
            with open(funcfile.name, 'rb') as funcfile:
                function_output = dill.load(funcfile)
            return function_output
        return super().submit(todo)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        """ Identical to the chunky ProcessPoolExecutor implementation,
            but underlying parts aren't exposed """
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")
        results = super().map(partial(_process_chunk, fn),
                              _get_chunks(*iterables, chunksize=chunksize),
                              timeout=timeout)
        return _chain_from_iterable_of_lists(results)

def _process_chunk(fn, chunk):
    return [fn(*args) for args in chunk]
def _get_chunks(*iterables, chunksize):
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk
def _chain_from_iterable_of_lists(iterable):
    for element in iterable:
        element.reverse()
        while element:
            yield element.pop()

def LocalExecutor(max_workers=None):
    if max_workers==1 or torch.cuda.device_count()<=1 or os.environ.copy().get("WORLD_SIZE",0)!=0:
        print("local")
        return futures.ThreadPoolExecutor(max_workers=1)
    else:
        return LocalGpuExecutor(max_workers)

# #LocalExecutor = LocalGpuExecutor
# class LocalExecutor(futures.ThreadPoolExecutor):
#     """Wraps ProcessPoolExecutor but distributes local gpus to the
#         processes #TODO: restrict gpu allocation. At the moment restricts
#         to sequential (single core and gpu) execution."""
#     def __init__(self,max_workers,*args,**kwargs):
#         super().__init__(max_workers=1,*args,**kwargs)
#         #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#         #os.environ["CUDA_VISIBLE_DEVICES"]="0"
        

if __name__=='__main__':
    if sys.argv[2]!='no_session':
        dill.load_session(sys.argv[2])
    with open(sys.argv[1], 'rb') as funcfile:
        (fn,args,kwargs) = dill.load(funcfile)
        out = fn(*args,**kwargs)
    with open(sys.argv[1], 'wb+') as funcfile:
        dill.dump(out,funcfile)
