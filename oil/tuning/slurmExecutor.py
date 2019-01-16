
import dill
from ast import literal_eval
import sys,os,stat
import time
import tempfile
import atexit
import subprocess
from concurrent import futures

DEFAULT_SBATCH_SETTINGS = {
    'N':1,
    'c':4,
    'mem':12000,
    'time': '24:00:00',
    'partition':'default_gpu',
    'gres':'gpu:1',
}

def kwargs_to_list(kwargs):
    return ["%s%s"%(('--'+k+'=',v) if len(k)>1
               else ('-'+k+' ',v)) for k,v in kwargs.items()]
               
def tmp_file_name(suffix=".sh"):
    t = tempfile.mktemp(dir='.',suffix=suffix)
    atexit.register(os.unlink, t)
    return t

class SlurmPoolExecutor(futures.ThreadPoolExecutor):
    def __init__(self,*args,slurm_cfg=DEFAULT_SBATCH_SETTINGS,**kwargs):
        self.slurm_cfg = slurm_cfg
        super().__init__(*args,**kwargs)

    def submit(self,fn,*args,**kwargs):
        def todo():
            with open(tmp_file_name(), 'wb+') as funcfile:
                dill.dump((fn,args,kwargs),funcfile)
            with open(tmp_file_name(), "wb+") as sh_script:
                sh_script.write(os.fsencode('#!/bin/sh\n{} slurmExecutor.py {}'\
                                                    .format(sys.executable,funcfile.name)))
                os.fchmod(sh_script.fileno(), stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
            cfg_args = kwargs_to_list(self.slurm_cfg)
            out = subprocess.check_output(['srun',*cfg_args,sh_script.name]).strip()
            with open(funcfile.name, 'rb') as funcfile:
                function_output = dill.load(funcfile)
            return function_output
        return super().submit(todo)

if __name__=='__main__':
    with open(sys.argv[1], 'rb') as funcfile:
        (fn,args,kwargs) = dill.load(funcfile)
        out = fn(*args,**kwargs)
    with open(sys.argv[1], 'wb+') as funcfile:
        dill.dump(out,funcfile)