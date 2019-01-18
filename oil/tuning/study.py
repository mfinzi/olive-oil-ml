import dill
from .configGenerator import sample_config, flatten_dict
import subprocess
import collections
import pandas as pd
import torch
import concurrent
import distutils.spawn
from functools import partial
from .slurmExecutor import SlurmExecutor, LocalExecutor
from ..logging.lazyLogger import LazyLogger
from ..utils.mytqdm import tqdm
import os
import __main__ as main

def slurm_available():
    return distutils.spawn.find_executable('salloc') is not None

DEFAULT_SLURM_SETTINGS = {
    'N':1,
    'c':1, # Number of cores per node
    'mem':24000, # mem specifies the maximum virtual memory which includes max GPU ram
    'time': '24:00:00',
    'partition':'default_gpu',
    'gres':'gpu:1080ti:1',
}

class Study(object):
    """ The study object allows hyperparameter search experimentation, with
        automatic trial parrallelization via Slurm if available. The trial configs
        that are run are stored in the dataframe self.configs, 
        and trial outcomes in self.outcomes. """
    def __init__(self, perform_trial, config_spec,
                    slurm_cfg={}, base_log_dir = None, study_name=None):
        self.perform_trial = perform_trial
        self.config_spec = config_spec
        slurm_settings = {**DEFAULT_SLURM_SETTINGS,**slurm_cfg}
        self.Executor = partial(SlurmExecutor,slurm_cfg=slurm_settings) \
                               if slurm_available() else LocalExecutor
        base_log_dir = base_log_dir or os.path.dirname(os.path.abspath(main.__file__))
        self.name = study_name or main.__file__[:-3]
        self.logger = LazyLogger(base_log_dir+'/'+self.name+'/')
        self.configs = pd.DataFrame()
        self.outcomes = pd.DataFrame()

    def flat_configs(self):
        """ Return a dataframe where rows are flattened versions of self.configs"""
        flat_cfgs = pd.DataFrame()
        for row in self.configs.apply(flatten_dict,axis=1):
            flat_cfgs.append(row,ignore_index=True)
        return flat_cfgs

    def run(self, num_trials, max_workers=10, new_config_spec=None):
        """ runs the study with num_trials and max_workers slurm nodes
            trials are executed in parallel by the slurm nodes, study object
            is updated and saved as results come in """
        if new_config_spec: self.config_spec=new_config_spec
        with self.Executor(max_workers) as executor:
            futures = [executor.submit(self.perform_trial,
                        sample_config(self.config_spec),i) for i in range(num_trials)]
            for j, future in enumerate(tqdm(concurrent.futures.as_completed(futures),
                                            total=len(futures),desc=self.name)):
                cfg, outcome = future.result()
                cfg_row = pd.DataFrame(flatten_dict(cfg),index=['config {}'.format(j)])
                outcome_row = outcome.iloc[-1].to_frame('outcome {}'.format(j)).T
                self.configs = self.configs.append(cfg_row)
                self.outcomes = self.outcomes.append(outcome_row)
                with pd.option_context('display.expand_frame_repr',False):
                    print(self.configs.iloc[-1:])
                    print(self.outcomes.iloc[-1:])
                self.logger.save_object(self,'study.s')

def train_trial(make_trainer,strict=False):
    """ a common trainer trial use case: make_trainer, train, return cfg and emas"""
    def _perform_trial(cfg,i):
        try:
            cfg['trainer_config']['log_dir'] += 'trial{}/'.format(i)
            trainer = make_trainer(cfg)
            trainer.logger.add_scalars('config',flatten_dict(cfg))
            outcome = trainer.train(cfg['num_epochs'])
            save_path = trainer.default_save_path(suffix='.trainer')
            torch.save(trainer,save_path,pickle_module=dill)
            cfg['saved_at']=save_path
            return cfg, outcome
        except Exception as e:
            if strict: raise
            else: return cfg, e
    return _perform_trial