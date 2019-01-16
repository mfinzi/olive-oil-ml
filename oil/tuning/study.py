import dill
from .configGenerator import sample_config, flatten
import subprocess
from slurmpy import Slurm
import collections
import pandas as pd
import torch
import concurrent
from functools import partial
from oil.tuning.slurmExecutor import SlurmExecutor, LocalExecutor

DEFAULT_SLURM_SETTINGS = {
    'N':1,
    'c':2,
    'mem':12000,
    't': '24:00:00',
    'partition':'default_gpu',
    'gres':'gpu:1',
}

# STUDY object 
class Study(object):
    def __init__(self, perform_trial, config_spec,
                        slurm=True, slurm_config={}, study_name=None):
        self.perform_trial = perform_trial
        self.config_spec = config_spec

        slurm_cfg = {**DEFAULT_SLURM_SETTINGS}.update(slurm_config)
        self.Executor = partial(SlurmExecutor,slurm_cfg=slurm_cfg) \
                                if slurm else LocalExecutor
        self.configs = pd.DataFrame()
        self.outcomes = pd.DataFrame()

    def flat_configs(self):
        flat_cfgs = pd.DataFrame()
        for row in self.configs.apply(flatten,axis=1):
            flat_cfgs.append(row,ignore_index=True)
        return flat_cfgs

    def run(self, num_trials, max_workers=10, new_config_spec=None):
        if new_config_spec: self.config_spec=new_config_spec
        with self.Executor(max_workers) as executor:
            futures = [executor.submit(self.perform_trial,
                        sample_config(self.config_spec)) for _ in range(num_trials)]
            for future in concurrent.futures.as_completed(futures):
                cfg, outcome = future.result()
                self.configs.append(cfg,ignore_index=True)
                self.outcomes.append(outcome,ignore_index=True)
                torch.save(self,"study.s",pickle_module=dill)

def train_trial(make_trainer,epochs):
    """ a common trainer trial use case """
    def _perform_trial(cfg):
        trainer = make_trainer(cfg)
        trainer.logger.add_scalars('Config',flatten(cfg))
        outcome = trainer.train(epochs)
        save_path = trainer.default_save_path(suffix='.trainer')
        torch.save(trainer,save_path,pickle_module=dill)
        outcome['saved_at']=save_path
        return cfg, outcome
    return _perform_trial