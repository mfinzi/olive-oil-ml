import dill
from .configGenerator import sample_config
import subprocess
from slurmpy import Slurm
import collections
import pandas as pd
import torch


class Trial(object):
    """ Abstract trial class, implements init given a config and run"""
    def __init__(self,cfg):
        raise NotImplementedError
    def run(self):
        """runs the trial and yields/returns the outcome """
        raise NotImplementedError

class TrainerFit(Trial):
    """ Makes a trial object out of a oil trainer class.
        If epochs is specified as an iterable e.g. range(0,100,10)
        then the training is split into 10 epoch segments, useful
        for search pruning, hyperband, etc."""
    def __init__(self,make_trainer,epochs,cfg):
        self.trainer = make_trainer(cfg)
        self.trainer.logger.add_text('Config',cfg)
        if isinstance(epochs,collections.Iterable):
            self.epochs = epochs
        else:
            self.epochs = [epochs]
    def run(self):
        # Replace with yield semantics?
        for epoch in self.epochs:
            return self.trainer.train_to(epoch)


DEFAULT_SBATCH_SETTINGS = {
    'N':1,
    'c':4,
    'mem':12000,
    't': '24:00:00',
    'partition':'default_gpu',
    'gres':'gpu:1',
}

def flatten(d):
    """ Flattens a dictionary, ignoring outer keys. """
    out = {}
    for k,v in d.items():
        if isinstance(v,dict):
            out.update(flatten(v))
        else:
            out[k] = v
    return out

# STUDY object 
class Study(object):
    def __init__(self, make_trial, config_spec,slurm=True,slurm_config=None):
        self.make_trial = make_trial
        self.config_spec = config_spec

        #self.slurm = slurm
        #self.slurm_config=slurm_config or {**DEFAULT_SBATCH_SETTINGS}.update(slurm_config)
        self.configs = pd.DataFrame()
        self.outcomes = pd.DataFrame()

    def flat_configs(self):
        flat_cfgs = pd.DataFrame()
        for row in self.configs.apply(flatten,axis=1):
            flat_cfgs.append(row,ignore_index=True)
        return flat_cfgs

    def run(self, num_trials=3):
        for i in range(num_trials):
            cfg = sample_config(self.config_spec)
            trial = self.make_trial(cfg)
            outcome = trial.run()
            self.configs.append(cfg)
            self.outcomes.append(outcome)
            torch.save(self,"study.S",pickle_module=dill)
            # if self.slurm:
            #     serial_cfg = dill.dumps(cfg)
            #     serial_trial_maker = dill.dumps(self.trial_maker)
            #     s = Slurm(self.name+str(i),self.slurm_config)
            #     s.run('apply '+serial_mapper+' '+serial_cfg)
            # else: # Do it serially
            #     self.trial_maker(cfg)
            #subprocess.call(self.run_cmd+['apply', serial_mapper,serial_cfg,str(i)])

