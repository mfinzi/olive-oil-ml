
import dill
from .configGenerator import sample_config, flatten_dict,grid_iter
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
from collections.abc import Iterable
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
    def __init__(self, perform_trial, config_spec={},
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

    def run(self, num_trials=None, max_workers=1, new_config_spec=None,ordered=True):
        """ runs the study with num_trials and max_workers slurm nodes
            trials are executed in parallel by the slurm nodes, study object
            is updated and saved as results come in """
        if new_config_spec: self.config_spec=new_config_spec
        with self.Executor(max_workers) as executor:
            start_id = len(self.configs)
            configs = grid_iter(self.config_spec,num_trials,shuffle = not ordered)
            futures = [executor.submit(self.perform_trial,
                        cfg,start_id+i) for i, cfg in enumerate(configs)]
            for j, future in enumerate(tqdm(concurrent.futures.as_completed(futures),
                                            total=len(futures),desc=self.name)):
                cfg, outcome = future.result()
                cfg_row = pd.DataFrame(flatten_dict(cfg),index=['config {}'.format(start_id+j)])
                outcome_row = outcome.iloc[-1].to_frame('outcome {}'.format(start_id+j)).T
                self.configs = self.configs.append(cfg_row)
                self.outcomes = self.outcomes.append(outcome_row)
                with pd.option_context('display.expand_frame_repr',False):
                    print(self.configs.iloc[-1:])
                    print(self.outcomes.iloc[-1:])
                save_loc = self.logger.save_object(self,'study.s')
        return save_loc
        # we could pass in a function that outputs the generator, and that function
        # can be pickled (by dill)
        # IF we save the current iteration, a generator can also be resumed,
        
                # TODO log current best? start with add_text current best 
                # & add_scalars of current best outcome
    def covariates(self):
        """ Returns the subset of columns from configs that is not all the same"""
        columns_that_vary = self.configs.apply(pd.Series.nunique,axis=0)!=1
        return self.configs.T[columns_that_vary].T

# Plan for pruning support (e.g. median rule, hyperband)
        # Support for trials that train in segments via generators
        # num_epochs can either be a number or an iteratable
        # _perform_trial will yield partial result
        # will need to extend futures as results come in
        # modify as_completed to allow mutation during iteration
        # or use concurrent.futures.wait
        # perhaps multiprocessing queue is a better abstraction

# Plan for non-random cfg selection (e.g. grid_search, bayesopt)
        # config spec will have to be generalized with SearchVariations
        # that not only can sample but have densities?
        # for grid search we just need to change the dictionary traversing order
        # sampleConfig should be replaced by a generator
        # the generator state will be updated as output information comes in
        # sample_config could be random (aka prior), but may have state dependent
        # on (partial) outcome

def train_trial(make_trainer,strict=False):
    """ a common trainer trial use case: make_trainer, train, return cfg and emas"""
    def _perform_trial(cfg,i=None):
        try:
            if i is not None:
                cfg['trainer_config']['log_suffix'] = 'trial{}/'.format(i)
            trainer = make_trainer(cfg)
            try: cfg['params(M)'] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)/10**6
            except AttributeError: pass
            trainer.logger.add_scalars('config',flatten_dict(cfg))
            epochs = cfg['num_epochs'] if isinstance(cfg['num_epochs'],Iterable) else [cfg['num_epochs']]
            for portion in epochs:
                outcome = trainer.train(portion)
                cfg['saved_at'] = trainer.logger.save_object(trainer,
                                    suffix='checkpoints/c{}.trainer'.format(trainer.epoch))
            return cfg, outcome
        except Exception as e:
            if strict: raise
            else: return cfg, e
    return _perform_trial