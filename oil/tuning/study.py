
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
from ..utils.utils import maybe_get_input
import os
import numpy as np
from collections.abc import Iterable
import __main__ as main
import logging

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

    def process_user_input(self,executor):
        inp = maybe_get_input()
        if inp is None: return
        if inp[:4]=="halt":pass
        elif inp[:3]=="add":
            if hasattr(executor,"add_gpus"): executor.add_gpus(inp[4:-1])
        elif inp[:2]=="rm":
            if hasattr(executor,"remove_gpus"): executor.remove_gpus(inp[3:-1])
        elif inp[:4]=="best":self.print_best_sofar()
        #print(inp)
        #else:pass

#    def process_result(self,result):
        # cfg,outcome = result
        # cfg_row = pd.DataFrame(flatten_dict(cfg),index=['config {}'.format(start_id+j)])
        # outcome_row = outcome.iloc[-1].to_frame('outcome {}'.format(start_id+j)).T
        # self.configs = self.configs.append(cfg_row)
        # self.outcomes = self.outcomes.append(outcome_row)
        # with pd.option_context('display.expand_frame_repr',False):
        #     print(self.configs.iloc[-1:])
        #     print(self.outcomes.iloc[-1:])
        # self.save_loc = self.logger.save_object(self,'study.s')

    def run(self, num_trials=-1, max_workers=None, new_config_spec=None,ordered=True):
        """ runs the study with num_trials and max_workers slurm nodes
            trials are executed in parallel by the slurm nodes, study object
            is updated and saved as results come in. A results df is pickled as results.df,
            if num_trials is a negative integer, then num_trials is interpreted as -num_trials
            full passes through the grid."""
        if new_config_spec: self.config_spec=new_config_spec
        with self.Executor(max_workers) as executor:
            start_id = len(self.configs)
            configs = grid_iter(self.config_spec,num_trials,shuffle = not ordered)
            futures = {executor.submit(self.perform_trial,
                        cfg,start_id+i) for i, cfg in enumerate(configs)}
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
                self.logger.save_object(self.results_df(),'results.df')
                self.save_loc = self.logger.save_object(self,'study.s')
        return self.save_loc
        # we could pass in a function that outputs the generator, and that function
        # can be pickled (by dill)
        # IF we save the current iteration, a generator can also be resumed,
        
                # TODO log current best? start with add_text current best 
                # & add_scalars of current best outcome
    def covariates(self):
        """ Returns the subset of columns from configs that is not all the same"""
        columns_that_vary = self.configs.apply(pd.Series.nunique,axis=0)!=1
        return self.configs.T[columns_that_vary].T

    def results_df(self):
        outcomes_df = self.outcomes.reset_index(drop=True)
        covariates_df = self.covariates().reset_index(drop=True)
        combined_df = pd.concat([covariates_df,outcomes_df],axis=1)
        return combined_df

    #TODO: add argument for objective when multiple possible exist, compute for each dataset?
    def print_best_sofar(self):
        validation_set = self.outcomes.columns.str.contains('Dev')
        if not np.any(validation_set):
            validation_set = self.outcomes.columns.str.contains('Test')
        if not np.any(validation_set):
            return
        objective = self.outcomes.columns[validation_set][0]
        imax = np.argmax(self.outcomes[objective].to_numpy())
        print("="*33+" Best So Far: "+"="*33)
        if not self.covariates().empty:
            print(self.covariates().iloc[imax:imax+1])
        print(self.outcomes.iloc[imax:imax+1])
        print("="*80)

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

def cleanup_cuda():
    torch.cuda.empty_cache()
    if os.environ.copy().get("WORLD_SIZE",0)!=0:
        torch.distributed.destroy_process_group()

def guess_metric_sign(metric_name):
    """ Returns true if metric should be maximized and false for minimize"""
    name = metric_name.lower()
    positive_metrics = ['acc','map','precision','score','reward']
    negative_metrics = ['mse','mae','loss','bpd','nll','err']
    if any(substring in name for substring in positive_metrics):
        return True
    elif any(substring in name for substring in negative_metrics):
        return False
    else:
        assert False, "Unknown sign for early stopping metric "+name

class train_trial(object):
    """ Assumes trainer is an object of type Trainer, trains for num_epochs which may be an
        integer or an iterable containing intermediate points at which to save.
        Pulls out special (resume, save, early_stop_metric, local_rank) args from the cfg """
    def __init__(self,make_trainer,strict=True):
        self.make_trainer = make_trainer
        self.strict=strict
    def __call__(self,cfg,i=None):
        try:
            cfg.pop('local_rank',None) #TODO: properly handle distributed
            resume = cfg.pop('resume',False)
            save = cfg.pop('save',False)
            levels = {'critical': logging.CRITICAL,'error': logging.ERROR,
                        'warn': logging.WARNING,'warning': logging.WARNING,
                        'info': logging.INFO,'debug': logging.DEBUG}
            logging.getLogger().setLevel(levels[cfg.pop('log_level','warn')])
            if i is not None:
                orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
                cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
            trainer = self.make_trainer(**cfg)
            try: cfg['params(M)'] = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)/10**6
            except AttributeError: pass
            trainer.logger.add_scalars('config',flatten_dict(cfg))
            epochs = cfg['num_epochs'] if isinstance(cfg['num_epochs'],Iterable) else [cfg['num_epochs']]
            if resume: trainer.load_checkpoint(None if resume==True else resume)
            epochs = [e for e in epochs if e>trainer.epoch]
            for epoch in epochs:
                trainer.train_to(epoch)
                if save: cfg['saved_at']=trainer.save_checkpoint()
            outcome = trainer.ckpt['outcome']
        except Exception as e:
            if self.strict: raise
            outcome = e
        cleanup_cuda()
        del trainer
        return cfg, outcome
