import numpy as np
import torch, torchvision
import torch.optim as optim
import torch.nn as nn
import os, copy
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import ray.tune.result
#from ray.tune.variant_generator import to_argv
from ray.tune import TrainingResult,register_trainable, run_experiments
from ray.tune.async_hyperband import AsyncHyperBandScheduler
from ray.tune.trial_scheduler import FIFOScheduler

#import oil.augLayers as augLayers
from oil.cnnTrainer import CnnTrainer
from oil.datasets import CIFAR10, C10augLayers
from oil.networkparts import layer13
from oil.schedules import cosLr


# def randchoice(array,probs=None):
#     return lambda spec: np.random.choice(array,p=probs)

def to_str(dic):
    return "".join(["_{}-{:.3g}".format(key,value) for key,value in dic.items()][:10])

def combineDicts(dicts):
    out = {}
    for dic in dicts:
        out.update(dic)
    return out

def makeTunable(configs, makeTrainer, train_unit):
    base_configs = copy.deepcopy(configs)
    class Tunable(ray.tune.Trainable):
        trialNum = 0
        def _setup(self):
            savedir_base = combineDicts(base_configs)['save_dir']
            self.config['save_dir'] = savedir_base+"_{}".format(type(self).trialNum) + to_str(self.config)
            type(self).trialNum+=1
            for key, value in self.config.items():
                hit = False
                for config_dict in base_configs:
                    if key in config_dict: 
                        config_dict[key]=value
                        hit = True
                assert hit, "tunable hyper not in config dicts"

            self.trainer = makeTrainer(*base_configs)
            self.metric = type(self.trainer).metric
            

        def _train(self):
            self.trainer.train(train_unit)
            val = self.trainer.writer.emas()[self.metric]
            epoch = self.trainer.epoch
            outcome = TrainingResult(mean_accuracy=val,timesteps_this_iter=train_unit)#timesteps_total=epoch)
            return outcome

        def _save(self, checkpoint_dir):
            path = os.path.join(checkpoint_dir, "checkpoint")
            self.trainer.save_checkpoint(path)
            return path

        def _restore(self, checkpoint_path):
            self.trainer.load_checkpoint(checkpoint_path)
        def _stop(self):
            # self.emas = self.trainer.writer.emas()
            # self.metric_value = self.emas[self.metric]
            # self.epoch = self.trainer.epoch
            del self.trainer
            #torch.cuda.empty_cache()
    return Tunable

def getBestHypers(trials):
    # values = np.array([trial.runner.metric_value for trial in trials])
    # configs = [trial.runner.config for trial in trials]
    values = np.array([trial.last_result.mean_accuracy for trial in trials])
    configs = [trial.config for trial in trials]
    best = np.argmax(values)
    bestHypers = configs[best]
    bestValue = values[best]
    return bestHypers, bestValue


def tune(makeTrainer,base_configs,tunable_hypers,train_unit,numTrials=1,
        scheduler=FIFOScheduler(), local_dir='/scratch/maf388/ray_results/'):
    ngpus = torch.cuda.device_count()
    print("Setting up ray server on {} gpus".format(ngpus))
    ray.init(num_gpus=ngpus)
    combinedDict = combineDicts(base_configs)
    tunable = makeTunable(base_configs, makeTrainer, train_unit)
    register_trainable("tunable",tunable)
    expt = {combinedDict['expt_name']:{
                "run":"tunable",
                "local_dir":local_dir,
                "repeat":numTrials,
                "trial_resources":{"cpu":4,"gpu":1},
                "stop":{"timesteps_total":combinedDict['numEpochs']},
                "config":tunable_hypers}}
    trials = run_experiments(expt,scheduler=scheduler)
    bestHypers, bestValue = getBestHypers(trials)
    print("Best trial achieved metric: {} \n with hyperparams {}".format(bestValue,to_str(bestHypers)))
    torch.save(trials,combinedDict['save_dir']+"tuning_expt.trials")
    return trials
    

# class SmokeTunable(ray.tune.Trainable):
#     def _setup(self):
#         self.mean = self.config['lab_BS']/50
#         self.traincount=0
#         print(self.config['save_dir'])
#     def _train(self):
#         time.sleep(2)
#         self.traincount+=1
#         value = self.traincount*self.mean + 2*np.random.randn()
#         return TrainingResult(mean_validation_accuracy=value,timesteps_this_iter=1)
#     def _save(self, checkpoint_dir):
#         path = os.path.join(checkpoint_dir, "checkpoint")
#         torch.save([self.mean, self.traincount],path)
#         return path
#     def _restore(self, checkpoint_path):
#         self.mean, self.traincount = torch.load(checkpoint_path)