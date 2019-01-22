


import unittest
from oil.datasetup.datasets import SVHN,CIFAR10,CIFAR100
from oil.model_trainers.classifier import Classifier, simpleClassifierTrial
import oil.architectures.img_classifiers as models
from oil.tuning.configGenerator import sample_config, logUniform
from oil.tuning.study import Study
import torch
import dill

class StudyTests(unittest.TestCase):
    def test_study_generates_configs(self):
        """ Takes 30 minutes on 1 gpu """
        config_spec = {'dataset': [CIFAR10,CIFAR100], 'num_epochs':1,
                    'network': models.layer13s,
                    'loader_config': {'amnt_dev':5000,'lab_BS':[16,32,64]},
                    'opt_config':{'lr':logUniform(.03,.3), 
                                    'momentum':lambda cfg: 1/(1-cfg['opt_config']['lr']), 
                                    'weight_decay':lambda cfg: cfg['opt_config']['momentum']*1e-4},
                    'trainer_config':{'log_args':{'no_print':True}},
                    }
        
        Trial = simpleClassifierTrial(strict=True)
        cutout_study = Study(Trial,config_spec, slurm_cfg={'time':'00:10:00'})
        save_loc = cutout_study.run(num_trials=2,max_workers=1)
        study = torch.load(save_loc,pickle_module=dill)
        assert isinstance(study,Study)

if __name__=="__main__":
    unittest.main()