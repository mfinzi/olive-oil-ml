
import unittest
from oil.datasetup.datasets import CIFAR10
from oil.model_trainers.classifier import Classifier, simpleClassifierTrial
import oil.architectures.img_classifiers as models
from oil.tuning.configGenerator import sample_config


class ClassifierNetworkArchitecturesTests(unittest.TestCase):
    def test_networks_load_and_train(self):
        """ Takes 30 minutes on 1 gpu """
        configs = [{'dataset': CIFAR10, 'num_epochs':1,
                    'network': getattr(models,modelname)} for modelname in models.__all__]
        Trial = simpleClassifierTrial(strict=True)
        for cfg in configs:
            outcome = Trial(cfg)

if __name__=="__main__":
    unittest.main()