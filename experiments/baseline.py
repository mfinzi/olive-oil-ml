
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial,base_cfg
from oil.tuning.args import argupdated_config
import os

config_spec = {**base_cfg,**{'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/deconv_bs_large/')}}}
Trial = simpleClassifierTrial
cutout_study = Study(Trial,argupdated_config(config_spec))
cutout_study.run()
