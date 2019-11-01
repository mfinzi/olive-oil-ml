
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial,base_cfg
from oil.tuning.args import argupdated_config
import os
from oil.tuning.study import Study
if __name__=="__main__":
    config_spec = {**base_cfg,**{'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/modelnet_hyper/')}}}
    Trial = simpleClassifierTrial
    cutout_study = Study(Trial,argupdated_config(config_spec),study_name="point3d",base_log_dir=os.path.expanduser('~/tb-experiments/modelnet_hyper/'))
    cutout_study.run()
