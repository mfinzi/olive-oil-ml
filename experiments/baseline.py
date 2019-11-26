
from oil.model_trainers.classifier import Classifier,simpleClassifierTrial,base_cfg
from oil.tuning.args import argupdated_config
import os
from oil.tuning.study import Study
if __name__=="__main__":
    config_spec = {**base_cfg,**{'trainer_config':{'log_dir':os.path.expanduser('~/tb-experiments/supermnist_hyper/')}}}
    Trial = simpleClassifierTrial
    thestudy = Study(Trial,argupdated_config(config_spec),study_name="point2d",base_log_dir=os.path.expanduser('~/tb-experiments/supermnist_hyper/'))
    thestudy.run(ordered=False)
    covars = thestudy.covariates()#
    covars['Dev_Acc'] = thestudy.outcomes['Dev_Acc'].values
    print(covars.drop(['log_suffix'],axis=1))