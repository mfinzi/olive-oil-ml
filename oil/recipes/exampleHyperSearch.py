import copy
from oil.recipes.simpleTrial import simpleTrial, makeTrainer
from oil.tuning.study import Study
from oil.architectures import layer13s
from oil.tuning.args import argupdated_config

if __name__=="__main__":
    config_spec = copy.deepcopy(makeTrainer.__kwdefaults__)
    config_spec.update(
        {'network':layer13s,'bs':[50,32,64],'lr':(lambda cfg: .002*cfg['bs']),
        'num_epochs':2,'net_config':{'k':[64,96]},'study_name':'example'}
    )
    config_spec = argupdated_config(config_spec)
    name = config_spec.pop('study_name')
    thestudy = Study(simpleTrial,config_spec,study_name=name,
            base_log_dir=config_spec['trainer_config'].get('log_dir',None))
    thestudy.run(3,ordered=False)
    print(thestudy.covariates())
    print(thestudy.outcomes)