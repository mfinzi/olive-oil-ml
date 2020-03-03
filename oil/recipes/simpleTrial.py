import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from oil.utils.utils import LoaderTo, cosLr, islice, export
from oil.tuning.study import train_trial
from oil.datasetup.datasets import CIFAR10, split_dataset
from oil.architectures.img_classifiers import layer13
from oil.utils.parallel import try_multigpu_parallelize
from oil.tuning.args import argupdated_config
from oil.model_trainers.classifier import Classifier
from functools import partial

def makeTrainer(*,dataset=CIFAR10,network=layer13,num_epochs=100,
                bs=50,lr=.1,aug=True,optim=SGD,device='cuda',trainer=Classifier,
                split={'train':-1,'val':.1},net_config={},opt_config={},
                trainer_config={'log_dir':None},save=False):

    # Prep the datasets splits, model, and dataloaders
    datasets = split_dataset(dataset(f'~/datasets/{dataset}/'),splits=split)
    datasets['test'] = dataset(f'~/datasets/{dataset}/', train=False)

    device = torch.device(device)
    model = network(num_targets=datasets['train'].num_targets,**net_config).to(device)
    if aug: model = torch.nn.Sequential(datasets['train'].default_aug_layers(),model)
    model,bs = try_multigpu_parallelize(model,bs)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    # Add some extra defaults if SGD is chosen
    if optim==SGD: opt_config={**{'momentum':.9,'weight_decay':1e-4,'nesterov':True},**opt_config}
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)
    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)

simpleTrial = train_trial(makeTrainer)
if __name__=='__main__':
    simpleTrial(argupdated_config(makeTrainer.__kwdefaults__))
