import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from oil.model_trainers.gan import Gan
from oil.architectures.img_gen import resnetgan
from oil.utils.utils import LoaderTo, cosLr, islice, dmap,export
from oil.tuning.study import train_trial
from oil.datasetup.datasets import CIFAR10
from oil.utils.parallel import try_multigpu_parallelize
from oil.tuning.args import argupdated_config
from functools import partial
from torchvision import transforms

def makeTrainer(*,gen=resnetgan.Generator,disc=resnetgan.Discriminator,
                num_epochs=500,dataset=CIFAR10,bs=64,lr=2e-4,
                device='cuda',net_config={},opt_config={'betas':(.5,.999)},
                trainer_config={'n_disc':5,'log_dir':None},save=False):

    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
    # Prep the datasets splits, model, and dataloaders
    datasets = {}
    datasets['train'] = dmap(lambda mb: mb[0],dataset(f'~/datasets/{dataset}/',transform=transform))
    datasets['test'] = dmap(lambda mb: mb[0],dataset(f'~/datasets/{dataset}/',train=False,transform=transform))

    device = torch.device(device)
    G = gen(**net_config).to(device)
    D = disc(**net_config).to(device)
    G,_ = try_multigpu_parallelize(G,bs)
    D,bs = try_multigpu_parallelize(D,bs)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    opt_constr = partial(Adam, lr=lr, **opt_config)
    lr_sched = lambda e:1
    return Gan(G,dataloaders,opt_constr,lr_sched,D=D,**trainer_config)

GanTrial = train_trial(makeTrainer)
if __name__=='__main__':
    GanTrial(argupdated_config(makeTrainer.__kwdefaults__))