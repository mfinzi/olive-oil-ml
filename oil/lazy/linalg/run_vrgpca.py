from __future__ import division
import os
import argparse
import sys
import time
import torch
import torch.nn.functional as F
import torchvision
import tabulate
import pandas as pd
import numpy as np
sys.path.append("/home/izmailovpavel/Documents/Projects/pristine-ml/")
from oil.lazy.lazy_matrix import LazyMatrix, Lazy
from oil.lazy.lazy_types import LazyAvg
from oil.utils.utils import reusable
from oil.lazy.linalg.VRmethods import GradLoader, oja_grad2,SGHA_grad2,SGD,SVRG, SGHA_grad,oja_subspace_grad
from oil.logging.lazyLogger import LazyLogger
from oil.lazy.hessian import Hessian, Fisher
import data_noaug
sys.path.append("/home/izmailovpavel/Documents/Projects/mode-geometry/")
import models
import utils


parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--transform', type=str, default='VGG', metavar='TRANSFORM',
                    help='transform name (default: VGG)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to run VR-GPCA on (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--swa', action='store_true', help='swa usage flag (default: off)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
loaders, num_classes = data_noaug.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    args.transform,
    True
)

print('Preparing model')
architecture = getattr(models, args.model)
model = architecture.base(num_classes=num_classes, **architecture.kwargs)
checkpoint = torch.load(args.ckpt)
if args.swa:
    model.load_state_dict(checkpoint["swa_state"])
else:
    model.load_state_dict(checkpoint["model_state"])
model.cuda()
model.device = next(model.parameters()).device
model.eval()

criterion = F.cross_entropy

lazy_H = Hessian(model, loaders["train"], loss=criterion)
lazy_F = Fisher(model, loaders["train"])
grads = GradLoader(SGHA_grad,[lazy_H, lazy_F])

logger = LazyLogger(**{'no_print':False, 'minPeriod':0, 'timeFrac':1})
logger.i = 0 # annoying but we will add some temporary state to keep track of step
def log(w,lr,grad):
    logger.i+=1
    with logger as do_log:
        if do_log:
            wallclocktime = time.time()
            metrics = {}
            metrics[r"$||\nabla L(w)||$"] = np.linalg.norm(grad)
            if (logger.i - 1) % 10 == 0:
                metrics["Rayleigh Quotient"] = w @ (lazy_H @ w) / (w @ (lazy_F @ w))
                print(logger.i, ":", metrics["Rayleigh Quotient"])
            else:
                metrics["Rayleigh Quotient"] = 0
            logger.add_scalars('metrics',metrics,step=logger.i)

w0 = torch.randn(lazy_H.shape[0]).cuda()
w0 /= torch.norm(w0)
lr = lambda e: .0025 if e < 65 else 0.001#*cosLr(num_epochs)(e)
num_epochs = 500
logger.scalar_frame=pd.DataFrame()
w = SVRG(grads,w0,lr,num_epochs,log)
np.save(os.path.join(args.dir, "w.npy"), w)
