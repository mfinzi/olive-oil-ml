import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from oil.dataloaders import getUnlabLoader, getLabLoader
from oil.utils import to_var_gpu, prettyPrintLog, to_gpu
from oil.utils import Eval, FixedNumpySeed, reconstructor
from oil.mlogging import SummaryWriter
from oil.lazyLogger import logTimer
from oil.schedules import cosLr
import torch.nn.functional as F
from torch.nn.parallel.replicate import replicate
import copy, os, random
import dill

class CnnTrainer:
    metric = 'Dev_Acc'

    def __init__(self, CNN, datasets, opt_constr=None, lr_sched = lambda e: 1,
                save_dir=None, load_path=None,
                lab_BS=50, ul_BS=50, amntLab=1, amntDev=0,
                num_workers=0, log=True, description='',
                extraInit=lambda:None, dataseed=0, rebuildable=True):

        # Magic method for capturing a closure with this init function
        self.reconstructor = reconstructor() if rebuildable else None
        
        # Setup tensorboard logger
        self.save_dir = save_dir
        self.writer = SummaryWriter(save_dir, log)
        self.metricLog = {}
        self.scheduleLog = {}

        # Setup Network and Optimizers
        assert torch.cuda.is_available(), "CUDA or GPU not working"
        self.CNN = CNN.cuda()
        self.writer.add_text('ModelSpec','CNN: '+type(CNN).__name__)
        if opt_constr is None: opt_constr = optim.Adam
        self.optimizer = opt_constr(self.CNN.parameters())
        try: self.lr_scheduler = lr_sched(self.optimizer)
        except: self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,lr_sched)

        # Setup Dataloaders and Iterators
        trainset, testset = datasets
        extraArgs = {'num_workers':num_workers}
        with FixedNumpySeed(dataseed): # For the random train-eval/lab-unlab split
            self.unl_train = getUnlabLoader(trainset, ul_BS, **extraArgs)
            self.lab_train, self.dev = \
                getLabLoader(trainset,lab_BS,amntLab,amntDev,**extraArgs)
        self.test = DataLoader(testset, batch_size=50, **extraArgs)
        self.train_iter = iter(self.lab_train)#self.getTrainIter()
        self.numBatchesPerEpoch = len(self.lab_train)

        # Init hyperparameter dictionary
        self.hypers = {'amntLab':amntLab, 'amntDev':amntDev,
                        'lab_BS':lab_BS, 'ul_BS':ul_BS, 'description':description}
        # Extra work to do (used for subclass)
        extraInit()
        # Load checkpoint if specified
        if load_path: self.load_checkpoint(load_path) 
        else: self.epoch = 0
        # Log the hyper parameters
        for tag, value in self.hypers.items():
            self.writer.add_text('ModelSpec',tag+' = '+str(value))
        
    #def getTrainIter(self): return iter(self.lab_train)

    def train(self, numEpochs=100):
        """ The main training loop called (also for subclasses)"""
        startEpoch = self.epoch
        for epoch in range(startEpoch, numEpochs):
            self.lr_scheduler.step(epoch)
            for i in range(self.numBatchesPerEpoch):
                trainData = to_gpu(next(self.train_iter))
                self.step(*trainData)
                self.maybeLogStuff(i, epoch, numEpochs+startEpoch, trainData)
            self.epoch = epoch+1
        #self.logStuff(i, epoch, numEpochs, trainData)

    def step(self, *data):
        self.optimizer.zero_grad()
        loss = self.loss(*data)
        loss.backward()
        self.optimizer.step()

    def loss(self, *data):
        """ Basic cross entropy loss { -E[log(P(y=yi|x))] } """
        x, y = data
        loss = nn.CrossEntropyLoss()(self.CNN(x),y)
        return loss

    def maybeLogStuff(self, *args):
        """ Uses logTimer to determine whether enough time has passed"""
        try: self.maybeLog
        except AttributeError: self.maybeLog = logTimer()
        with self.maybeLog as performLog:
            if performLog: self.logStuff(*args)

    def logStuff(self, i, epoch, numEpochs, trainData):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """
        step = i+1 + (epoch+1)*self.numBatchesPerEpoch
        numSteps = numEpochs*self.numBatchesPerEpoch
        xy_labeled = self.getLabeledXYonly(trainData)
        self.metricLog['Train_Acc(Batch)'] = self.batchAccuracy(*xy_labeled)
        self.metricLog['Test_Acc'] = self.getAccuracy(self.test)
        if len(self.dev): self.metricLog['Dev_Acc'] = self.getAccuracy(self.dev)
        self.writer.add_scalars('metrics', self.metricLog, step)
        self.scheduleLog['lr'] = self.lr_scheduler.get_lr()[0]
        self.writer.add_scalars('schedules', self.scheduleLog, step)
        prettyPrintLog(self.writer.emas(), epoch+1, numEpochs, step, numSteps)

    def getLabeledXYonly(self, trainData):
        """ should return a tuple (x,y) that will be used to calc acc 
            subclasses should override this method """
        return trainData

    def batchAccuracy(self, *labeledData, model = None):
        if model is None: model = self.CNN
        with Eval(model), torch.no_grad():
            x, y = labeledData
            predictions = model(x).max(1)[1].type_as(y)
            correct = predictions.eq(y).cpu().data.numpy().mean()
        return correct
    
    def getAccuracy(self, loader=None, model = None):
        if loader is None: loader = self.test
        numCorrect, numTotal = 0, 0
        for xy in loader:
            xy = to_gpu(xy)
            numCorrect += self.batchAccuracy(*xy, model=model)*xy[1].size(0)
            numTotal += xy[1].size(0)
        return numCorrect/numTotal

    def getMetric(self):
        assert len(self.dev)!=0, "No devset to evaluate on"
        return self.getAccuracy(self.dev)


    def get_state(self):
        state = {
            'epoch':self.epoch,
            'model_state':self.CNN.state_dict(),
            'optim_state':self.optimizer.state_dict(),
            'logger_state':self.writer.state_dict(),
            'reconstructor':self.reconstructor,
        }
        return state

    def load_state(self,state):
        self.epoch = state['epoch']
        self.CNN.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optim_state'])
        self.writer.load_state_dict(state['logger_state'])
        self.reconstructor = state['reconstructor']

    def save_checkpoint(self, save_path = None):
        if save_path is None: 
            checkpoint_save_dir = self.save_dir + 'checkpoints/'
            save_path = checkpoint_save_dir + 'c.{}.ckpt'.format(self.epoch)
            #save_path_all = checkpoint_save_dir + 'c.{}.dump'.format(self.epoch)
        else:
            checkpoint_save_dir = os.path.dirname(save_path)
          # so that we use the same subset of data as before
        os.makedirs(checkpoint_save_dir, exist_ok=True)
        torch.save(self.get_state(), save_path, pickle_module=dill)
        # if dump: ## Uses dill to pickle the entire trainer and save it
        #     with open(save_path_all, 'wb') as f:
        #         dill.dump(self,f)         
    def load_checkpoint(self, load_path):
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            state = torch.load(load_path, pickle_module=dill)
            self.load_state(state)
        else:
            print("=> no checkpoint found at '{}'".format(load_path))









    ### Methods For performing SWA after regular training is done ####

    def constSWA(self, numEpochs, lr=1e-3, period=1):
        """ runs Stochastic Weight Averaging for numEpochs epochs using const lr"""
        # Set the new constant learning rate (scaling)
        swa_lr_lambda = lambda epoch: lr/self.optimizer.defaults['lr']
        # update SWA after period epochs
        startEpoch = self.epoch+1
        update_condition = lambda epoch: (epoch-startEpoch)%period==0
        self.genericSWA(numEpochs, swa_lr_lambda, update_condition)

    def cosineSWA(self, numEpochs, lr=1e-2, period=10):
        """ runs Stochastic Weight Averaging for numEpochs epochs using cosine lr """
        # Set the new cosine cycle learning rate (scaling)
        swa_lr_lambda = lambda epoch: cosLr(period)(epoch)*lr/self.optimizer.defaults['lr']
        # Only update at the minimum of a cycle
        update_condition = lambda epoch: swa_lr_lambda(epoch+1) > swa_lr_lambda(epoch)
        self.genericSWA(numEpochs, swa_lr_lambda, update_condition)

    def genericSWA(self, numEpochs, swa_lr_lambda, update_condition):
        self.initSWA()
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,swa_lr_lambda)
        startEpoch = self.epoch+1
        for epoch in range(startEpoch, numEpochs+startEpoch):
            self.lr_scheduler.step(epoch); self.epoch = epoch
            for i in range(self.numBatchesPerEpoch):
                trainData = to_gpu(next(self.train_iter))
                self.step(*trainData)
                self.logStuff(i, epoch, numEpochs+startEpoch, trainData)
                self.swaLogStuff(i, epoch)
            if update_condition(epoch): self.updateSWA()
        self.updateBatchNorm(self.SWA)
        self.CNN.load_state_dict(self.SWA.state_dict())

    def initSWA(self):
        self.SWAupdates = 0
        try: self.SWA
        except AttributeError:
            self.SWA = replicate(self.CNN,[0])[0]
            #set_trainable(self.SWA, False)

    def updateSWA(self):
        n = self.SWAupdates
        for param1, param2 in zip(self.SWA.parameters(),self.CNN.parameters()):
            param1.data = param1.data*n/(n+1) + param2.data/(n+1)
        self.SWAupdates += 1

    def swaLogStuff(self, i, epoch):
        step = i + epoch*self.numBatchesPerEpoch
        if step%2000==0:
            self.updateBatchNorm(self.SWA)   
            self.metricLog['SWA_Test_Acc'] = self.getAccuracy(self.test, model=self.SWA)
            if len(self.dev): 
                self.metricLog['SWA_Dev_Acc'] = self.getAccuracy(self.dev, model=self.SWA)

    def updateBatchNorm(self, model):
        with torch.no_grad():
            for _ in range(self.numBatchesPerEpoch):
                trainData = to_gpu(next(self.train_iter))
                out = model(self.getLabeledXYonly(trainData)[0])


