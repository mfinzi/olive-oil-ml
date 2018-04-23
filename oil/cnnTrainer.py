import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from oil.dataloaders import getUnlabLoader, getLabLoader
from oil.utils import to_var_gpu, prettyPrintLog
from oil.logging import SummaryWriter
import torch.nn.functional as F
from torch.nn.parallel.replicate import replicate
import copy, os

class CnnTrainer:
    
    def __init__(self, CNN, datasets, save_dir=None, load_path=None,
                base_lr=2e-4, lab_BS=50, ul_BS=50, amntLab=1, amntDev=0,
                num_workers=0, opt_constr=None, log=True,
                extraInit=lambda:None, lr_lambda = lambda e: 1):

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
        self.optimizer = opt_constr(self.CNN.parameters(), base_lr)
        self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda)

        # Setup Dataloaders and Iterators
        trainset, testset = datasets
        extraArgs = {'num_workers':num_workers}
        self.unl_train = getUnlabLoader(trainset, ul_BS, **extraArgs)
        self.lab_train, self.dev = \
            getLabLoader(trainset,lab_BS,amntLab,amntDev,**extraArgs)
        self.test = DataLoader(testset, batch_size=50, **extraArgs)
        self.train_iter = iter(self.lab_train)
        self.numBatchesPerEpoch = len(self.lab_train)

        # Init hyperparameter dictionary
        self.hypers = {'base_lr':base_lr, 'amntLab':amntLab, 'amntDev':amntDev,
                        'lab_BS':lab_BS, 'ul_BS':ul_BS}
        # Extra work to do (used for subclass)
        extraInit()
        # Load checkpoint if specified
        if load_path: self.load_checkpoint(load_path) 
        else: self.epoch = 0
        # Log the hyper parameters
        for tag, value in self.hypers.items():
            self.writer.add_text('ModelSpec',tag+' = '+str(value))

    def train(self, numEpochs=100):
        """ The main training loop called (also for subclasses)"""
        for epoch in range(self.epoch, numEpochs):
            self.lr_scheduler.step(epoch); self.epoch = epoch
            for i in range(self.numBatchesPerEpoch):
                trainData = to_var_gpu(next(self.train_iter))
                self.step(*trainData)
                self.logStuff(i, epoch, numEpochs, trainData)

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

    def logStuff(self, i, epoch, numEpochs, trainData):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """
        step = i + epoch*self.numBatchesPerEpoch
        numSteps = numEpochs*self.numBatchesPerEpoch
        if step%2000==0:
            xy_labeled = self.getLabeledXYonly(trainData)
            self.metricLog['Train_Acc(Batch)'] = self.batchAccuracy(*xy_labeled)
            self.metricLog['Test_Acc'] = self.getAccuracy(self.test)
            if len(self.dev): self.metricLog['Dev_Acc'] = self.getAccuracy(self.dev)
            self.writer.add_scalars('metrics', self.metricLog, step)
            prettyPrintLog(self.metricLog, epoch, numEpochs, step, numSteps)

            self.scheduleLog['lr'] = self.lr_scheduler.get_lr()[0]
            self.writer.add_scalars('schedules', self.scheduleLog, step)

    def getLabeledXYonly(self, trainData):
        """ should return a tuple (x,y) that will be used to calc acc 
            subclasses should override this method """
        return trainData

    def batchAccuracy(self, *labeledData, model = None):
        if model is None: model = self.CNN
        x, y = labeledData
        model.eval()
        predictions = model(x).max(1)[1].type_as(y)
        correct = predictions.eq(y).cpu().data.numpy().mean()
        model.train()
        return correct
    
    def getAccuracy(self, loader=None, model = None):
        if loader is None: loader = self.test
        numCorrect, numTotal = 0, 0
        for xy in loader:
            xy = to_var_gpu(xy, volatile=True)
            numCorrect += self.batchAccuracy(*xy, model=model)*xy[1].size(0)
            numTotal += xy[1].size(0)
        return numCorrect/numTotal

    def save_checkpoint(self, save_dir = None):
        if save_dir is None: save_dir = self.save_dir
        save_path = filepath = save_dir + 'checkpoints/'
        os.makedirs(save_path, exist_ok=True)
        filepath = save_path + 'c.{}.ckpt'.format(self.epoch)
        state = {
            'epoch':self.epoch+1,
            'model_state':self.CNN.state_dict(),
            'optim_state':self.optimizer.state_dict(),
            'lab_sampler':self.lab_train.batch_sampler,
            'dev_sampler':self.dev.batch_sampler,
        } # Saving the sampler for the labeled dataset is crucial
          # so that we use the same subset of data as before
        torch.save(state, filepath)

    def load_checkpoint(self, load_path):
        if os.path.isfile(load_path):
            print("=> loading checkpoint '{}'".format(load_path))
            state = torch.load(load_path)
            self.epoch = state['epoch']
            self.CNN.load_state_dict(state['model_state'])
            self.optimizer.load_state_dict(state['optim_state'])
            self.lab_train.batch_sampler = state['lab_sampler']
            try: self.dev.batch_sampler = state['dev_sampler']
            except: print("Older checkpoint, dev loader may be inconsistent")
        else:
            print("=> no checkpoint found at '{}'".format(load_path))









    ### Methods For performing SWA after regular training is done ####

    def constSWA(self, numEpochs, lr=1e-3, period=1):
        """ runs Stochastic Weight Averaging for numEpochs epochs using const lr"""
        # Set the new constant learning rate (scaling)
        swa_lr_lambda = lambda epoch: lr/self.hypers['base_lr']
        # update SWA after period epochs
        startEpoch = self.epoch+1
        update_condition = lambda epoch: (epoch-startEpoch)%period==0
        self.genericSWA(numEpochs, swa_lr_lambda, update_condition)

    def cosineSWA(self, numEpochs, lr=1e-2, period=10):
        """ runs Stochastic Weight Averaging for numEpochs epochs using cosine lr """
        # Set the new cosine cycle learning rate (scaling)
        swa_lr_lambda = lambda epoch: cosLr(period)(epoch)*lr/self.hypers['base_lr']
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
                trainData = to_var_gpu(next(self.train_iter))
                self.step(*trainData)
                self.swaLogStuff(i, epoch)
                self.logStuff(i, epoch, numEpochs+startEpoch, trainData)
            if update_condition(epoch): self.updateSWA()
        self.updateBatchNorm(self.SWA)
        self.CNN.load_state_dict(self.SWA.state_dict())

    def initSWA(self):
        try: self.SWA
        except AttributeError:
            self.SWAupdates = 0
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
        model.train()
        for _ in range(self.numBatchesPerEpoch):
            tensors = next(self.train_iter)
            trainData = to_var_gpu(tensors, volatile=True)
            out = model(self.getLabeledXYonly(trainData)[0])


