import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils
import tensorboardX
import itertools
from bgan.utils import to_var_gpu, prettyPrintLog, logOneMinusSoftmax
from bgan.cnnTrainer import CnnTrainer
import copy

class GanTrainer(CnnTrainer):
    
    def __init__(self, G, *args, base_lr=2e-4, n_disc = 1, opt_constr=None, **kwargs):
        def initClosure():
            self.D = self.CNN
            self.G = G.cuda(); self.writer.add_text('ModelSpec','Generator: '+type(G).__name__)

            if opt_constr is None: 
                opt_constr = lambda params, lr: optim.Adam(params, lr, betas=(.5,.999))
            self.d_optimizer = opt_constr(self.D.parameters(),base_lr)
            self.g_optimizer = opt_constr(self.G.parameters(),base_lr)
            self.hypers.update({'n_disc':n_disc})
            self.train_iter = zip(iter(self.lab_train), iter(self.unl_train))
            self.numBatchesPerEpoch = len(self.unl_train)

            self.fixed_z = self.sampleZ(32)
        super().__init__(*args, extraInit = initClosure, **kwargs)
        
    def step(self, *data):
        # Step the Generator
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        G_loss = self.genLoss(*data)
        G_loss.backward()
        self.g_optimizer.step()
        # Step the Discriminator
        for _ in range(self.hypers['n_disc']):
            self.d_optimizer.zero_grad()
            self.g_optimizer.zero_grad()
            D_loss = self.discLoss(*data)
            D_loss.backward()
            self.d_optimizer.step()
        
    def genLoss(self, *data):
        """ generator loss is (non-saturating loss) -E[log(1-P(y=K+1|x))] """
        _, (x_unlab, _) = data
        z = self.getInputNoise(self.hypers['ul_BS'])
        fake_logits = self.D(self.G(z))
        g_losses = -1*logOneMinusSoftmax(fake_logits)[:,self.D.numClasses-1]
        return torch.mean(g_losses)

    def sampleZ(self, n_samples=1):
        return Variable(torch.randn(n_samples, self.G.z_dim).cuda())

    def discLoss(self, *data):
        (x_lab, y_lab), (x_unlab, _) = data
        # Losses for generated samples are -E[log P(y=K+1|x)]
        z = self.sampleZ(self.hypers['ul_BS'])
        fake_logits = self.D(self.G(z))
        fake_losses = -1*nn.LogSoftmax(dim=1)(fake_logits)[:,self.D.numClasses-1]
        fake_loss = torch.mean(fake_losses)
        # Losses for unlabeled samples are -E[log(1-P(y=K+1|x))]
        unl_logits = self.D(x_unlab)
        unl_losses =  -1*logOneMinusSoftmax(unl_logits)[:,self.D.numClasses-1]
        unl_loss = torch.mean(unl_losses)
        # Losses for labeled samples are -E[log(P(y=yi|x,y!=K+1))]
        if y_lab is not None: 
            lab_logits = self.D(x_lab)[:,:self.D.numClasses-1] # conditioning on not fake
            lab_loss = nn.CrossEntropyLoss()(logits,y_lab)
        else: lab_loss = 0
        # discriminator loss = l_fake + l_unl + l_lab
        return fake_losses + unl_losses + lab_losses


    def getLabeledXYonly(self, trainingData):
        """ Extract labeled data for evaluation (superclass)"""
        labeledData, unlabeledData = trainingData
        return labeledData

    def batchAccuracy(self, x, y):
        self.D.eval()
        fullLogits = self.D(x)
        # Exclude the logits for generated class in predictions
        notFakeLogits = fullLogits[:,:self.D.numClasses-1]
        predictions = notFakeLogits.max(1)[1].type_as(y)
        correct = predictions.eq(y).cpu().data.numpy().mean()
        self.D.train()
        return correct

    def logStuff(self, i, epoch, numEpochs, trainData):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """
        step = i + epoch*self.numBatchesPerEpoch
        numSteps = numEpochs*self.numBatchesPerEpoch
        if step%2000==0:
            self.metricLog['G_loss'] = self.genLoss(*trainData).cpu().data[0]
            self.metricLog['D_loss'] = self.discLoss(*trainData).cpu().data[0]
            if len(self.lab_train):
                xy_lab = self.getLabeledXYonly(trainData)
                self.metricLog['Train_Acc(Batch)'] = self.batchAccuracy(*xy_lab)
                self.metricLog['Val_acc'] = self.getDevsetAccuracy()
            #TODO: add Inception and FID
            self.writer.add_scalars('metrics', self.metricLog, step)
            prettyPrintLog(self.metricLog, epoch, numEpochs, step, numSteps)

            self.scheduleLog['lr'] = self.lr_scheduler.get_lr()[0]
            self.writer.add_scalars('schedules', self.scheduleLog, step)

            fakeImages = self.G(self.fixed_z).cpu().data
            self.writer.add_image('fake_samples', 
                    vutils.make_grid(fakeImages, normalize=True), step)
    
    def sampleImages(self, n_samples=1):
        return self.G(self.sampleZ(n_samples))

    def getInceptionScore(self):
        pass
    
    def getFIDscore(self):
        pass





        # def loss(self, *data):
        # (x_lab, y_lab), (x_unlab, _) = data
        # # decouple the batch sizes for possible adjustments
        # x_fake = self.sample(self.hypers['ul_BS'])
        # fake_logits = self.D(x_fake)
        # logSoftMax = nn.LogSoftmax(dim=1)
        # # Losses for generated samples are -log P(y=K+1|x)
        # fake_losses = -1*logSoftMax(fake_logits)[:,self.D.numClasses-1]

        # unl_losses, lab_losses = 0, 0
        # if self.mode!="fully":# Losses for unlabeled samples are -log(1-P(y=K+1|x))
        #     x_real = x_unlab; real_logits = self.D(x_real)
        #     unl_losses =  -1*logOneMinusSoftmax(real_logits)[:,self.D.numClasses-1]

        # if self.mode!="uns": # Losses for labeled samples are -log(P(y=yi|x,y!=K+1))
        #     logits = self.D(x_lab)[:,:self.D.numClasses-1] # conditioning on not fake
        #     lab_losses = nn.CrossEntropyLoss(reduce=False)(logits,y_lab)
        
        # #discriminator loss
        # d_loss = torch.mean(fake_losses + unl_losses + lab_losses)
        
        # if self.hypers['featMatch']:
        #     if self.mode=="fully": x_comp = x_lab
        #     else: x_comp = x_unlab
        #     real_features = torch.mean(self.D(x_comp, getFeatureVec=True),0)
        #     fake_features = torch.mean(self.D(x_fake, getFeatureVec=True),0)
        #     g_loss = 1000*torch.mean(torch.abs(real_features-fake_features))

        # if self.hypers['bayesianG']:
        #     g_loss += self.sghmcNoiseLoss(self.G)
        
        # if self.hypers['bayesianD']:
        #     d_loss += self.sghmcNoiseLoss(self.D)

        # return d_loss, g_loss

        # def sghmcNoiseLoss(self, model):
        # noise_std = np.sqrt(2 * (1-self.hypers['momentum']) * self.hypers['lr'])
        # std = torch.from_numpy(np.array([noise_std])).float().cuda()[0]
        # loss = 0
        # for param in model.parameters():
        #     means = torch.zeros(param.size()).cuda()
        #     n = Variable(torch.normal(means, std=std).cuda())
        #     loss += torch.sum(n * param)
        # corrected_loss = loss / (self.hypers['genObserved'] * self.hypers['lr'])
        # return corrected_loss