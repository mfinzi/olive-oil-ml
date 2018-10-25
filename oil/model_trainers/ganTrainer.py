import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.utils as vutils
import tensorboardX
import itertools
from oil.utils import to_var_gpu, prettyPrintLog, logOneMinusSoftmax
from oil.cnnTrainer import CnnTrainer
import copy

def hinge_loss_G(fake_logits):
    return -torch.mean()

class GanTrainer(CnnTrainer):
    
    def __init__(self, G, *args, base_lr=2e-4, n_disc = 1, opt_constr=None, **kwargs):
        def initClosure():
            self.D = self.model
            self.G = G
            self.logger.add_text('ModelSpec','Generator: '+type(G).__name__)

            if opt_constr is None: 
                opt_constr = lambda params, lr: optim.Adam(params, 0.0002, betas=(.5,.999))
            self.d_optimizer = opt_constr(self.D.parameters())
            self.g_optimizer = opt_constr(self.G.parameters())
            self.hypers.update({'n_disc':n_disc})

            self.fixed_z = self.sampleZ(32)
        super().__init__(*args, extraInit = initClosure, **kwargs)
        
    def step(self, *data):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        # Step the Generator
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
        
    def G_loss(self):
        """ hinge based generator loss -E[D(G(z))] """
        z = self.sampleZ(len(self.dataloaders['train']))
        fake_logits = self.D(self.G(z))
        return -torch.mean(fake_logits)

    def sampleZ(self, n_samples=1):
        return torch.randn(n_samples, self.G.z_dim).to(self.G.device)

    def D_loss(self, *data):
        (x, _) = data
        z = self.sampleZ(len(self.dataloaders['train']))
        fake_logits = self.D(self.G(z))
        real_logits = self.D()
        return fake_losses + unl_losses + lab_losses

    def logStuff(self, i, minibatch):
        """ Handles Logging and any additional needs for subclasses,
            should have no impact on the training """
        step = i+1 + (self.epoch+1)*len(self.dataloaders['train'])

        metrics = {}
        #metrics['Train_Acc(Batch)'] = self.batchAccuracy(minibatch)
        metrics['G_loss'] = self.genLoss(*trainData).cpu().data[0]
        metrics['D_loss'] = 
        except KeyError: pass
        try: metrics['Dev_Acc'] = self.getAccuracy(self.dataloaders['dev'])
        except KeyError: pass
        self.logger.add_scalars('metrics', metrics, step)

        schedules = {}
        schedules['lr'] = self.lr_scheduler.get_lr()[0]
        self.logger.add_scalars('schedules', schedules, step)
        super().logStuff(i,minibatch)

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