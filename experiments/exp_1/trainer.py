#!/usr/bin/env python

import os
from functools import reduce
import shutil

from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# TODO: check if new object required per script
summary_writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(object):
    """docstring for Trainer"""
    def __init__(self, config=None, data=None, models=None):
        super(Trainer, self).__init__()
        self.config = config
        self.data = data

        self.train_loss = 0
        self.criterion = None
        self.disc_optimizer = None
        self.gen_optimizer = None
        self.fe_optimizer = None
        self.curr_lr = 0
        self.start_epoch = 0
        self.best_precision = 0
        self.disc_model, self.gen_model, self.feature_extractor_model = models

    def setConfig(self, config):
        self.config = config
        return True

    def setData(self, data):
        self.data = data
        return True

    def setModel(self, model):
        self.model = model
        self.count_parameters()
        return True

    def setCriterion(self, criterion):
        self.criterion = criterion
        return True

    def setDiscOptimizer(self, optimizer):
        self.disc_optimizer = optimizer
        return True

    def setGenOptimizer(self, optimizer):
        self.gen_optimizer = optimizer
        return True

    def setFEOptimizer(self, optimizer):
        self.fe_optimizer = optimizer
        return True

    def count_parameters(self):
        if self.model is None:
            raise ValueError('[-] No model has been provided')

        self.trainable_parameters = sum(reduce( lambda a, b: a*b, x.size()) for x in self.model.parameters())

    def getTrainableParameters(self):
        if self.model is not None and self.trainable_parameters == 0:
            self.count_parameters()

        return self.trainable_parameters

    def save_checkpoint(self, state, is_best, checkpoint=None):
        if not os.path.exists(self.config.checkpoints.loc):
            os.makedirs(self.config.checkpoints.loc)
        if checkpoint is None:
            ckpt_path = os.path.join(self.config.checkpoints.loc, self.config.checkpoints.ckpt_fname)
        else:
            ckpt_path = os.path.join(self.config.checkpoints.loc, checkpoint)
        best_ckpt_path = os.path.join(self.config.checkpoints.loc, \
                            self.config.checkpoints.best_ckpt_fname)
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copy(ckpt_path, best_ckpt_path)

    def load_saved_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            path = os.path.join(self.config.checkpoints.loc, \
                    self.config.checkpoints.ckpt_fname)
            checkpoint = torch.load(path)
        else:
            path = os.path.join(self.config.checkpoints.loc, checkpoint)

        self.start_epoch = checkpoint['epoch']
        self.best_precision = checkpoint['best_precision']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("[#] Loaded Checkpoint '{}' (epoch {})"
            .format(self.config.checkpoints.ckpt_fname, self.start_epoch))
        return (self.start_epoch, self.best_precision)

    def adjust_learning_rate(self, epoch: int):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        self.curr_lr = self.config.hyperparameters.lr * (self.config.hyperparameters.lr_decay ** (epoch // self.config.hyperparameters.lr_decay_epoch))
        for param_group in self.disc_optimizer.param_groups:
            param_group['lr'] = self.curr_lr

    def train(self, epoch: int):
        raise NotImplementedError()


class GANTrainer(Trainer):
    def train(self, epoch: int):
        if self.disc_model is None or self.gen_model is None or self.feature_extractor_model is None:
            raise ValueError('[-] Models has been provided')
        if self.config is None:
            raise ValueError('[-] No Configurations present')
        if self.criterion is None:
            raise ValueError('[-] Loss Function hasn\'t been mentioned for the model')
        if self.disc_optimizer is None or self.gen_optimizer is None or self.fe_optimizer is None:
            raise ValueError('[-] Optimizers haven\'t been mentioned for the models')
        if self.data is None:
            raise ValueError('[-] No Data available to train on')
        

        self.train_loss = 0
        self.disc_model.train()
        self.gen_model.train()
        self.feature_extractor_model.train()
        
        for batch_idx, (lr_images, hr_images) in enumerate(self.data):
            if self.config.gpu:
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

            # Adversarial ground truths
            Tensor = torch.cuda.FloatTensor if self.config.gpu else torch.FloatTensor
            valid = Variable(Tensor(np.ones((lr_images.size(0), *self.disc_model.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((lr_images.size(0), *self.disc_model.output_shape))), requires_grad=False)

            output = self.disc_model(hr_images)
            print('Shape of Valid tensor: {}'.format(valid.shape))
            print('Shape of Output tensor: {}'.format(output.shape))
            loss = self.criterion(output, valid)

            self.disc_optimizer.zero_grad()
            loss.backward()
            self.train_loss = loss.item()
            self.disc_optimizer.step()

            summary_writer.add_scalar('train_loss', loss.item())
            if batch_idx % self.config.logs.log_interval == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f}'.format(
                    epoch+1, batch_idx * len(self.data), len(self.data.dataset),
                    100. * batch_idx / len(self.data),
                    loss.item() / len(self.data), self.curr_lr)
                )

        # self.visualizer.add_values(epoch, loss_train=self.train_loss)
        # self.visualizer.redraw()
        # self.visualizer.block()