#!?usr/bin/env python

import os
from functools import reduce
import shutil

from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
# TODO: check if new object required per script
summary_writer = SummaryWriter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Evaluator(object):
    """docstring for Evaluator"""
    def __init__(self, config=None, data=None, models=None):
        super(Evaluator, self).__init__()
        self.config = config
        self.data = data

        self.eval_loss = 0
        self.disc_model, self.gen_model, self.feature_extractor_model = models
        self.criterion = None
        self.disc_optimizer = None
        self.gen_optimizer = None
        self.fe_optimizer = None
        ## visualization config
        # self.visualizer = None

    def setConfig(self, config):
        self.config = config
        return True

    def setData(self, data):
        seld.data = data
        return True

    def setModel(self, model):
        self.model = model
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

    def load_saved_checkpoint(self, checkpoint=None):
        if checkpoint is None:
            path = os.path.join(self.config.checkpoints['loc'], \
                    self.config.checkpoints['ckpt_fname'])
        else:
            path = os.path.join(self.config.checkpoints['loc'], checkpoint)
        torch.load(path)

        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("[#] Loaded Checkpoint '{}' (epoch {})"
            .format(self.config.checkpoints['ckpt_fname'], checkpoint['epoch']))
        return (start_epoch, best_prec1)

    def evaluate(self, epoch):
        raise NotImplementedError()


class GANEvaluator(Evaluator):
    def evaluate(self, epoch):
        if self.disc_model is None or self.gen_model is None or self.feature_extractor_model is None:
            raise ValueError('[-] Models has been provided')
        if self.config is None:
            raise ValueError('[-] No Configurations present')
        if self.criterion is None:
            raise ValueError('[-] Loss Function hasn\'t been mentioned for the model')
        if self.data is None:
            raise ValueError('[-] No Data available to train on')

        self.eval_loss = 0
        correct = 0
        # eval mode
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (lr_images, hr_images) in enumerate(self.data):
                if self.config.gpu:
                    lr_images = lr_images.to(device)
                    hr_images = hr_images.to(device)

                # Adversarial ground truths
                Tensor = torch.cuda.FloatTensor if self.config.gpu else torch.FloatTensor
                valid = Variable(Tensor(np.ones((lr_images.size(0), *self.disc_model.output_shape))), requires_grad=False)
                fake = Variable(Tensor(np.zeros((lr_images.size(0), *self.disc_model.output_shape))), requires_grad=False)

                # compute output
                output = self.disc_model(hr_images)
                self.eval_loss = self.criterion(output, valid).item()

                # get the index of the max log-probability
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(labels.view_as(pred)).sum().item()

            self.eval_loss /= len(self.data.dataset)
            summary_writer.add_scalar('eval_loss', self.eval_loss)

            print('\nEval Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                self.eval_loss, correct, len(self.data.dataset),
                100. * correct / len(self.data.dataset)))

            return self.eval_loss