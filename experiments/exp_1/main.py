#!/usr/bin/env python

import sys
import os.path
import argparse
from argparse import RawTextHelpFormatter
from inspect import getsourcefile

import numpy as np
import yaml
import torch

import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.backends import cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from dataloader import ImageDataset
from model import Discriminator, GeneratorResNet, FeatureExtractor
from trainer import GANTrainer
from evaluator import GANEvaluator
from mapper import *

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
parent_dir = parent_dir[:parent_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
from utils import *
sys.path.pop(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):
    np.random.seed(0)
    torch.manual_seed(0)

    with open('config.yaml', 'r') as file:
    	stream = file.read()
    	config_dict = yaml.safe_load(stream)
    	config = mapper(**config_dict)

    disc_model = Discriminator(input_shape=(config.data.channels, config.data.hr_height, config.data.hr_width))
    gen_model = GeneratorResNet()
    feature_extractor_model = FeatureExtractor()
    plt.ion()

    if config.distributed:
        disc_model.to(device)
        disc_model = nn.parallel.DistributedDataParallel(disc_model)
        gen_model.to(device)
        gen_model = nn.parallel.DistributedDataParallel(gen_model)
        feature_extractor_model.to(device)
        feature_extractor_model = nn.parallel.DistributedDataParallel(feature_extractor_model)
    elif config.gpu:
        # disc_model = nn.DataParallel(disc_model).to(device)
        # gen_model = nn.DataParallel(gen_model).to(device)
        # feature_extractor_model = nn.DataParallel(feature_extractor_model).to(device)
        disc_model = disc_model.to(device)
        gen_model = gen_model.to(device)
        feature_extractor_model = feature_extractor_model.to(device)
    else: return

    train_dataset = ImageDataset(config.data.path, hr_shape=(config.data.hr_height, config.data.hr_width), lr_shape=(config.data.lr_height, config.data.lr_width))
    test_dataset = ImageDataset(config.data.path, hr_shape=(config.data.hr_height, config.data.hr_width), lr_shape=(config.data.lr_height, config.data.lr_width))

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle,
        num_workers=config.data.workers, pin_memory=config.data.pin_memory, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle,
        num_workers=config.data.workers, pin_memory=config.data.pin_memory)

    if args.train:
        # trainer settings
        trainer = GANTrainer(config.train, train_loader, (disc_model, gen_model, feature_extractor_model))
        criterion = nn.MSELoss().to(device)
        disc_optimizer = torch.optim.Adam(disc_model.parameters(), config.train.hyperparameters.lr)
        gen_optimizer = torch.optim.Adam(gen_model.parameters(), config.train.hyperparameters.lr)
        fe_optimizer = torch.optim.Adam(feature_extractor_model.parameters(), config.train.hyperparameters.lr)

        trainer.setCriterion(criterion)
        trainer.setDiscOptimizer(disc_optimizer)
        trainer.setGenOptimizer(gen_optimizer)
        trainer.setFEOptimizer(fe_optimizer)


    	# evaluator settings
        evaluator = GANEvaluator(config.evaluate, val_loader, (disc_model, gen_model, feature_extractor_model))
        # optimizer = torch.optim.Adam(disc_model.parameters(), lr=config.evaluate.hyperparameters.lr, 
        # 	weight_decay=config.evaluate.hyperparameters.weight_decay)
        evaluator.setCriterion(criterion)

    if args.test:
    	pass

    # Turn on benchmark if the input sizes don't vary
    # It is used to find best way to run models on your machine
    cudnn.benchmark = True
    start_epoch = 0
    best_precision = 0
    
    # optionally resume from a checkpoint
    if config.train.resume:
        [start_epoch, best_precision] = trainer.load_saved_checkpoint(checkpoint=None)

    # change value to test.hyperparameters on testing
    for epoch in range(start_epoch, config.train.hyperparameters.total_epochs):
        if config.distributed:
            train_sampler.set_epoch(epoch)

        if args.train:
            trainer.adjust_learning_rate(epoch)
            trainer.train(epoch)
            prec1 = evaluator.evaluate(epoch)

        if args.test:
        	pass

        # remember best prec@1 and save checkpoint
        if args.train:
            is_best = prec1 > best_precision
            best_precision = max(prec1, best_precision)
            trainer.save_checkpoint({
                'epoch': epoch+1,
                'state_dict': disc_model.state_dict(),
                'best_precision': best_precision,
                'optimizer': optimizer.state_dict(),
            }, is_best, checkpoint=None)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='', formatter_class=RawTextHelpFormatter)
	parser.add_argument('--train', type=str2bool, default='1', \
				help='Turns ON training; default=ON')
	parser.add_argument('--test', type=str2bool, default='0', \
				help='Turns ON testing; default=OFF')
	args = parser.parse_args()
	main(args)
