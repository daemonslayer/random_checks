#!/usr/bin/env python

import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, lr_shape):
        hr_height, hr_width = hr_shape
        lr_height, lr_width = lr_shape
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((lr_height, lr_width), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = Image.fromarray(np.concatenate((img, img, img), axis=2), 'RGB')
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        # return {"lr": img_lr, "hr": img_hr}
        return (img_lr, img_hr)

    def __len__(self):
        return len(self.files)