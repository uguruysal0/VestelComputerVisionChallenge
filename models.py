"""
    Bu modül pretrained modellerin prediction layerler değiştirilerek bu modeller oluşturruldu.
    PyTOrch v1.1 kullanıldı resnetx gibi modeller bu versiyonda geldi ve bu versiyonda bazı katmanlar
    eski versiyonla  uyumlu değiller.
"""
import argparse
import json
import datetime
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import os
import os.path as osp
from threading import Thread
import torchvision
import random
import timeit
import logging
import glob
import csv
from scipy import ndimage

from torchvision.models import resnext50_32x4d, resnet34, vgg16, squeezenet1_0, densenet121, inception_v3


def init_weights(m):
    print("init")
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal(m.weight)
        # m.bias.data.fill_(0.01)
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_normal(m.weight)
        # m.bias.data.fill_(0.01)

class ResNetClassifier(nn.Module):
    def __init__(self, n_classes=15):
        super(ResNetClassifier, self).__init__()
        used_model = resnet34(True)
        self.firstConv = nn.Sequential(
            *list(used_model.children())[:4]
        )
        self.down1 = used_model.layer1
        self.down2 = used_model.layer2
        self.down3 = used_model.layer3
        self.down4 = used_model.layer4

        self.predLayer = nn.Sequential(
            nn.Linear(512*7*7, 256*7*7, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256*7*7),
            nn.Linear(256*7*7, 256*7*7, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256*7*7),
            nn.Linear(256*7*7, n_classes, bias=False),
        )

        self.predLayer.apply(init_weights)


    def forward(self, x):
        x = self.firstConv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = x.view((x.shape[0], 512*7*7))
        x = self.predLayer(x)
        return x

class VGGClassifier(nn.Module):
    def __init__(self, n_classes=15):
        super(VGGClassifier, self).__init__()

        used_model = vgg16(True)
        self.net = nn.Sequential(
            *list(used_model.children())[:2]
        )

        self.net2 = nn.Sequential(
            nn.Linear(512*7*7, 256*7*7, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(256*7*7),
            nn.Linear(256*7*7, 128*7*7, bias=False),
            nn.ReLU(),
            nn.BatchNorm1d(128*7*7),
            nn.Linear(128*7*7, n_classes, bias=False),
        )

        self.net2.apply(init_weights)


    def forward(self, x):
        x = self.net(x)
        x = x.view((x.shape[0], 512*7*7))
        x = self.net2(x)
        return x

class SqueezeNetClassifier(nn.Module):
    def __init__(self, n_classes=15):
        super(SqueezeNetClassifier, self).__init__()        
        self.model = squeezenet1_0(pretrained=True)
        self.model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1,1), stride=(1,1))
        self.model.num_classes = n_classes
        self.model.classifier[1].apply(init_weights)
        
    def forward(self, x):
        x = self.model(x)
        return x

class ResNetXClassifier(nn.Module):
    def __init__(self, n_classes=15):
        super(ResNetXClassifier, self).__init__()        
        self.model = resnext50_32x4d(True)
        self.model.fc = nn.Linear(2048, n_classes)
        # self.model.fc(init_weights)
        
    def forward(self, x):
        x = self.model(x)
        return x

class InceptionClassifier(nn.Module):
    def __init__(self, n_classes=15):
        super(InceptionClassifier, self).__init__()        
        self.model = inception_v3(pretrained=True)
        
        num_ftrs = self.model.AuxLogits.fc.in_features
        self.model.AuxLogits.fc = nn.Linear(num_ftrs, n_classes)
        
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs,n_classes)
        self.model.fc.apply(init_weights)
        self.model.AuxLogits.fc.apply(init_weights)

    def forward(self, x):
        x = self.model(x)
        return x

class DenseNetClassifier(nn.Module):
    def __init__(self, n_classes=15):
        super(DenseNetClassifier, self).__init__()        
        self.model = densenet121(pretrained=True)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, n_classes)
        self.model.classifier.apply(init_weights)
        
    def forward(self, x):
        x = self.model(x)
        return x

           