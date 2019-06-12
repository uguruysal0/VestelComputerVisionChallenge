"""
Data loader modülü
test ve train için 2 tane basit data loader yazıldı.
Bu loaderlar gerekli transformları yapıp datayı hazır hale getiriyorlar.
Data agümantasyonu kullanıldı. 
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
from torchvision.models import resnet34

labels = {}

with open('train_target.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    c = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
        else:
            if row[1] not in labels:
                c = 0
                labels[row[1]] = np.zeros((15, 1))
                labels[row[1]][c][0] = int(row[3])
            else:
                labels[row[1]][c][0] = int(row[3])
            c += 1
            
class VestelData(data.Dataset):
    def __init__(self, path="./train", image_suffix="*.jpg",
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=True, mirror=True, size=224):

        self.path = path
        self.size = (size,size)
        if self.path[-1] != "/":
            self.path += "/"

        self.scale = scale
        self.mean = mean
        self.std = std
        self.is_mirror = mirror

        files_1 = glob.glob1(path, image_suffix)
        files_2 = glob.glob1(path, "*.jpeg")
        files_3 = glob.glob1(path, "*.JPG")

        files = files_1+files_2+files_3
        self.files = []

        for name in files:
            file_name = name[:-4]
            if file_name[-1] == '.':
                file_name = file_name[:-1]
            self.files.append({
                "img": self.path+name,
                "file_name": file_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = cv2.resize(image, self.size)
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        image = np.asarray(image, np.float32)

        label = labels[datafiles["file_name"]]
        assert datafiles["file_name"] != None

        image /= 255
        image -= np.array([0.485, 0.456, 0.406])
        image /= np.array([0.229, 0.224, 0.225])

        angles = [0., 90., 180., 270., 45., 135., 225., 315.]
        if self.is_mirror:
            if random.randint(0, 10) % 2 == 0:
                angle = angles[random.randint(0, len(angles) - 1)]
                flip = random.randint(0, 2)-1
                dst_im = image.copy()
                dst_im = cv2.flip(dst_im, flip)
                dst_im = ndimage.rotate(dst_im, angle)
                image = dst_im
                image = cv2.resize(image, self.size)

        image = image.transpose(2, 0, 1)

        return image.copy(), label.copy()


class VestelTestData(data.Dataset):
    def __init__(self, path="./test", image_suffix="*.jpg",
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], scale=True, mirror=True):

        self.path = path

        if self.path[-1] != "/":
            self.path += "/"

        self.scale = scale
        self.mean = mean
        self.std = std
        self.is_mirror = mirror

        files_1 = glob.glob1(path, image_suffix)
        files_2 = glob.glob1(path, "*.jpeg")
        files_3 = glob.glob1(path, "*.JPG")

        files = files_1+files_2+files_3
        self.files = []

        for name in files:
            file_name = name[:-4]
            if file_name[-1] == '.':
                file_name = file_name[:-1]
            self.files.append({
                "img": self.path+name,
                "file_name": file_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))

        img2 = cv2.resize(image, (299,299))
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        
        b, g, r = cv2.split(img2)
        img2 = cv2.merge([r, g, b])

        image = np.asarray(image, np.float32)
        img2 = np.asarray(img2, np.float32)
        # label = labels[datafiles["file_name"]]
        assert datafiles["file_name"] != None

        image /= 255
        image -= np.array([0.485, 0.456, 0.406])
        image /= np.array([0.229, 0.224, 0.225])
        
        img2 /= 255
        img2 -= np.array([0.485, 0.456, 0.406])
        img2 /= np.array([0.229, 0.224, 0.225])

        image = image.transpose(2, 0, 1)
        img2 = img2.transpose(2, 0, 1)

        return image.copy(), img2.copy(), datafiles["file_name"]
