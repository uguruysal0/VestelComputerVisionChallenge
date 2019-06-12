"""
    Bu dosya test kümesindeki imajları verilen modeller ile test zamanlı agümantasyon(TZA) uygulayarak
    her modelin katsayısı aynı olacak şekilde tahminleri ensemble yapıyor.
    En başarıyı 100 epoch ile train ettiğim, resnetx & densenet ve vgg ensemle modelleri aldı.
    TZA olarak 3 şekilde flip ve 3 açıyla rotasyon uygulandı.
    45, 135 derece gibi dik olmayan açılarla yapılan rotasyonlar ile eğitilen modeller test zamanında 
    bu açılar kullanılarak submission yapıldı ancak bu modeller daha iyi başarım elde etmedi.
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
import csv
from data import VestelTestData

from models import *

NUM_EPOCHS = 300
EXPERIMENT_OWNER = 'Ugur Uysal - ITU VISION LAB'
TRAIN_NAME = "vestel_vgg"
BATCH_SIZE = 6
DATA_DIRECTORY = './test'
MODEL_NAME = "model_130.pth"

class_to_index = {
    "bleach_with_non_chlorine": 0,
    "do_not_bleach": 1,
    "do_not_dryclean": 2,
    "do_not_tumble_dry": 3,
    "do_not_wash": 4,
    "double_bar": 5,
    "dryclean": 6,
    "low_temperature_tumble_dry": 7,
    "normal_temperature_tumble_dry": 8,
    "single_bar": 9,
    "tumble_dry": 10,
    "wash_30": 11,
    "wash_40": 12,
    "wash_60": 13,
    "wash_hand": 14,
}
index_to_class = {class_to_index[i]: i for i in class_to_index}


def transfer(image, c):
    image = image.transpose(1, 2, 0)
    angles = [0., 180.,  90., 270.]
    flips = [-1, 0, 1]
    res = []
    for a in angles:
        rotated = ndimage.rotate(image, a)
        if c == 10:
            rotated = cv2.resize(rotated, (299, 299))
        else:
            rotated = cv2.resize(rotated, (224, 224))
        for f in flips:
            res.append(cv2.flip(rotated, f).transpose(2, 0, 1))

    return res


def make_pred(models, image1, image2):
    preds = []
    for co, model in enumerate(models):
        size = 224
        if co == 10:
            size = 299
            imList = transfer(image2.view(3, 299, 299).numpy(), co)
            for image in imList:
                image_t = torch.from_numpy(image).view(1, 3, 299, 299).cuda()
                x = model(image_t)[0].cpu().view(15, 1)
                x = nn.Sigmoid()(x)
                preds.append(x.numpy())
        else:
            imList = transfer(image1.view(3, 224, 224).numpy(), co)
            for image in imList:
                image_t = torch.from_numpy(image).view(1, 3, 224, 224).cuda()
                x = model(image_t).cpu().view(15, 1)
                x = nn.Sigmoid()(x)
                preds.append(x.numpy())

    preds = np.array(preds)
    print(preds.shape)
    preds = np.mean(preds, axis=0)

    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0

    return preds


ensemble_models = [
    ("resnet_x", "model_125.pth"),
    ("vgg", "model_125.pth")
]

results = [["Id", "ImageID", "CareSymbolTag", "Predicted"]]
if __name__ == "__main__":
    validloader = data.DataLoader(
        VestelTestData(), shuffle=False, num_workers=6, pin_memory=True)

    def model_load(i): return torch.load(
        "./experiments/{}/snapshots/{}".format(i[0], i[1])).cuda()
    
    models = [model_load(i).eval() for i in ensemble_models]
    counter = 1
    with torch.no_grad():
        for i, data in enumerate(validloader):
            image1, image2, name = data
            preds = make_pred(models, image1, image2)
            for key in index_to_class:
                prediction = [
                    counter, name[0], index_to_class[key], int(preds[key])
                ]
                counter += 1
                results.append(prediction)

    with open("output.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    print("Submission created under the submissions file")
