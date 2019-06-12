"""
 Bu kodlar tranin olduğu kodlar. gerekli dizinleri oluşturup checkpointleri alıyor.
 Kodların çoğunluğu bir projemden taslak olarak alındı. Minimal hale getirildi bir sürü gereksiz import gözükebilir,
 onları değiştirmek ile uğraşmadım.
 Ön eleme olduğu için minimal çalışıldı. 
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
from data import VestelData
from models import *

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
TRAIN_NAME = "resnet"
random.seed(1234)

def hammingLoss(pred,labels):
  L = 15
  pred[pred>=0.5] = 1
  pred[pred<0.5] = 0
  
  BATCH_SIZE = pred.shape[0]
  pred = pred.detach().cpu().numpy().astype(np.int32)
  labels = labels.detach().cpu().numpy().astype(np.int32)
  loss = np.sum(pred^labels)/(L*BATCH_SIZE)
  return loss

def adjust_learning_rate(optimizer, e):
    """
    Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs
    """
    if (e+1) % 20 == 0:
        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.7
    return optimizer.param_groups[0]['lr']

def save_model(model, path, epoch):
    print("Saving model " + str(epoch))
    torch.save(model, path + "/snapshots/model_{}.pth".format(epoch+1))
    return

def save_train():
    """
        Create directories to save train params, results and model
        returns directory to save results and models
    """
    if not os.path.exists("./experiments"):
        os.mkdir("./experiments")

    if not os.path.exists("./experiments/{}".format(TRAIN_NAME)):
        os.mkdir("./experiments/{}".format(TRAIN_NAME))

    if not os.path.exists("./experiments/{}/snapshots".format(TRAIN_NAME)):
        os.mkdir("./experiments//{}/snapshots".format(TRAIN_NAME))

    if not os.path.exists("./experiments/{}/losscurves".format(TRAIN_NAME)):
        os.mkdir("./experiments/{}/losscurves".format(TRAIN_NAME))

    return "./experiments/{}".format(TRAIN_NAME)

def train(model, dataloader, e, criterion, optimizer):

    model.train()
    total_loss = 0
    total_iter = 0
    total_hamm_loss = 0
    for _, batch in enumerate(dataloader):

        current_batch_size = batch[0].shape[0]
        total_iter += current_batch_size

        images, label = batch
        images = images.cuda()

        label = label.float().cuda()

        label = label.view((current_batch_size, 15))
        
        preds  = model(images).view((current_batch_size, 15))
        

        optimizer.zero_grad()

        loss = criterion(preds, label)
        total_loss += (current_batch_size * loss.item())
        loss.backward()

        optimizer.step()
        batch_loss = hammingLoss(preds,label)
        total_hamm_loss +=current_batch_size*batch_loss
        print('Train -- Epoch = {},  total iter = {}, BCE loss = {}, Ham Loss ={} '.format(
            e+1, total_iter, loss.item(),batch_loss ))

    return total_loss/total_iter, total_hamm_loss/total_iter

def save_figure(losses, path, epoch):
    """
    losses-> array of loses
    path -> path for loss curves
    train_name -> name of the train will appear in title of plot
    epoch -> epoch
    labels-> array of labels(labels corresponds to losses)
    """

    plt.title("Experiment: epoch {} ".format(epoch))
    plt.plot([i for i in range(len(losses))], losses)
    plt.legend()
    plt.savefig(path+"/losscurves/{}.pdf".format("losscurve"))
    plt.close()

def main():
    """Create the model and start the training."""
    cudnn.enabled = True
    result_dir_prefix = save_train()
    # Create network
    model = ResNetClassifier()
    model.train()
    model.float()
    model.cuda()
    # MultiLabelMarginLoss
    # BCEWithLogitsLoss
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion.cuda()
    cudnn.benchmark = True
    trainloader = data.DataLoader(VestelData("./train", size=224), batch_size=24, shuffle=True, num_workers=8, pin_memory=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-4 )
    optimizer.zero_grad()
    epoch = 126
    losses = []
    h_losses = []
    for e in range(epoch):
        loss,h_loss = train(model, trainloader, e, criterion, optimizer)
        losses.append(loss)
        h_losses.append(h_loss)
        print("Epoch BCE loss {}, hamming loss {}".format(loss,h_loss))
        if (e+1) % 25 == 0:
            save_model(model, result_dir_prefix, e)

        adjust_learning_rate(optimizer, e)

        save_figure(losses, result_dir_prefix, e)

if __name__ == '__main__':
    main()
