import cv2
import numpy as np
import torch
import torch.nn as nn
from DataLoader.Helper.Helper_TargetUnpacker import *
from Model.Model import Model_CNN_0
from Common.TooBox import ToolBox

np.random.seed(999)
numBoxes = 1
w, h = 16 / 448.0, 16 / 448.0
N = 32

def id2oh(id):
    return [1, 0] if id == 0 else [0, 1]

def darwGrids(imgs):
    for n in range(0, N):
        img = imgs[n]
        for i in range(0, 7): # vertical lines
            startX = (i+1)*64
            startY = 0
            endX = (i+1)*64
            endY = int(img.shape[0])
            cv2.line(img, (startX, startY), (endX, endY), (255, 255, 255), 1, 1)
        for i in range(0, 7): # horiz lines
            startX = 0
            startY = (i+1)*64
            endX = int(img.shape[1])
            endY = (i+1)*64
            cv2.line(img, (startX, startY), (endX, endY), (255, 255, 255), 1, 1)
    return imgs

def makeLabel():
    label = np.zeros((N, 7, 7, numBoxes, 7))
    for n in range(0, N):
        for i in range(0, 7):
            for j in range(0, 7):
                for b in range(0, numBoxes):
                    if np.random.rand(1) < 0.5:
                        label[n, i, j, b, 0:5] = np.array([0, 0, 0, 0, 0])
                    else:
                        x = np.random.rand(1)*0.6+0.2
                        y = np.random.rand(1)*0.6+0.2
                        label[n, i, j, b, 0:5] = np.array([1, x, y, w, h])
                        id = 0 if np.random.rand(1) < 0.5 else 1
                        label[n, i, j, b, 5:7] = np.array(id2oh(id))
    return label

def unpackLable(label):
    bboxes = np.zeros((N, 49, 4))
    bboxList = []
    xxx = 0
    for n in range(0, N):
        for i in range(0, 7):
            offsetX = i * 64
            for j in range(0, 7):
                offsetY = j * 64
                for b in range(0, numBoxes):
                    if label[n, i, j, b, 0] >= 0.5:
                        width = label[n, i, j, b, 3] * 448
                        height = label[n, i, j, b, 4] * 448
                        x = offsetX + label[n, i, j, b, 1] * 64
                        y = offsetY + label[n, i, j, b, 2] * 64
                        bboxList.append([x, y, width, height])
        xxx =  np.array(bboxList).shape[0]
        bboxes[n, 0:xxx, :] = np.array(bboxList)
        bboxList = []
    return bboxes

def genImage(bboxes):
    imgs = np.zeros((N, 448, 448, 3))
    for n in range(0, N):
        for i in range(0, bboxes.shape[1]):
            info = bboxes[n, i].astype(np.int)
            x, y, = info[0], info[1]
            if x<=0 and y <=0:
                continue
            else:
                imgs[n] = cv2.circle(imgs[n], (x, y), 14, (0, 255, 0), -1)
    return imgs

def drawRect(imgs, bboxes, isGT=True):
    for n in range(0, N):
        for i in range(0, bboxes.shape[1]):
            info = bboxes[n, i].astype(np.int)
            x, y = info[0] - int(info[2]/2), info[1] - int(info[3]/2)
            if x<=0 and y <=0:
                continue
            else:
                rw, rh = x + info[2], y + info[3]
                if isGT:
                    imgs[n] = cv2.rectangle(imgs[n], (x, y), (rw, rh), (0, 0, 255), 2)
                else:
                    imgs[n] = cv2.rectangle(imgs[n], (x, y), (rw, rh), (255, 0, 0), 2)
    return imgs