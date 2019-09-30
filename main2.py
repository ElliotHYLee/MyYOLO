import cv2
import numpy as np
import torch
import torch.nn as nn
from DataLoader.Helper.Helper_TargetUnpacker import *
from Model.Model import Model_CNN_0

np.random.seed(0)
numBoxes = 1
w, h = 16 / 448.0, 16 / 448.0

def id2oh(id):
    return [1, 0] if id == 0 else [0, 1]

def makeLabel():

    label = np.zeros((2, 7, 7, numBoxes, 7))
    for n in range(0, 2):
        for i in range(0, 7):
            for j in range(0, 7):
                for b in range(0, numBoxes):
                    x = np.random.rand(1)
                    y = np.random.rand(1)
                    label[n, i, j, b, 0:5] = np.array([1, x, y, w, h])
                    #label[n, i, j, b, 5:7] = np.array([1, x, y, w, h])
                    id = 0 if np.random.rand(1) < 0.5 else 1
                    label[n, i, j, b, 5:7] = np.array(id2oh(id))
    return label

def unpackLable(label):
    bboxList = []
    for n in range(0, 2):
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
    return np.array(bboxList)

def genImage(label):
    img = np.zeros((448, 448, 3))
    for i in range(0, bboxes.shape[0]):
        info = bboxes[i].astype(np.int)
        x, y, = info[0], info[1]
        img = cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
    return img

def drawRect(img, bboxes):
    for i in range(0, bboxes.shape[0]):
        info = bboxes[i].astype(np.int)
        x, y = info[0] - int(info[2]/2), info[1] - int(info[3]/2)
        rw, rh = x + info[2], y + info[3]
        img = cv2.rectangle(img, (x, y), (rw, rh), (0, 0, 255), 2)
    return img

label = makeLabel()
bboxes = unpackLable(label)
img = genImage(label)
img = drawRect(img, bboxes)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model_CNN_0()
model = nn.DataParallel(model).to(device)
model.train()


cv2.imshow('asdf', img)
cv2.waitKey(10000)


