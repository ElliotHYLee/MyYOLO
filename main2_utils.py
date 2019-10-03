import cv2
import numpy as np
import torch
import torch.nn as nn
from DataLoader.Helper.Helper_TargetUnpacker import *
from Model.Model import Model_CNN_0
from Common.TooBox import ToolBox

np.random.seed(999)
numBoxes = 2
imgW = 448
imgH = imgW
numGrid = 7
gridWH = int(imgW / numGrid)
bboxW, bboxH = 16 / imgW, 16 / imgH
numClass = 20
featureDim = 2*5 + numClass

class LabelMaker():
    def __init__(self, N):
        self.N = N

    def id2oh(self, id):
        result = np.zeros((numClass))
        result[id] = 1
        return result

    def drawGrids(self, imgs):
        for n in range(0, imgs.shape[0]):
            img = imgs[n]
            for i in range(0, numGrid): # vertical lines
                startX = (i+1)*gridWH
                startY = 0
                endX = (i+1)*gridWH
                endY = int(img.shape[0])
                cv2.line(img, (startX, startY), (endX, endY), (255, 255, 255), 1, 1)
            for i in range(0, numGrid): # horiz lines
                startX = 0
                startY = (i+1)*gridWH
                endX = int(img.shape[1])
                endY = (i+1)*gridWH
                cv2.line(img, (startX, startY), (endX, endY), (255, 255, 255), 1, 1)
        return imgs

    def makeLabel(self):
        label = np.zeros((self.N, numGrid, numGrid, featureDim))
        for n in range(0, self.N):
            for i in range(0, numGrid):
                for j in range(0, numGrid):
                    if np.random.rand(1) >= 0.5:
                        feature = np.zeros((30))
                        x = np.random.rand(1) * 0.8 + 0.1
                        y = np.random.rand(1) * 0.8 + 0.1
                        feature[0:5]  = np.array([1, x, y, bboxW, bboxH])
                        id = 0 if np.random.rand(1) < 0.5 else 1
                        feature[10:] = self.id2oh(id)
                        label[n, i, j] = feature
        return label

    def unpackLable(self, label):
        bboxes = np.zeros((self.N, numGrid * numGrid, 4))
        bboxList = []
        xxx = 0
        for n in range(0, label.shape[0]):
            for i in range(0, numGrid):
                offsetX = i * 64
                for j in range(0, numGrid):
                    offsetY = j * 64
                    if label[n, i, j, 0] >= 0.5:
                        width = label[n, i, j, 3] * imgW
                        height = label[n, i, j, 4] * imgH
                        x = offsetX + label[n, i, j, 1] * gridWH
                        y = offsetY + label[n, i, j, 2] * gridWH
                        bboxList.append([x, y, width, height])
            xxx =  np.array(bboxList).shape[0]
            bboxes[n, 0:xxx, :] = np.array(bboxList)
            bboxList = []
        return bboxes

    def genImage(self, bboxes):
        imgs = np.zeros((self.N, imgW, imgH, 3))
        for n in range(0, self.N):
            for i in range(0, bboxes.shape[1]):
                info = bboxes[n, i].astype(np.int)
                x, y, = info[0], info[1]
                if x<=0 and y <=0:
                    continue
                else:
                    imgs[n] = cv2.circle(imgs[n], (x, y), 7, (0, 255, 0), -1)
        return imgs

    def drawRect(self, imgs, bboxes, isGT=True):
        for n in range(0, imgs.shape[0]):
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