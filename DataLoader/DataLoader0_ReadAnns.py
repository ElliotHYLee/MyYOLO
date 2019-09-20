from DataLoader.Utils import *
import json
import cv2
import numpy as np
from DataLoader.Helper_Global2Local import Global2Local
from DataLoader.Helper_TargetPacker import TargetPacker

class DataLoader0_ReadAnns():
    def __init__(self):
        self.data = None
        self.conv_g2l = Global2Local()
        self.packer = TargetPacker()
        with open(getTrainAnnPath()) as json_file:
            self.data = json.load(json_file)
        self.N = len(self.data)

    def getImgAt(self, i):
        imgId = self.getImgIdAt(i)
        imgPath = getImgPath(imgId)
        img = cv2.imread(imgPath)
        return img

    def getResizedInfoAt(self, i):
        img = self.getImgAt(i)
        bboxes =self. getBBoxesAt(i)
        objNames = self.getObjNamesAt(i)
        objIds = self.getObjIdsAt(i)
        img, bboxes = self.conv_g2l.resize(img, bboxes)
        return img, bboxes, objNames, objIds

    def getTargetAt(self, i):
        img = self.getImgAt(i)
        bboxes = self.getBBoxesAt(i)
        img, res_bb = self.conv_g2l.resize(img, bboxes)
        label = self.packer.packTarget(res_bb)
        return label

    def printAnnsAt(self, i):
        print(self.getImgIdAt(i))
        print(self.getObjIdsAt(i))
        print(self.getObjNamesAt(i))
        print(self.getBBoxesAt(i))

    def getImgIdAt(self, i):
        return self.data[i]['imgId']

    def getObjIdsAt(self, i):
        return np.array(self.data[i]['objId'], dtype=int)

    def getBBoxesAt(self, i):
        return np.array(self.data[i]['bboxes'], dtype=int)

    def getObjNamesAt(self, i):
        return self.data[i]['objName']