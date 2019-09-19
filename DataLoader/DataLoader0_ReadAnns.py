from DataLoader.Utils import *
import json
import cv2
import numpy as np

class DataLoader0_ReadAnns():
    def __init__(self):
        self.data = None
        with open(getTrainAnnPath()) as json_file:
            self.data = json.load(json_file)
        self.N = len(self.data)
        self.numGridX, self.numGridY = 7, 7
        self.imgW, self.imgH = 448, 448
        self.gridW = self.imgW/self.numGridX
        self.gridH = self.imgH/self.numGridY


    def getBBoxCenter(self, bboxes):
        if bboxes.shape[0] > 0:
            x, y = bboxes[:,0,None]+bboxes[:,2,None]/2, bboxes[:,1,None]+bboxes[:,3,None]/2
            offset = np.zeros((x.shape[0], 2))
            for index in range(x.shape[0]):
                for i in range(0, self.numGridX):
                    low, high = i * self.gridW, (i + 1) * self.gridW
                    if (x[index] >= low and x[index] < high):
                        offset[index, 0] = low
                        break
                for j in range(0, self.numGridY):
                    low, high = j * self.gridH, (j + 1) * self.gridH
                    if (y[index] >= low and y[index] < high):
                        offset[index, 1] = low
                        break

            relX = x - offset[:, 0, None]
            relY = y - offset[:, 1, None]
        else:
            offset = np.zeros((1, 2))
            relX  = np.zeros((1,1))
            relY = np.zeros((1,1))
        return offset, relX, relY

    def resize(self, img, bboxes, w=448, h=448):
        origW, origH = img.shape[1], img.shape[0]
        img = cv2.resize(img, (w, h))
        if bboxes.shape[0] > 0:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w / origW
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h / origH
        return img, bboxes


    def getImgAt(self, i):
        imgId = self.getImgIdAt(i)
        imgPath = getImgPath(imgId)
        img = cv2.imread(imgPath)
        return img

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