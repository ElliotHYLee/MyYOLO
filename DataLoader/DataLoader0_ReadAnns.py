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

        # for i in range(0, N):
        # i = 123
        # img = self.getImgAt(i)
        # bboxes = self.getBBoxesAt(i)
        # objNames = self.getObjNamesAt(i)
        # objIds = self.getObjIdsAt(i)
        # img, bboxes = self.resize(img, bboxes)
        # offset, relX, relY = self.getBBoxCenter(bboxes)
        # img = self.drawBBox(img, bboxes, objNames)
        # img = self.drawBBoxCenter(img, offset, relX, relY, bboxes.shape[0])
        # self.showImg(img)

    def drawBBoxCenter(self, img, offset, relX, relY, N):
        x = offset[:, 0, None] + relX
        y = offset[:, 1, None] + relY
        for i in range(0, N):
            img = cv2.circle(img, (x[i], y[i]), 4, (0, 0, 255), -1)
        return img

    def getBBoxCenter(self, bboxes):
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

        relX = x - offset[:,0, None]
        relY = y - offset[:,1, None]
        return offset, relX, relY

    def resize(self, img, bboxes, w=448, h=448):
        origW, origH = img.shape[1], img.shape[0]
        img = cv2.resize(img, (w, h))
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w / origW
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h / origH
        return img, bboxes

    def printAnnsAt(self, i):
        print(self.getImgIdAt(i))
        print(self.getObjIdsAt(i))
        print(self.getObjNamesAt(i))
        print(self.getBBoxesAt(i))

    def showImg(self, img, name='asdf'):
        cv2.imshow(name, img)
        cv2.waitKey(5000)

    def getImgAt(self, i):
        imgId = self.getImgIdAt(i)
        imgPath = getImgPath(imgId)
        img = cv2.imread(imgPath)
        return img

    def drawBBox(self, img, bboxes, objNames):
        for i in range(0, bboxes.shape[0]):
            x, y = bboxes[i, 0], bboxes[i, 1]
            x2, y2 = x + bboxes[i, 2], y + bboxes[i, 3]
            img = cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
            img = cv2.putText(img, objNames[i], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return img

    def getImgIdAt(self, i):
        return self.data[i]['imgId']

    def getObjIdsAt(self, i):
        return np.array(self.data[i]['objId'], dtype=int)

    def getBBoxesAt(self, i):
        return np.array(self.data[i]['bboxes'], dtype=int)

    def getObjNamesAt(self, i):
        return self.data[i]['objName']