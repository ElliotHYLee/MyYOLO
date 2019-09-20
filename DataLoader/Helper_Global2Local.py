import cv2
import numpy as np

class Global2Local():
    def __init__(self):
        self.numGridX, self.numGridY = 7, 7
        self.imgW, self.imgH = 448, 448
        self.gridW = self.imgW / self.numGridX
        self.gridH = self.imgH / self.numGridY

    def getBBoxCenter_Absolute(self, bboxes):
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
            bboxes = self.getBBoxesLocal_Absolute(origW, origH, bboxes, w, h)
        return img, bboxes

    def getBBoxesLocal_Absolute(self, origW, origH, bboxes, w=448, h=448):
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * w / origW
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * h / origH
        return bboxes

    def getBBoxes_Relative(self, offset, relX, relY, bboxes):
        label = np.zeros((bboxes.shape[0], 5))
        ox, oy = np.zeros(((bboxes.shape[0], 1)), dtype=int), np.zeros(((bboxes.shape[0], 1)), dtype=int)
        for i in range(0, bboxes.shape[0]):
            x = int(offset[i, 0] / self.gridW)
            y = int(offset[i, 1] / self.gridH)

            w = bboxes[i, 2]/self.imgW
            h = bboxes[i, 3]/self.imgH

            ox[i] = x
            oy[i] = y
            label[i,:] = np.array([1, relX[i] / 64, relY[i] / 64, w, h])
        return ox, oy, label