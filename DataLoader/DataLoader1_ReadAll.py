from DataLoader.DataLoader0_ReadAnns import *
from DataLoader.Helper.Helper_Global2Local import Global2Local
import threading
from Common.CommonClasses import *

class DataLoader1_ReadAll():
    def __init__(self, start=0, N=1000):
        self.beginAt = start
        self.totalN = N#self.anns.N
        self.initHelper()

    def initHelper(self):
        self.anns = DataLoader0_ReadAnns()
        self.conv_g2l = Global2Local()
        self.dataLabel = np.zeros(
            (self.totalN, GridParams().numGridX, GridParams().numGridY, GridParams().numBBox, GridParams().dimFeat),
            dtype=float)
        self.imgList = np.zeros((self.totalN, 448, 448, 3), dtype=float)

    def getDataLabelFromTo(self, start, partN):
        end, N = getEnd(start, partN, self.totalN)
        for i in range(start, end):
            img, objIds, isMoreThanOneObjPerGrid, counter, label = self.anns.getTargetAt(i + self.beginAt)
            self.imgList[i] = img
            self.dataLabel[i,:] = label
            # if np.mod(i, 100) == 0:
            #     print(i)
        print("Done Reading imgs from %d to %d" %(start, end))

    def getDataLable(self):
        print("Allocating threads to read imgs")
        partN = 500
        #nThread = int(self.anns.N/partN) + 1
        nThread = getNumThread(self.totalN, partN)
        #print(nThread)
        threads=[]
        for i in range(0, nThread):
            start = i*partN
            threads.append(threading.Thread(target=self.getDataLabelFromTo, args=(start, partN)))
            threads[i].start()
            #print(i)

        for thread in threads:
            thread.join()

if __name__ == '__main__':
    from DataLoader.DataLoader0_ReadAnns import DataLoader0_ReadAnns
    from DataLoader.DataVis import *
    from DataLoader.Helper.Helper_TargetUnpacker import *
    from Calculation import Calculation
    import time

    r = DataLoader0_ReadAnns()
    visG = Visualizer_Global()
    unpacker = TargetUnpacker()
    c = Calculation()

    reader = DataLoader1_ReadAll(1000, 1000)
    s = time.time()
    reader.getDataLable()
    print(time.time() - s)

    index = 150
    img = reader.imgList[index].copy()
    label = reader.dataLabel[index]
    objIds, offset, bb = unpacker.unpackLabel(label)
    print(label.shape)
    fakebbs = np.ones_like(bb) * 5 + bb
    iou = c.getIOU(fakebbs, bb)
    img = visG.drawBBox(img, fakebbs, YOLOObjects().getNamesFromObjIds(objIds))
    img = visG.drawBBox(img, bb, YOLOObjects().getNamesFromObjIds(objIds))
    visG.showImg(img)
