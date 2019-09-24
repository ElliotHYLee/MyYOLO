from DataLoader.DataLoader1_ReadAll import DataLoader1_ReadAll
import numpy as np
from Params.GridParams import *
from DataLoader.Utils import *
from Common.CommonClasses import *

class DataLoader2_Prep(Singleton):
    def initHelper(self, start=0, N=1000,  isTrain=True):
        self.N = N
        self.reader = DataLoader1_ReadAll(start, N)
        self.reader.getDataLable()
        self.isTrain = isTrain
        self.standardize(usePrev=True)

    def standardize(self, usePrev = True):
        mean = np.mean(self.reader.imgList, axis=(0, 1, 2))
        std = np.std(self.reader.imgList, axis=(0, 1, 2))
        normPath = 'Data/Norms/' + getBranchName()
        if not self.isTrain or usePrev:
            mean = np.loadtxt(normPath + '_img_mean.txt')
            std = np.loadtxt(normPath + '_img_std.txt')
        else:
            np.savetxt(normPath + '_img_mean.txt', mean)
            np.savetxt(normPath + '_img_std.txt', std)


        # standardize imgs
        print('standardizing imgs')
        mean = mean.astype(np.float32)
        std = std.astype(np.float32)
        for i in range(0, self.reader.imgList.shape[3]):
            self.reader.imgList[:, :, :, i] = (self.reader.imgList[:, :, :, i] - mean[i]) / std[i]
        print('done standardizing imgs')

    def getImgList(self):
        return self.reader.imgList

    def getLabel(self):
        return self.reader.dataLabel

    def getImgListAt(self, i):
        return self.reader.imgList[i]

    def getLabelAt(self, i):
        return self.reader.dataLabel[i]

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

    reader = DataLoader2_Prep()
    reader.initHelper(0, 100)

    index = 10
    img = reader.reader.imgList[index].copy()
    label = reader.reader.dataLabel[index]
    objIds, offset, bb = unpacker.unpackLabel(label)
    print(label.shape)
    fakebbs = np.ones_like(bb) * 5 + bb
    iou = c.getIOU(fakebbs, bb)
    img = visG.drawBBox(img, fakebbs, YOLOObjects().getNamesFromObjIds(objIds))
    img = visG.drawBBox(img, bb, YOLOObjects().getNamesFromObjIds(objIds))
    visG.showImg(img)
