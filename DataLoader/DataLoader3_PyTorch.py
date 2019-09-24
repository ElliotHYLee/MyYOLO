from torch.utils.data import Dataset, DataLoader
from DataLoader.DataLoader2_Prep import *
import numpy as np
import cv2
import time
from sklearn.utils import shuffle

class YOLODataSetManager():
    def __init__(self,start=0, N=1000, isTrain=True, split=0.2):
        data = DataLoader2_Prep()
        data.initHelper(start, N, isTrain)
        N = data.N
        idx = np.arange(0, N)
        if isTrain:
            idx = shuffle(idx)
            valN = int(N * split)
            trainN = N - valN
            trainIdx = idx[0:trainN]
            valIdx = idx[trainN:]
            self.trainSet = YOLODataSet(trainN, trainIdx)
            self.valSet = YOLODataSet(valN, valIdx)
        else:
            self.testSet = YOLODataSet(N, idx)

class YOLODataSet(Dataset):
    def __init__(self, N, idxList):
        self.dm = DataLoader2_Prep()
        self.N = N
        self.idxList = idxList

    def __getitem__(self, i):
        index = self.idxList[i]
        try:
            return self.dm.getImgListAt(index), self.dm.getLabelAt(index)
        except:
            print('this is an error @ VODataSet_CNN of VODataSet.py')
            print(self.dm.getImgList().shape)
            print(i, index)

    def __len__(self):
        return self.N

if __name__ == '__main__':
    from DataLoader.DataVis import *
    from DataLoader.Helper.Helper_TargetUnpacker import *
    unpacker = TargetUnpacker()
    visG = Visualizer_Global()
    #start = time.time()
    dm = YOLODataSetManager(start=0, N = 100, isTrain=False)
    # print(time.time() - start)
    # trainSet, valSet = dm.trainSet, dm.valSet
    dataSet = dm.testSet
    trainLoader = DataLoader(dataset=dataSet, batch_size=64)
    sum = 0
    for batch_idx, (img, label) in enumerate(trainLoader):
        img = img.data.numpy()
        label = label.data.numpy()
        sum += img.shape[0]

        for i in range(img.shape[0]):
            img_ = img[i, :]
            objIds, offset, bb = unpacker.unpackLabel(label[i])
            img_ = visG.drawBBox(img_, bb, YOLOObjects().getNamesFromObjIds(objIds))
            cv2.imshow('img_', img_)
            cv2.waitKey(1000)