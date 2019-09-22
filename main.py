from DataLoader.DataLoader0_ReadAnns import DataLoader0_ReadAnns
from DataLoader.DataLoader1_ReadAll import DataLoader1_ReadAll
from DataLoader.DataVis import *
from DataLoader.Helper.Helper_TargetUnpacker import *
from Calculation import Calculation
import time

def main():
    r = DataLoader0_ReadAnns()
    visG = Visualizer_Global()
    unpacker = TargetUnpacker()
    c = Calculation()

    reader = DataLoader1_ReadAll()
    s = time.time()
    reader.getDataLable()
    print(time.time() - s)


    # index = 100
    # img, objIds, isMoreThanOneObjPerGrid, counter, label  = r.getTargetAt(index)
    # bjIds, offset, bb = unpacker.unpackLabel(label)
    # print(label.shape)
    # fakebbs = np.ones_like(bb) * 5 + bb
    # iou = c.getIOU(fakebbs, bb)
    # img = visG.drawBBox(img, fakebbs,  r.getNamesFromObjIds(objIds))
    # img = visG.drawBBox(img, bb, r.getNamesFromObjIds(objIds))
    # visG.showImg(img)


if __name__ == '__main__':
    main()