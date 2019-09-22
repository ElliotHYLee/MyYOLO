from DataLoader.DataLoader0_ReadAnns import DataLoader0_ReadAnns
from DataLoader.DataVis import *
from DataLoader.Helper.Helper_TargetUnpacker import *
from Calculation import Calculation
def main():
    r = DataLoader0_ReadAnns()
    index = 100
    img, res_bb, objNames, objIds = r.getResizedInfoAt(index)

    visG = Visualizer_Global()
    unpacker = TargetUnpacker()
    isMoreThanOneObjPerGrid, counter, label  = r.getTargetAt(index)
    print(label.shape)
    bjIds, offset, bb = unpacker.unpackLabel(label)

    fakebbs = np.ones_like(bb) * 5 + bb
    c = Calculation()
    iou = c.getIOU(fakebbs, bb)
    print(iou)
    img = visG.drawBBox(img, fakebbs,  r.getNamesFromObjIds(objIds))
    img = visG.drawBBox(img, bb, r.getNamesFromObjIds(objIds))
    visG.showImg(img)


if __name__ == '__main__':
    main()