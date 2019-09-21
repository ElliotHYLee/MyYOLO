from DataLoader.DataLoader0_ReadAnns import DataLoader0_ReadAnns
from DataLoader.DataVis import *
from DataLoader.Helper.Helper_TargetUnpacker import *

def main():
    r = DataLoader0_ReadAnns()
    index = 552
    img, res_bb, objNames, objIds = r.getResizedInfoAt(index)

    visG = Visualizer_Global()
    unpacker = TargetUnpacker()
    isMoreThanOneObjPerGrid, counter, label_box, label_ohc  = r.getTargetAt(index)
    objIds = unpacker.ohc2num(label_ohc)
    #print(label_box)
    offset, bb = unpacker.unpackTarget(label_box)
    print(label_box.shape)
    print(isMoreThanOneObjPerGrid)
    print(objIds)
    img = visG.drawBBox(img, bb,  r.getNamesFromObjIds(objIds))
    visG.showImg(img)


if __name__ == '__main__':
    main()