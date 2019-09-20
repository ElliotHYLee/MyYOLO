from DataLoader.Utils import *
from DataLoader.DataLoader0_ReadAnns import DataLoader0_ReadAnns
from DataLoader.DataLoader1_ReadAll import DataLoader1_ReadAll
from DataLoader.DataVis import *
from DataLoader.Helper_Global2Local import *
from DataLoader.Helper_TargetPacker import *


def main():
    r = DataLoader0_ReadAnns()
    index = 0
    img, res_bb, objNames, objIds = r.getResizedInfoAt(index)

    visG = Visualizer_Global()
    img = visG.drawBBox(img, res_bb, objNames)
    label = r.getTargetAt(index)
    print(label)

    unpacker = TargetUnpacker()
    offset, bboxes = unpacker.unpackTarget(label)
    # print(offset)
    # print(bboxes)
    img = visG.drawBBoxCenter(img, offset, bboxes[:,0,None], bboxes[:,1,None])
    visG.showImg(img)


if __name__ == '__main__':
    main()