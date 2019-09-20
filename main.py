from DataLoader.Utils import *
from DataLoader.DataLoader0_ReadAnns import DataLoader0_ReadAnns
from DataLoader.DataLoader1_ReadAll import DataLoader1_ReadAll
from DataLoader.DataVis import *
from DataLoader.Helper_Global2Local import *
from DataLoader.Helper_TargetPacker import *
from DataLoader.Helper_TargetUnpacker import *

def main():
    r = DataLoader0_ReadAnns()
    index = 0
    img, res_bb, objNames, objIds = r.getResizedInfoAt(index)

    visG = Visualizer_Global()
    #img = visG.drawBBox(img, res_bb, objNames)
    counter, label = r.getTargetAt(index)
    print(counter)

    unpacker = TargetUnpacker()
    offset, bb = unpacker.unpackTarget(label)
    # print(offset)
    # print(bboxes)
    #img = visG.drawBBoxCenter(img, offset, bb_centers_wh[:,0,None], bb_centers_wh[:,1,None])
    img = visG.drawBBox(img, bb, objNames)
    visG.showImg(img)


if __name__ == '__main__':
    main()