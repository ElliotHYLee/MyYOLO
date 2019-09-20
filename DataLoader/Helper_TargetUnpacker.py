from Params.GridParams import GridParams
import numpy as np

class TargetUnpacker():
    def __init__(self):
        pass

    def unpackTarget(self, label):
        bbList = []
        oxyList = []
        for i in range(0, GridParams().numGridX):
            for j in range(0, GridParams().numGridY):
                for k in range(0, GridParams().numBBox):
                    flag = label[i, j, k*GridParams().numBBoxElements]
                    if flag >= 0.5:
                        start = k*GridParams().numBBoxElements + 1
                        bb = label[i, j, start:start+4]
                        oxy = np.array([i, j])
                        bbList.append(bb)
                        oxyList.append(oxy)
        bb_centers_wh = np.array(bbList)
        bb_centers_wh[:,0:2] *= GridParams().gridW
        bb_centers_wh[:,2:4] *= GridParams().imgW
        offset = GridParams().gridW*np.array(oxyList)
        bboxes = bb_centers_wh
        bboxes[:, 0] += offset[:, 0] - bboxes[:, 2]/2
        bboxes[:, 1] += offset[:, 1] - bboxes[:, 3]/2

        return offset, bboxes.astype(int)



