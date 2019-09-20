import numpy as np
from DataLoader.Helper_Global2Local import Global2Local

class TargetPacker():
    def __init__(self):
        self.convG2L = Global2Local()

    def packTarget(self, res_bb):
        res = np.zeros((7, 7, 5 * 10))
        offset, relX, relY = self.convG2L.getBBoxCenter_Absolute(res_bb)
        ox, oy, label = self.convG2L.getBBoxes_Relative(offset, relX, relY, res_bb)
        counter = np.zeros((7, 7, 1), dtype=int)
        N = res_bb.shape[0]
        for i in range(0, N):
            c = counter[ox[i, 0], oy[i, 0], 0]
            res[ox[i, 0], oy[i, 0], c * 5 : (c + 1) * 5] = label[i]
            counter[ox[i, 0], oy[i, 0], 0] += 1
        return res

class TargetUnpacker():
    def __init__(self):
        pass

    def unpackTarget(self, label):
        bbList = []
        oxyList = []
        for i in range(0, 7):
            for j in range(0, 7):
                for k in range(0, 10):
                    flag = label[i, j, k*5]
                    if flag >= 0.4:
                        start = k*5 + 1
                        bb = label[i, j, start:start+4]
                        oxy = np.array([i, j])
                        bbList.append(bb)
                        oxyList.append(oxy)
        bboxes = np.array(bbList)
        bboxes[:,0:2] *= 64
        bboxes[:,2:4] *= 448
        offset = 64*np.array(oxyList)
        return offset, bboxes.astype(int)



