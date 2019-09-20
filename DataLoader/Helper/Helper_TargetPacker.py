import numpy as np
from DataLoader.Helper.Helper_Global2Local import Global2Local
from Params.GridParams import GridParams


class TargetPacker():
    def __init__(self):
        self.convG2L = Global2Local()

    def packBBoxAndObj(self, res_bb, objIds):
        res = np.zeros((GridParams().numGridX, GridParams().numGridY, GridParams().numBBoxElements * GridParams().limNumBBoxPerGrid))
        offset, relX, relY = self.convG2L.getBBoxCenter_Absolute(res_bb)
        ox, oy, label = self.convG2L.getBBoxes_Relative(offset, relX, relY, res_bb)
        counter = np.zeros((GridParams().numGridX, GridParams().numGridY), dtype=int)
        ohc = np.zeros((GridParams().numGridX, GridParams().numGridY, GridParams().numClass * GridParams().limNumBBoxPerGrid), dtype=int)
        N = res_bb.shape[0]
        for i in range(0, N):
            c = counter[ox[i], oy[i]]
            try:
                res[ox[i], oy[i], c * GridParams().numBBoxElements: (c + 1) * GridParams().numBBoxElements] = label[i]
                ohc[ox[i], oy[i], c * GridParams().numClass: (c + 1) * GridParams().numClass] = self.getOneHotCode(objIds[i])
            except:
                # more than 1 object per grid will be ignored
                pass
            counter[ox[i], oy[i]] += 1
        return counter, res, ohc

    def getOneHotCode(self, classId):
        ohc = np.zeros((100), dtype=int)
        ohc[classId] = 1
        return ohc

    def isMoreThanOneObjPerGrid(self, counter):
        for i in range(0, GridParams().numGridX):
            for j in range(0, GridParams().numGridY):
                if counter[i, j] > 1:
                    return True
        return False


