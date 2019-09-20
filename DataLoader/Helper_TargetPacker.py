import numpy as np
from DataLoader.Helper_Global2Local import Global2Local
from Params.GridParams import GridParams
import sys
class TargetPacker():
    def __init__(self):
        self.convG2L = Global2Local()

    def packTarget(self, res_bb):
        counter, label  = self.packBBox(res_bb)
        return counter, label

    def packObjIds(self, objIds):
        N = objIds.shape[0]
        och = np.zeros((GridParams().numGridX, GridParams().numGridY, GridParams().numClass))
        for i in range(0, N):
            pass

    def packBBox(self, res_bb):
        res = np.zeros((GridParams().numGridX, GridParams().numGridY, GridParams().numBBoxElements * GridParams().numBBox))
        offset, relX, relY = self.convG2L.getBBoxCenter_Absolute(res_bb)
        ox, oy, label = self.convG2L.getBBoxes_Relative(offset, relX, relY, res_bb)
        counter = np.zeros((GridParams().numGridX, GridParams().numGridY), dtype=int)
        N = res_bb.shape[0]
        for i in range(0, N):
            c = counter[ox[i, 0], oy[i, 0]]
            try:
                res[ox[i, 0], oy[i, 0], c * GridParams().numBBoxElements: (c + 1) * GridParams().numBBoxElements] = label[i]
            except:
                print('Error: Helper_TargetPacker in packBBox()')
                sys.exit('Probably, increase the number of bounding boxes.')

            counter[ox[i, 0], oy[i, 0]] += 1
        return counter, res

    def getOneHotCode(self, classId):
        ohc = np.zeros((1, 100))
        ohc[classId] = 1
        return ohc


