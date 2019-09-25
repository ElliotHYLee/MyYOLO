import numpy as np
from DataLoader.Helper.Helper_Global2Local import Global2Local
from Params.GridParams import GridParams
import sys
# class TargetPacker3D():
#     def __init__(self):
#         self.convG2L = Global2Local()
#
#     def packBBoxAndObj(self, res_bb, objIds):
#         res = np.zeros((GridParams().numGridX, GridParams().numGridY, GridParams().numBBoxElements * GridParams().numBBox))
#         offset, relX, relY = self.convG2L.getBBoxCenter_Absolute(res_bb)
#         ox, oy, label = self.convG2L.getBBoxes_Relative(offset, relX, relY, res_bb)
#         counter = np.zeros((GridParams().numGridX, GridParams().numGridY), dtype=int)
#         ohc = np.zeros((GridParams().numGridX, GridParams().numGridY, GridParams().numClass * GridParams().numBBox), dtype=int)
#         N = res_bb.shape[0]
#         for i in range(0, N):
#             c = counter[ox[i], oy[i]]
#             try:
#                 if c < GridParams().limNumBBoxPerGrid:
#                     res[ox[i], oy[i], c * GridParams().numBBoxElements: (c + 1) * GridParams().numBBoxElements] = label[i]
#                     ohc[ox[i], oy[i], c * GridParams().numClass: (c + 1) * GridParams().numClass] = self.getOneHotCode(objIds[i])
#                     counter[ox[i], oy[i]] += 1
#             except:
#                 print('limNumBBoxPerGrid needs to be <= numBBox')
#                 sys.exit(-1)
#         return counter, res, ohc
#
#     def getOneHotCode(self, classId):
#         ohc = np.zeros((100), dtype=int)
#         ohc[classId] = 1
#         return ohc
#
#     def isMoreThanOneObjPerGrid(self, counter):
#         for i in range(0, GridParams().numGridX):
#             for j in range(0, GridParams().numGridY):
#                 if counter[i, j] > 1:
#                     return True
#         return False

class TargetPacker4D():
    def __init__(self):
        self.convG2L = Global2Local()

    def packBBoxAndObj(self, res_bb, objIds):
        label = np.zeros((GridParams().numGridX, GridParams().numGridY, GridParams().numBBox, 5 + GridParams().numClass))
        offset, relX, relY = self.convG2L.getBBoxCenter_Absolute(res_bb)

        ox, oy, label_box = self.convG2L.getBBoxes_Relative(offset, relX, relY, res_bb)
        #print(label_box.shape)
        counter = np.zeros((GridParams().numGridX, GridParams().numGridY), dtype=int)
        for i in range(0, res_bb.shape[0]):
            c = counter[ox[i], oy[i]]
            if c < GridParams().limNumBBoxPerGrid:
                table = np.concatenate([label_box[i,:], self.getOneHotCode(objIds[i])], axis=0)
                label[ox[i], oy[i], c, :] = table
                counter[ox[i], oy[i]] += 1
        return counter, label

    def getOneHotCode(self, classId, dim = 100):
        ohc = np.zeros((dim), dtype=int)
        ohc[classId] = 1
        return ohc

    def isMoreThanOneObjPerGrid(self, counter):
        for i in range(0, GridParams().numGridX):
            for j in range(0, GridParams().numGridY):
                if counter[i, j] > 1:
                    return True
        return False
