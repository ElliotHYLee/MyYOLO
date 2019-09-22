from DataLoader.DataLoader0_ReadAnns import *
from DataLoader.Helper.Helper_Global2Local import Global2Local
import threading


class DataLoader1_ReadAll():
    def __init__(self):
        self.anns = DataLoader0_ReadAnns()
        self.conv_g2l = Global2Local()
        self.dataLabel = np.zeros(
            (self.anns.N, GridParams().numGridX, GridParams().numGridY, GridParams().numBBox, GridParams().dimFeat),
            dtype=float)
        self.totalN = 10000#self.anns.N

    def getDataLabelFromTo(self, start, partN):
        end, N = getEnd(start, partN, self.totalN)
        for i in range(start, end):
            img, objIds, isMoreThanOneObjPerGrid, counter, label = self.anns.getTargetAt(i)
            self.dataLabel[i,:] = label
            if np.mod(i, 100) == 0:
                print(i)

    def getDataLable(self):
        partN = 500
        #nThread = int(self.anns.N/partN) + 1
        nThread = getNumThread(self.totalN, partN)
        print(nThread)
        threads=[]
        for i in range(0, nThread):
            start = i*partN
            threads.append(threading.Thread(target=self.getDataLabelFromTo, args=(start, partN)))
            threads[i].start()
            #print(i)

        for thread in threads:
            thread.join()



