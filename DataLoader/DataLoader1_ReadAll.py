from DataLoader.DataLoader0_ReadAnns import *
from DataLoader.Helper_Global2Local import Global2Local
import threading


class DataLoader1_ReadAll():
    def __init__(self):
        self.anns = DataLoader0_ReadAnns()
        self.conv_g2l = Global2Local()
        self.getDataLable()

    def getDataLabelFromTo(self, start, partN):
        end, N = getEnd(start, partN, self.anns.N)
        for i in range(start, end):
            img, bboxes, objNames, objIds = self.anns.getResizedInfoAt(i)
            offset, relX, relY = self.conv_g2l.getBBoxCenter_Absolute(bboxes)

            if np.mod(i, 100) == 0:
                print(i)

    def getDataLable(self):
        self.dataLabel = np.zeros((self.anns.N, 7, 7, 5), dtype=float)
        partN = 500
        #nThread = int(self.anns.N/partN) + 1
        nThread = int(500 / partN) + 1
        print(nThread)
        threads=[]
        for i in range(0, nThread):
            start = i*partN
            threads.append(threading.Thread(target=self.getDataLabelFromTo, args=(start, partN)))
            threads[i].start()
            print(i)

        for thread in threads:
            thread.join()



