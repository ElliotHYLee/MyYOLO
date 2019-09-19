from DataLoader.DataLoader0_ReadAnns import *
import threading


class DataLoader1_ReadAll():
    def __init__(self):
        self.anns = DataLoader0_ReadAnns()
        self.label = np.zeros((self.anns.N, 19, 19, 100), dtype=float)
        self.getDataLable()

    def getDataLabelFromTo(self, start, partN):
        end, N = getEnd(start, partN, self.anns.N)
        for i in range(start, end):
            img, bboxes, objNames, objIds = self.getResizedInfoAt(i)
            offset, relX, relY = self.anns.getBBoxCenter(bboxes)
            if np.mod(i, 1000) == 0:
                print(i)

    def getDataLable(self):
        dataLabel = np.zeros((self.anns.N, 7, 7, 100), dtype=float)
        partN = 5000
        nThread = int(self.anns.N/partN + 1)
        threads=[]
        for i in range(0, nThread):
            start = i*partN
            threads.append(threading.Thread(target=self.getDataLabelFromTo, args=(start, partN)))
            threads[i].start()

        for thread in threads:
            thread.join()

    def getResizedInfoAt(self, i):
        img = self.anns.getImgAt(i)
        bboxes =self. anns.getBBoxesAt(i)
        objNames = self.anns.getObjNamesAt(i)
        objIds = self.anns.getObjIdsAt(i)
        img, bboxes = self.anns.resize(img, bboxes)
        return img, bboxes, objNames, objIds

