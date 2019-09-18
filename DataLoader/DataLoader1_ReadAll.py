from DataLoader.DataLoader0_ReadAnns import *

class DataLoader1_ReadAll():
    def __init__(self):
        self.anns = DataLoader0_ReadAnns()
        self.getDataLable()
        # i = 150
        # img, bboxes, objNames, objIds = self.getResizedInfoAt(i)
        # offset, relX, relY = self.anns.getBBoxCenter(bboxes)
        # img = self.anns.drawBBoxCenter(img, offset, relX, relY, bboxes.shape[0])
        # img = self.anns.drawBBox(img, bboxes, objNames)
        # self.anns.showImg(img)

    def getDataLable(self):
        for i in range(0, self.anns.N):
            img, bboxes, objNames, objIds = self.getResizedInfoAt(i)
            offset, relX, relY = self.anns.getBBoxCenter(bboxes)
            if np.mod(i, 100) == 0:
                print(i)

    def getResizedInfoAt(self, i):
        img = self.anns.getImgAt(i)
        bboxes =self. anns.getBBoxesAt(i)
        objNames = self.anns.getObjNamesAt(i)
        objIds = self.anns.getObjIdsAt(i)
        img, bboxes = self.anns.resize(img, bboxes)
        return img, bboxes, objNames, objIds