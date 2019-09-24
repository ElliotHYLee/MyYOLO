from Common.CommonClasses import Singleton
class GridParams(Singleton):
    def __init__(self):
        self.numClass = 100
        self.numBBox = 5
        self.dimFeat = self.numBBox + self.numClass
        self.limNumBBoxPerGrid = 1
        self.numBBoxElements = 5
        self.numGridX, self.numGridY = 7, 7
        self.imgW, self.imgH = 448, 448
        self.gridW = self.imgW / self.numGridX
        self.gridH = self.imgH / self.numGridY
