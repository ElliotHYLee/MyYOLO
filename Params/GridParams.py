class Singleton:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

class GridParams(Singleton):
    def __init__(self):
        self.numClass = 100
        self.numBBox = 1
        self.limNumBBoxPerGrid = 1
        self.numBBoxElements = 5
        self.numGridX, self.numGridY = 7, 7
        self.imgW, self.imgH = 448, 448
        self.gridW = self.imgW / self.numGridX
        self.gridH = self.imgH / self.numGridY