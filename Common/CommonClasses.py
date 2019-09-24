from DataLoader.Utils import *
import pandas as pd
class Singleton:
    __instance = None
    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super(Singleton, cls).__new__(cls, *args, **kwargs)
        return cls.__instance

class YOLOObjects(Singleton):
    def __init__(self):
        self.catNames = pd.read_csv(getCatNamePath())["CatIds"].str.join("")

    def getNamesFromObjIds(self, objIds):
        res = []
        for i in range(0, len(objIds)):
            res.append(self.getNameFromObjId(objIds[i]))
        return res

    def getNameFromObjId(self, objId):
        return self.catNames[objId-1]