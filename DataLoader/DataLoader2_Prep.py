from DataLoader.DataLoader1_ReadAll import DataLoader1_ReadAll

class DataLoader2_Prep():
    def __init__(self, start, N,  isTrain):
        self.reader = DataLoader1_ReadAll(start, N)
        self.isTrain = isTrain

    def standardize(self):
        pass

    def saveStats(self):
        pass


