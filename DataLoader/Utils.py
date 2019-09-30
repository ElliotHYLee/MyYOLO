import numpy as np

def getCatNamePath():
    return './Data/objNames.txt'

def getTrainAnnPath():
    dirTrainAnn = './Data/train.json'
    return dirTrainAnn

def getValAnnPath():
    dirValAnn = './Data/train.json'
    return dirValAnn

def getImgPath(id):
    fName = 'COCO_train2014_' + getFileName(id)
    #dirImgPath = 'F:/DLData/COCO/2014/train2014'
    dirImgPath = 'E:/COCO/2014/train2014'
    imgFileName = '{}/{}.jpg'.format(dirImgPath, fName)
    return imgFileName

def getFileName(id):
    return  str(id).zfill(12)

def getEnd(start, N, totalN):
    end = start+N
    if end > totalN:
        end = totalN
        N = end-start
    return end, N

def getNumThread(totalN, partN):
    return int(totalN / partN) if np.mod(totalN, partN) == 0 else int(totalN / partN) + 1

def getBranchName():
    return 'master'

def getNormPath():
    pass