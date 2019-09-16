
def getTrainAnnPath():
    dirTrainAnn = './Data/train.json'
    return dirTrainAnn

def getValAnnPath():
    dirValAnn = './Data/train.json'
    return dirValAnn

def getImgPath(id):
    fName = 'COCO_train2014_' + getFileName(id)
    dirImgPath = 'D:/DLData/COCO/2014/train2014'
    imgFileName = '{}/{}.jpg'.format(dirImgPath, fName)
    return imgFileName

def getFileName(id):
    return  str(id).zfill(12)
