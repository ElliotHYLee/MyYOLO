from DataLoader.DataLoader0_ReadAnns import DataLoader0_ReadAnns

class DataLoader1_ReadAll():
    def __init__(self):
        anns = DataLoader0_ReadAnns()
        i = 150
        img = anns.getImgAt(i)
        bboxes = anns.getBBoxesAt(i)
        objNames = anns.getObjNamesAt(i)
        objIds = anns.getObjIdsAt(i)
        img, bboxes = anns.resize(img, bboxes)
        offset, relX, relY = anns.getBBoxCenter(bboxes)
        img = anns.drawBBox(img, bboxes, objNames)
        img = anns.drawBBoxCenter(img, offset, relX, relY, bboxes.shape[0])
        anns.showImg(img)
