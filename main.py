from DataLoader.DataLoader3_PyTorch import *
from Model.ModelContainer import *
from Model.Model import  *
from DataLoader.Helper.Helper_TargetUnpacker import *
from DataLoader.DataVis import *

def train():
    dm = YOLODataSetManager(start=0, N = 16, isTrain=True)
    mc = ModelContainer_YOLO(Model_CNN_0())
    mc.regress(dm, epochs=100, batch_size=8, shuffle=False)

def test():
    unpacker = TargetUnpacker()
    visG = Visualizer_Global()

    dm = YOLODataSetManager(start=0, N=16, isTrain=False)
    mc = ModelContainer_YOLO(YOLO_V1(), wName='Weights/main_best.pt')
    dataSet = dm.testSet
    trainLoader = DataLoader(dataset=dataSet, batch_size=8)
    y = mc.predict(dm, batch_size=8)
    for batch_idx, (img, label) in enumerate(trainLoader):
        img = img.data.numpy()
        label = label.data.numpy()
        pred_label=y[batch_idx*8:(batch_idx+1)*8]

        for i in range(img.shape[0]):
            img_ = img[i, :]
            objIds, offset, bb = unpacker.unpackLabel(label[i])
            pred_label[:,:,:,:,3:5]  = np.zeros_like(pred_label[:,:,:,:,3:5])
            p_objIds, p_offset, p_bb = unpacker.unpackLabel(pred_label[i])
            print(pred_label[:,1:2])
            print(p_bb.shape)
            print(p_bb)

            img_ = visG.drawBBox(img_, p_bb, YOLOObjects().getNamesFromObjIds(objIds))
            cv2.imshow('img_', img_)
            cv2.waitKey(3000)

if __name__ == '__main__':
    #train()
    test()





