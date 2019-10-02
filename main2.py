from main2_utils import *
from Model.Model import *
import time
from torch.utils.data import Dataset, DataLoader
import sys

class MyDataSet(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.x.shape[0]

def train(trainSet, validSet):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLO_V1()
    model = nn.DataParallel(model).to(device)

    epoch = 10**5
    optimizer = torch.optim.SGD(model.parameters(), lr=10 **-3, weight_decay= 0)
    lossMSE = nn.modules.loss.MSELoss()

    output = 0

    tb  = ToolBox()
    bn = 32
    trainSetLoader = DataLoader(trainSet, batch_size=bn)
    validSetLoader = DataLoader(validSet, batch_size=bn)
    for i in range(0, epoch):
        model.train()
        trainLoss, validLoss = 0, 0
        for batch_idx, (input, target) in enumerate(trainSetLoader):
            input, target = tb.t2gpu(input), tb.t2gpu(target)
            output = model.forward(input)
            p_prob = output[:, :, :, 0]
            prob = target[:, :, :, 0]

            p_xy = output[:, :, :, 1:3]
            xy = target[:, :, :, 1:3]

            trainLoss = lossMSE(p_prob, prob) + lossMSE(p_xy, xy)
            optimizer.zero_grad()
            trainLoss.backward()
            optimizer.step()
            msg = "===> Epoch[{}]({}/{}): Batch Loss: {:.4f}".format(i, (batch_idx+1)*bn, len(trainSet), trainLoss.item())
            sys.stdout.write('\r' + msg)

        model.eval()
        for batch_idx, (input, target) in enumerate(validSetLoader):
            input, target = tb.t2gpu(input), tb.t2gpu(target)
            output = model.forward(input)
            p_prob = output[:, :, :, 0]
            prob = target[:, :, :, 0]

            p_xy = output[:, :, :, 1:3]
            xy = target[:, :, :, 1:3]

            validLoss = lossMSE(p_prob, prob) + lossMSE(p_xy, xy)

        print('\nepoch: {0}, trainLoss: {1}, validLoss: {2}'.format(i,
                     np.round(trainLoss.item(), 7),
                     np.round(validLoss.item(), 7)))
        torch.save({'model_state_dict': model.state_dict(),
                    'model_optim_state': optimizer.state_dict()}, 'asdf.pt')

def test(imgs, bboxesGT, testSet):
    cp = torch.load('asdf.pt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO_V1()
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()

    tb = ToolBox()

    testSetLoader = DataLoader(testSet, batch_size=32)
    outputList = []
    for batch_idx, (input, target) in enumerate(testSetLoader):
        input, target = tb.t2gpu(input), tb.t2gpu(target)
        output = model.forward(input)
        outputList.append(tb.gpu2np(output))


    p_label = np.concatenate(outputList, axis=0)
    lm = LabelMaker(N)


    p_bboxes = lm.unpackLable(p_label)
    p_bboxes[:,:, 2:] = 16

    imgs = lm.drawRect(imgs, bboxesGT)
    imgs = lm.drawRect(imgs, p_bboxes, False)
    #imgs = drawGrids(imgs)

    for i in range(0, imgs.shape[0]):
        cv2.imshow('asdf', imgs[i])
        cv2.waitKey(10000)

if __name__ == '__main__':
    print('making labels...')
    s = time.time()
    N = 10**4
    lm = LabelMaker(N)
    label = lm.makeLabel()
    bboxes = lm.unpackLable(label)
    imgs = (lm.genImage(bboxes)/255.0).astype(np.float32)

    wallA = int(N*0.8)
    wallB = int(N*0.9)

    trainSet = MyDataSet(imgs[:wallA], label[:wallA])
    validSet = MyDataSet(imgs[wallA:wallB], label[wallA:wallB])
    testSet = MyDataSet(imgs[wallB:], label[wallB:])

    print(time.time() - s)
    print('done making')


    train(trainSet, validSet)
    test(imgs[wallB:], bboxes[wallB:], testSet)

