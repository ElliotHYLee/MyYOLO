from MyPyTorchAPI.AbsModelContainer import *
from Common.TooBox import *

class ModelContainer_YOLO(AbsModelContainer):
    def __init__(self, model, wName='Weights/main'):
        super().__init__(model, wName, device_ids=[0])
        self.bn = 0
        self.optimizer = optim.RMSprop(model.parameters(), lr=10 ** -3, weight_decay= 0)
        self.lossMSE = nn.modules.loss.MSELoss()
        self.lossMAE = nn.modules.loss.L1Loss()
        self.lossClass = nn.modules.loss.BCELoss()
        self.tb = ToolBox()

    def forwardProp(self, dataInTuple):
        (x, y) = dataInTuple
        self.x, self.y = self.tb.t2gpu(x), self.tb.t2gpu(y)
        self.bboxOut, self.classOut = self.model(self.x)
        self.label = torch.cat([self.bboxOut, self.classOut], dim=4)

    def getLoss(self):
        target_prob = self.y[:, :, :, :, 0]
        target_bbxy = self.y[:, :, :, :, 1:3]
        target_bbwh = self.y[:, :, :, :, 3:5]
        target_class = self.y[:, :, :, :, 5:]

        pred_prob = self.bboxOut[:, :, :, :, 0]
        pred_bbxy = self.bboxOut[:, :, :, :, 1:3]
        pred_bbwh = self.bboxOut[:, :, :, :, 3:5]
        noIndice = pred_prob < 0.4
        yesno = torch.ones_like(pred_prob)
        yesno[noIndice] = 0

        lambCoord = 5
        lambNoObj = 0.5
        loss = yesno*(self.lossMSE(pred_prob, target_prob)
                     + lambCoord * self.lossMSE(pred_bbxy, target_bbxy))
                      #+ lambCoord * self.lossMSE(pred_bbwh, target_bbwh)
                     #+ self.lossClass(self.classOut, target_class)) \
               #+ lambNoObj*(torch.ones_like(yesno) - yesno)* self.lossClass(self.classOut, target_class)
        loss = torch.reshape(loss, (loss.shape[0], -1)).sum(dim=[1]).mean(dim=0)
        return loss

    def prepResults(self, N):
        self.result0 = np.zeros((N, 7, 7, 5, 105))

    def saveToResults(self, start, last):
        with torch.no_grad():
            self.result0[start:last] = self.tb.gpu2np(self.label)

    def returnResults(self):
        return self.result0

    def changeOptim(self, epoch):
        pass









