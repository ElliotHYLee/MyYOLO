from MyPyTorchAPI.CNNUtils import *
import numpy as np
from MyPyTorchAPI.CustomActivation import *
import torch.nn.functional as F
class Model_CNN_0(nn.Module):
    def __init__(self, dsName='airsim'):
        super().__init__()
        input_channel = 3
        input_size = (input_channel, 416, 416)
        seq1 = MySeqModel(input_size, [
            Conv2DBlock(input_channel, 32, kernel=3, stride = 1, padding = 1, atvn='prlu', bn = True, dropout=False),
            MaxPool2DBlock(32, kernel = 2, stride = 2, padding = 0),

            Conv2DBlock(32, 64, kernel=3, stride=1, padding = 0, atvn='prlu', bn = True, dropout=False),
            MaxPool2DBlock(64, kernel=2, stride=2, padding= 0),

            Conv2DBlock(64, 128, kernel=3, stride=1, padding=1, atvn='prlu', bn = True, dropout=False),
            Conv2DBlock(128, 64, kernel=1, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(64, 128, kernel=3, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            MaxPool2DBlock(128, kernel=2, stride=2, padding=0),

            Conv2DBlock(128, 256, kernel=3, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(256, 128, kernel=1, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(128, 256, kernel=3, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            MaxPool2DBlock(256, kernel=2, stride=2, padding=0),

            Conv2DBlock(256, 512, kernel=3, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(512, 256, kernel=1, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(256, 512, kernel=3, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(512, 256, kernel=1, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(256, 512, kernel=3, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            MaxPool2DBlock(512, kernel=2, stride=2, padding=0),

            Conv2DBlock(512, 1024, kernel=3, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(1024, 512, kernel=1, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(512, 1024, kernel=3, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(1024, 512, kernel=1, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),
            Conv2DBlock(512, 1024, kernel=3, stride=1, padding=1, atvn='prlu', bn=True, dropout=False),]
        )
        self.encoder = seq1.block
        NN_size = int(seq1.flattend_size)
        print(seq1.output_size)
        self.classOut = nn.Sequential(Conv2DBlock(1024, 1 * 2, kernel=1, stride=1, padding=0, atvn=None, bn=False, dropout=False).block)
        self.bboxOut = nn.Sequential(Conv2DBlock(1024, 1 * 5, kernel=1, stride=1, padding=0, atvn='prlu', bn=True, dropout=True).block)
        #self.init_w()

    # def init_w(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
    #             m.weight.data.normal_(0, 0.5 / np.sqrt(n))
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

    def forward(self, img):
        input = torch.reshape(img, (-1, 3, 416, 416))
        x = self.encoder(input)
        print(x.shape)
        # bboxOut = torch.sigmoid(self.bboxOut(x))
        # bboxOut = torch.reshape(bboxOut, (-1, 7, 7, 1, 5))
        # classOut = self.classOut(x)
        # classOut = torch.reshape(classOut, (-1, 7, 7, 1, 2)) # * bboxOut[:, :, :, :, 0, None]
        # classOut = F.softmax(classOut, dim=4)
        # return bboxOut, classOut

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = nn.DataParallel(Model_CNN_0()).to(device)
    img1 = torch.ones((10, 3, 448, 448), dtype=torch.float).cuda()
    m.forward(img1)
    #bboxOut, classOut= m.forward(img1)
    # print(bboxOut.shape)
    # print(classOut.shape)
    # du, dw, dtr, du_cov, dw_cov, dtr_cov = m.forward(img1, img2)
    # print(m)
