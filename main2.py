from main2_utils import *
from Model.Model import *


def train(label, bboxes, imgs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = YOLO_V1()
    model = nn.DataParallel(model).to(device)
    model.train()

    tb  = ToolBox()
    input = tb.np2gpu(imgs.copy())
    target = tb.np2gpu(label)
    epoch = 10**3
    optimizer = torch.optim.SGD(model.parameters(), lr=10 ** -2, weight_decay= 0)
    lossMSE = nn.modules.loss.MSELoss()
    output = 0
    for i in range(0, epoch):
        output = model.forward(input)
        p_prob = output[:, :, :, 0]
        prob = target[:, :, :, 0]

        p_xy = output[:, :, :, 1:3]
        xy = target[:, :, :, 1:3]

        loss = lossMSE(p_prob, prob) + lossMSE(p_xy, xy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch: {0}, loss: {1}'.format(i, loss.item()))

    torch.save({'model_state_dict': model.state_dict(),
                'model_optim_state': optimizer.state_dict()}, 'asdf.pt')

def test(label, bboxes, imgs):
    cp = torch.load('asdf.pt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = YOLO_V1()
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(cp['model_state_dict'])
    model.eval()

    tb = ToolBox()
    input = tb.np2gpu(imgs.copy())
    #target = tb.np2gpu(label)

    output = model.forward(input)

    p_label = tb.gpu2np_float(output)
    p_bboxes = unpackLable(p_label)
    p_bboxes[:,:, 2:] = 16

    imgs = drawRect(imgs, bboxes)
    imgs = drawRect(imgs, p_bboxes, False)
    #imgs = drawGrids(imgs)

    for i in range(0, 32):
        cv2.imshow('asdf', imgs[i])
        cv2.waitKey(10000)

if __name__ == '__main__':
    label = makeLabel()
    bboxes = unpackLable(label)
    imgs = genImage(bboxes)/255.0
    #train(label, bboxes, imgs)
    test(label, bboxes, imgs)

