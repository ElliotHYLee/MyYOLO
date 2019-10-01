from main2_utils import *

label = makeLabel()
bboxes = unpackLable(label)
imgs = genImage(bboxes)

print(bboxes.shape)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Model_CNN_0()
model = nn.DataParallel(model).to(device)
model.train()


tb  = ToolBox()
input = tb.np2gpu(imgs.copy())
target = tb.np2gpu(label)
epoch = 10000
optimizer = torch.optim.SGD(model.parameters(), lr=10 ** -3, weight_decay= 0)
lossMSE = nn.modules.loss.MSELoss()
for i in range(0, epoch):
    p_bb, p_class = model.forward(input)
    p_prob = p_bb[:, :, :, :, 0]
    prob = target[:, :, :, :, 0]
    loss = lossMSE(p_prob, prob)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(loss.item())


p_label = torch.cat([p_bb, p_class], dim=4)
p_label = tb.gpu2np_float(p_label)
p_bboxes = unpackLable(p_label)
p_bboxes[:,:, 2:] = 16



imgs = drawRect(imgs, bboxes)
imgs = drawRect(imgs, p_bboxes, False)
imgs = darwGrids(imgs)
for i in range(0, 32):
    cv2.imshow('asdf', imgs[i])
    cv2.waitKey(10000)


