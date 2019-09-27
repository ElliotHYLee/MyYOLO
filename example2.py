import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections
from DataLoader.Helper.Helper_TargetPacker import *
from Common.TooBox import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=1)
        return x

def main():
    N = 1000
    x = np.float32((np.random.rand(N, 2) - 0.5) * 2)
    y = np.longlong(np.zeros((N)))

    for i in range(0, N):
        a, b = x[i,1], x[i,0]
        if a >= b + 0.5:
            y[i] = 2
        elif a >= b - 0.5:
            y[i] = 1
        else:
            y[i] = 0

    #print(collections.Counter(y))
    packer = TargetPacker4D()
    ohc = np.zeros((N, 3))
    for i in range(0, N):
        ohc[i,:] = packer.getOneHotCode(y[i], dim = 3)

    tb = ToolBox()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torchX = tb.np2gpu_float(x)
    torchY = tb.np2gpu_float(y)
    torchOhC = tb.np2gpu_float(ohc)

    m = nn.DataParallel(Net()).to(device)
    m.train()
    optimizer = optim.SGD(m.parameters(), lr=10**-1, weight_decay=10**-4)

    myLoss = nn.modules.loss.BCELoss()
    print('training...')
    for i in range(0, 2000):
        optimizer.zero_grad()
        predY =  m.forward(torchX)
        loss = myLoss(predY, torchOhC)

        #loss = F.nll_loss(predY, tb.t2t_ohc2index(torchOhC))
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        m.eval()
        torchPredY = m.forward(torchX)

    predY = tb.gpu2np(tb.t2t_ohc2index(torchPredY))
    error = np.abs(predY - y)
    print(collections.Counter(error))

if __name__ == '__main__':
    main()