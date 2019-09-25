import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import collections

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def main():
    N = 1000
    x = (np.random.rand(N, 20) - 0.5) * 4
    y = np.zeros((N))
    for i in range(0, N):
        a, b = x[i,1], x[i,0]
        if a >= b + 1:
            y[i] = 2
        elif a >= b - 1:
            y[i] = 1
        else:
            y[i] = 0

    #print(collections.Counter(y))
    device = torch.device("cuda")
    torchX =

if __name__ == '__main__':
    main()