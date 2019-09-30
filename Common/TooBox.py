import torch
import numpy as np

class ToolBox():
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def np2gpu(self, x):
        return torch.from_numpy(x).to(self.device)

    def np2gpu_float(self, x):
        return self.np2gpu(x).type(torch.float)

    def t2gpu(self, x):
        return x.to(self.device, dtype=torch.float)

    def gpu2np(self, x):
        return x.cpu().numpy()

    def gpu2np_float(self, x):
        return (self.gpu2np(x)).astype(np.float)

    def np2np_ohc2index(self, ohc):
        pass

    def t2t_ohc2index(self, ohc):
        return torch.argmax(ohc, dim=1)

    def np2np_index2ohc(self, c):
        pass

    def t2t_index2ohc(self, c):
        pass
