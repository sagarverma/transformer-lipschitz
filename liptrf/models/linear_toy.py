import numpy as np 

import torch
import torch.nn as nn
from torch.linalg import norm
import torch.nn.functional as F 

from scipy.linalg.interpolative import estimate_spectral_norm as esn


def l2_normalize(x):
    return x / (torch.sqrt(torch.sum(x**2.)) + 1e-7)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(784, 512, bias=True)  # 5*5 from image dimension
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, 10, bias=True)

        # nn.init.orthogonal_(self.fc1.weight)
        # nn.init.orthogonal_(self.fc2.weight)
        # nn.init.orthogonal_(self.fc3.weight)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def lipschitz(self):
        v1 = self.power_iter(self.fc1.weight, torch.randn(784).cuda())
        v2 = self.power_iter(self.fc2.weight, torch.randn(512).cuda())
        v3 = self.power_iter(self.fc3.weight, torch.randn(256).cuda())
        # v1 = esn(self.fc1.weight.data.cpu().numpy().astype(np.float64), its=5)
        # v2 = esn(self.fc2.weight.data.cpu().numpy().astype(np.float64), its=5)
        # v3 = esn(self.fc3.weight.data.cpu().numpy().astype(np.float64), its=5)
        # print (v1, v2, v3)
        return v1 * v2 * v3

    def power_iter(self, W, x):
        for i in range(100):
            x = l2_normalize(x)
            x_p = W @ x 
            x = W.T @ x_p 

        return torch.sqrt(torch.sum(W @ x)**2 / (torch.sum(x**2) + 1e-7))
        
