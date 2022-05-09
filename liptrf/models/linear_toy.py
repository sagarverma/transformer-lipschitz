import numpy as np 
from scipy.stats import truncnorm 

import torch
import torch.nn as nn
from torch.linalg import norm
import torch.nn.functional as F 

from scipy.linalg.interpolative import estimate_spectral_norm as esn


def l2_normalize(x):
    return x / (torch.sqrt(torch.sum(x**2.)) + 1e-9)

def trunc(shape):
    return torch.from_numpy(truncnorm.rvs(-2, 2, size=shape)).cuda().float()


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512, bias=True)  # 5*5 from image dimension
        self.fc2 = nn.Linear(512, 256, bias=True)
        self.fc3 = nn.Linear(256, 10, bias=True)
        self.fc1_x = trunc(784)
        self.fc2_x = trunc(512)
        self.fc3_x = trunc(256)
        self.iter = 100

        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def lipschitz(self):
        v1, self.fc1_x = self.power_iter(self.fc1.weight, self.fc1_x)
        v2, self.fc2_x = self.power_iter(self.fc2.weight, self.fc2_x)
        v3, self.fc3_x = self.power_iter(self.fc3.weight, self.fc3_x)
        return v1 * v2 * v3

    def power_iter(self, W, x):
        for i in range(self.iter):
            x = l2_normalize(x)
            x_p = W @ x 
            x = W.T @ x_p 

        return torch.sqrt(torch.sum(W @ x)**2 / (torch.sum(x**2) + 1e-9)), x
        
