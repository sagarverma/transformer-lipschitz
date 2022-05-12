from re import S
import numpy as np
from scipy.stats import truncnorm 

import random
import torch
import torch.nn as nn
from torch.linalg import norm
import torch.nn.functional as F 

from scipy.linalg.interpolative import estimate_spectral_norm as esn


def l2_normalize(x):
    return x / (torch.sqrt(torch.sum(x**2.)) + 1e-9)

def trunc(shape):
    return torch.from_numpy(truncnorm.rvs(-2, 2, size=shape)).float().cuda()


class LinearX(nn.Module):
    def __init__(self, input, output, iter=5, lmbda=2.5):
        super(LinearX, self).__init__()
        self.input = input
        self.weight = nn.Parameter(torch.Tensor(output, input))
        self.rand_x = trunc(self.input)
        self.iter = iter
        self.lmbda = lmbda
        self.lc = 1.0

        nn.init.orthogonal_(self.weight)

        # self.lipschitz() 

    def forward(self, x):
        return F.linear(x, self.weight)

    def lipschitz(self):
        for i in range(self.iter):
            x = l2_normalize(self.rand_x)
            x_p = F.linear(x, self.weight) 
            self.rand_x = F.linear(x_p, self.weight.T) 

        self.lc = torch.sqrt(torch.sum(self.weight @ x)**2 / (torch.sum(x**2) + 1e-9))
        del x, x_p
        torch.cuda.empty_cache()
        return self.lc.data.cpu()

    def apply_spec(self):
        fc = self.weight.clone().detach()
        fc = fc * 1 / (max(1, self.lc / self.lmbda))
        self.weight = nn.Parameter(fc)
        del fc
        torch.cuda.empty_cache()


class Net(nn.Module):

    def __init__(self, iter=2, lmbda=1):
        super(Net, self).__init__()
        self.fc1 = LinearX(784, 512, iter=iter, lmbda=lmbda)  # 5*5 from image dimension
        self.fc2 = LinearX(512, 256, iter=iter, lmbda=lmbda)
        self.fc3 = LinearX(256, 10, iter=iter, lmbda=lmbda)
       
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc

    def apply_spec(self):
        for layer in self.children():
            layer.apply_spec()
        torch.cuda.empty_cache()