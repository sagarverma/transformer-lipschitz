import numpy as np 

import torch
import torch.nn as nn
from torch.linalg import norm
import torch.nn.functional as F 

from scipy.linalg.interpolative import estimate_spectral_norm as esn


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(784, 512)  # 5*5 from image dimension
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def lipschitz(self):
        v1 = norm(self.fc1.weight, ord=2)
        v2 = norm(self.fc2.weight, ord=2)
        v3 = norm(self.fc3.weight, ord=2)
        # v1 = esn(self.fc1.weight.data.cpu().numpy().astype(np.float64), its=2)
        # v2 = esn(self.fc2.weight.data.cpu().numpy().astype(np.float64), its=2)
        # v3 = esn(self.fc3.weight.data.cpu().numpy().astype(np.float64), its=2)
        return v1 * v2 * v3
