import math
import numpy as np 

import torch
import torch.nn as nn
from torch.linalg import norm
import torch.nn.functional as F 

from scipy.linalg.interpolative import estimate_spectral_norm as esn


class Flatten(nn.Module): ## =nn.Flatten()
    def forward(self, x):
        return x.view(x.size()[0], -1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.flatten = Flatten()
        self.fc5 = nn.Linear(64*7*7,512)
        self.fc6 = nn.Linear(512,512)
        self.fc7 = nn.Linear(512,10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.fc7(x)

        return x

    def lipschitz(self):
        v1 = norm(self.conv1.weight, ord=2)
        v2 = norm(self.conv2.weight, ord=2)
        v3 = norm(self.conv3.weight, ord=2)
        v4 = norm(self.conv4.weight, ord=2)
        v5 = norm(self.fc5.weight, ord=2)
        v6 = norm(self.fc6.weight, ord=2)
        v7 = norm(self.fc7.weight, ord=2)

        return v1 * v2 * v3 * v4 * v4 * v5 * v6 * v7