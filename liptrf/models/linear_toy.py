import torch
import torch.nn as nn
import torch.nn.functional as F 

from liptrf.models.layers import LinearX


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