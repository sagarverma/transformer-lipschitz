import torch
import torch.nn as nn
import torch.nn.functional as F 

from liptrf.models.layers.linear import LinearX


class Net(nn.Module):

    def __init__(self,
                 power_iter=5, lmbda=1, 
                 lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(Net, self).__init__()
        self.fc1 = LinearX(784, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)  # 5*5 from image dimension
        self.fc2 = LinearX(512, 256, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.fc3 = LinearX(256, 10, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
       
    def forward(self, x):
        # Max pooling over a (2, 2) window
        # x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
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