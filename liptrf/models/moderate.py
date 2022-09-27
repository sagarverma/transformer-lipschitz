import math 

import torch
import torch.nn as nn 

from liptrf.models.layers.linear import LinearX
from liptrf.models.layers.conv import Conv2dX


class ReLU_x(nn.Module):
    # learnable relu, has a threshold for each input entry
    def __init__ (self, input_size, init=1.0, **kwargs):
        super(ReLU_x, self).__init__(**kwargs)
        self.threshold = nn.Parameter(torch.Tensor(input_size))
        self.threshold.data.fill_(init)

    def forward(self, x):
        return torch.clamp(torch.min(x, self.threshold), min=0.0)


class Flatten(nn.Module): ## =nn.Flatten()
    def forward(self, x):
        return x.view(x.size()[0], -1)

class ClampGroupSort(nn.Module):
    def __init__(self, input_size, maxthres, minthres):
        super().__init__()
        # input size should be the half channel size as the actual size
        self.min = nn.Parameter(torch.Tensor(input_size))
        self.min.data.fill_(minthres)
        
        self.max = nn.Parameter(torch.Tensor(input_size))
        self.max.data.fill_(maxthres)
        
    def forward(self, x):
        a, b = x.split(x.size(1) // 2, 1)
        a, b = torch.min(torch.max(a, b), self.max), torch.max(torch.min(a, b), self.min)
        return torch.cat([a, b], dim=1)

def mnist_model_large_relux(): 
    model = nn.Sequential(
        nn.Conv2d(1, 32, 3, stride=1, padding=1),
        ReLU_x(torch.Size([1, 32, 28, 28])),
        nn.Conv2d(32, 32, 4, stride=2, padding=1),
        ReLU_x(torch.Size([1, 32, 14, 14])),
        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        ReLU_x(torch.Size([1, 64, 14, 14])),
        nn.Conv2d(64, 64, 4, stride=2, padding=1),
        ReLU_x(torch.Size([1, 64, 7, 7])),
        Flatten(),
        nn.Linear(64*7*7,512),
        ReLU_x(torch.Size([1, 512])),
        nn.Linear(512,512),
        ReLU_x(torch.Size([1, 512])),
        nn.Linear(512,10)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model
    
class MNIST_4C3F_ReLUx(nn.Module):
    
    def __init__(self, 
                 power_iter=5, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(MNIST_4C3F_ReLUx, self).__init__()
        self.conv1 = Conv2dX(1, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu1 = ReLU_x(torch.Size([1, 32, 28, 28]))
        self.conv2 = Conv2dX(32, 32, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu2 = ReLU_x(torch.Size([1, 32, 14, 14]))
        self.conv3 = Conv2dX(32, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu3 = ReLU_x(torch.Size([1, 64, 14, 14]))
        self.conv4 = Conv2dX(64, 64, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu4 = ReLU_x(torch.Size([1, 64, 7, 7]))
        
        self.flatten = Flatten()
        
        self.fc1 = LinearX(64*7*7, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu5 = ReLU_x(torch.Size([1, 512]))
        self.fc2 = LinearX(512, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu6 = ReLU_x(torch.Size([1, 512]))
        self.fc3 = LinearX(512, 10, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        
        x = self.flatten(x)

        x = self.relu5(self.fc1(x))
        x = self.relu6(self.fc2(x))
        x = self.fc3(x)

        return x 

    def forward_lip(self, x):
        x = self.conv1(x)
        x[x < 0] -= self.conv1.lc
        x[x > 0] += self.conv1.lc
        x = self.relu1(x)

        x = self.conv2(x)
        x[x < 0] -= self.conv2.lc
        x[x > 0] += self.conv2.lc
        x = self.relu2(x)

        x = self.conv3(x)
        x[x < 0] -= self.conv3.lc
        x[x > 0] += self.conv3.lc
        x = self.relu3(x)

        x = self.conv4(x)
        x[x < 0] -= self.conv4.lc
        x[x > 0] += self.conv4.lc
        x = self.relu4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x[x < 0] -= self.fc1.lc
        x[x > 0] += self.fc1.lc
        x = self.relu5(x)

        x = self.fc2(x)
        x[x < 0] -= self.fc2.lc
        x[x > 0] += self.fc2.lc
        x = self.relu6(x)

        x = self.fc3(x)

        return x 

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, Conv2dX) or isinstance(layer, LinearX):
                lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc


class MNIST_4C3F_ReLU(nn.Module):
    
    def __init__(self, 
                 power_iter=5, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(MNIST_4C3F_ReLU, self).__init__()
        self.conv1 = Conv2dX(1, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu1 = nn.ReLU()
        self.conv2 = Conv2dX(32, 32, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu2 = nn.ReLU()
        self.conv3 = Conv2dX(32, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu3 = nn.ReLU()
        self.conv4 = Conv2dX(64, 64, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu4 = nn.ReLU()
        
        self.flatten = Flatten()
        
        self.fc1 = LinearX(64*7*7, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu5 = nn.ReLU()
        self.fc2 = LinearX(512, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu6 = nn.ReLU()
        self.fc3 = LinearX(512, 10, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        
        x = self.flatten(x)

        x = self.relu5(self.fc1(x))
        x = self.relu6(self.fc2(x))
        x = self.fc3(x)

        return x

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, Conv2dX) or isinstance(layer, LinearX):
                lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc


class CIFAR10_4C3F_ReLUx(nn.Module):

    def __init__(self,
                 power_iter=5, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(CIFAR10_4C3F_ReLUx, self).__init__()
        self.conv1 = Conv2dX(3, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu1 = ReLU_x(torch.Size([1, 32, 32, 32]))
        self.conv2 = Conv2dX(32, 32, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu2 = ReLU_x(torch.Size([1, 32, 16, 16]))
        self.conv3 = Conv2dX(32, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu3 = ReLU_x(torch.Size([1, 64, 16, 16]))
        self.conv4 = Conv2dX(64, 64, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu4 = ReLU_x(torch.Size([1, 64, 8, 8]))
        
        self.flatten = Flatten()
        
        self.fc1 = LinearX(64*8*8, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu5 = ReLU_x(torch.Size([1, 512]))
        self.fc2 = LinearX(512, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu6 = ReLU_x(torch.Size([1, 512]))
        self.fc3 = LinearX(512, 10, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        
        x = self.flatten(x)

        x = self.relu5(self.fc1(x))
        x = self.relu6(self.fc2(x))
        x = self.fc3(x)

        return x 

    def forward_lip(self, x):
        x = self.conv1(x)
        x[x < 0] -= self.conv1.lc
        x[x > 0] += self.conv1.lc
        x = self.relu1(x)

        x = self.conv2(x)
        x[x < 0] -= self.conv2.lc
        x[x > 0] += self.conv2.lc
        x = self.relu2(x)

        x = self.conv3(x)
        x[x < 0] -= self.conv3.lc
        x[x > 0] += self.conv3.lc
        x = self.relu3(x)

        x = self.conv4(x)
        x[x < 0] -= self.conv4.lc
        x[x > 0] += self.conv4.lc
        x = self.relu4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x[x < 0] -= self.fc1.lc
        x[x > 0] += self.fc1.lc
        x = self.relu5(x)

        x = self.fc2(x)
        x[x < 0] -= self.fc2.lc
        x[x > 0] += self.fc2.lc
        x = self.relu6(x)

        x = self.fc3(x)

        return x 

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, Conv2dX) or isinstance(layer, LinearX):
                lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc


class CIFAR10_6C2F_ReLUx(nn.Module):
    def __init__(self, init=2.0, 
                 power_iter=5, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(CIFAR10_6C2F_ReLUx, self).__init__()
        self.conv1 = Conv2dX(3, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu1 = ReLU_x(torch.Size([1, 32, 32, 32]), init)
        self.conv2 = Conv2dX(32, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu2 = ReLU_x(torch.Size([1, 32, 32, 32]), init)
        self.conv3 = Conv2dX(32, 32, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu3 = ReLU_x(torch.Size([1, 32, 16, 16]), init)
        self.conv4 = Conv2dX(32, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu4 = ReLU_x(torch.Size([1, 64, 16, 16]), init)
        self.conv5 = Conv2dX(64, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu5 = ReLU_x(torch.Size([1, 64, 16, 16]), init)
        self.conv6 = Conv2dX(64, 64, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu6 = ReLU_x(torch.Size([1, 64, 8, 8]), init)
        
        self.flatten = Flatten()
        
        self.fc1 = LinearX(4096, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu7 = ReLU_x(torch.Size([1, 512]), init)
        self.fc2 = LinearX(512, 10, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        
        x = self.flatten(x)

        x = self.relu7(self.fc1(x))
        x = self.fc2(x)

        return x 

    def forward_lip(self, x):
        x = self.conv1(x)
        x[x < 0] -= self.conv1.lc
        x[x > 0] += self.conv1.lc
        x = self.relu1(x)

        x = self.conv2(x)
        x[x < 0] -= self.conv2.lc
        x[x > 0] += self.conv2.lc
        x = self.relu2(x)

        x = self.conv3(x)
        x[x < 0] -= self.conv3.lc
        x[x > 0] += self.conv3.lc
        x = self.relu3(x)

        x = self.conv4(x)
        x[x < 0] -= self.conv4.lc
        x[x > 0] += self.conv4.lc
        x = self.relu4(x)

        x = self.conv5(x)
        x[x < 0] -= self.conv5.lc
        x[x > 0] += self.conv5.lc
        x = self.relu5(x)

        x = self.conv6(x)
        x[x < 0] -= self.conv6.lc
        x[x > 0] += self.conv6.lc
        x = self.relu6(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x[x < 0] -= self.fc1.lc
        x[x > 0] += self.fc1.lc
        x = self.relu7(x)

        x = self.fc2(x)

        return x 

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, Conv2dX) or isinstance(layer, LinearX):
                lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc


class CIFAR10_6C2F_CLMaxMin(nn.Module):

    def __init__(self, init=2.0, 
                 power_iter=5, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(CIFAR10_6C2F_CLMaxMin, self).__init__()
        self.conv1 = Conv2dX(3, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp1 = ClampGroupSort(torch.Size([1, 16, 32, 32]), 0.1, -0.1)
        self.conv2 = Conv2dX(32, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp2 = ClampGroupSort(torch.Size([1, 16, 32, 32]), 0.1, -0.15)
        self.conv3 = Conv2dX(32, 32, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp3 = ClampGroupSort(torch.Size([1, 16, 16, 16]), 0.15, -0.15)
        self.conv4 = Conv2dX(32, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp4 = ClampGroupSort(torch.Size([1, 32, 16, 16]), 0.15, -0.15)
        self.conv5 = Conv2dX(64, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp5 = ClampGroupSort(torch.Size([1, 32, 16, 16]), 0.2, -0.2)
        self.conv6 = Conv2dX(64, 64, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp6 = ClampGroupSort(torch.Size([1, 32, 8, 8]), 0.3, -0.3)
        
        self.flatten = Flatten()
        
        self.fc1 = LinearX(4096, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp7 = ClampGroupSort(torch.Size([1, 256]), 1.2, -1.2)
        self.fc2 = LinearX(512, 10, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)


    def forward(self, x):
        x = self.clamp1(self.conv1(x))
        x = self.clamp2(self.conv2(x))
        x = self.clamp3(self.conv3(x))
        x = self.clamp4(self.conv4(x))
        x = self.clamp5(self.conv5(x))
        x = self.clamp6(self.conv6(x))
        
        x = self.flatten(x)

        x = self.clamp7(self.fc1(x))
        x = self.fc2(x)

        return x 

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, Conv2dX) or isinstance(layer, LinearX):
                lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc

        
class CIFAR10_6C2F_ReLU(nn.Module):

    def __init__(self, init=2.0, 
                 power_iter=5, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(CIFAR10_6C2F_ReLU, self).__init__()
        self.conv1 = Conv2dX(3, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu1 = nn.ReLU()
        self.conv2 = Conv2dX(32, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu2 = nn.ReLU()
        self.conv3 = Conv2dX(32, 32, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu3 = nn.ReLU()
        self.conv4 = Conv2dX(32, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu4 = nn.ReLU()
        self.conv5 = Conv2dX(64, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu5 = nn.ReLU()
        self.conv6 = Conv2dX(64, 64, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu6 = nn.ReLU()
        
        self.flatten = Flatten()
        
        self.fc1 = LinearX(4096, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu7 = nn.ReLU()
        self.fc2 = LinearX(512, 10, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        
        x = self.flatten(x)

        x = self.relu7(self.fc1(x))
        x = self.fc2(x)

        return x

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, Conv2dX) or isinstance(layer, LinearX):
                lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc

class CIFAR100_6C2F_ReLUx(nn.Module):
    def __init__(self, init=2.0, 
                 power_iter=5, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(CIFAR100_6C2F_ReLUx, self).__init__()
        self.conv1 = Conv2dX(3, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu1 = ReLU_x(torch.Size([1, 32, 32, 32]), init)
        self.conv2 = Conv2dX(32, 32, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu2 = ReLU_x(torch.Size([1, 32, 32, 32]), init)
        self.conv3 = Conv2dX(32, 32, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu3 = ReLU_x(torch.Size([1, 32, 16, 16]), init)
        self.conv4 = Conv2dX(32, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu4 = ReLU_x(torch.Size([1, 64, 16, 16]), init)
        self.conv5 = Conv2dX(64, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu5 = ReLU_x(torch.Size([1, 64, 16, 16]), init)
        self.conv6 = Conv2dX(64, 64, 4, stride=2, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu6 = ReLU_x(torch.Size([1, 64, 8, 8]), init)
        
        self.flatten = Flatten()
        
        self.fc1 = LinearX(4096, 512, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu7 = ReLU_x(torch.Size([1, 512]), init)
        self.fc2 = LinearX(512, 100, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        
        x = self.flatten(x)

        x = self.relu7(self.fc1(x))
        x = self.fc2(x)

        return x 

    def forward_lip(self, x):
        x = self.conv1(x)
        x[x < 0] -= self.conv1.lc
        x[x > 0] += self.conv1.lc
        x = self.relu1(x)

        x = self.conv2(x)
        x[x < 0] -= self.conv2.lc
        x[x > 0] += self.conv2.lc
        x = self.relu2(x)

        x = self.conv3(x)
        x[x < 0] -= self.conv3.lc
        x[x > 0] += self.conv3.lc
        x = self.relu3(x)

        x = self.conv4(x)
        x[x < 0] -= self.conv4.lc
        x[x > 0] += self.conv4.lc
        x = self.relu4(x)

        x = self.conv5(x)
        x[x < 0] -= self.conv5.lc
        x[x > 0] += self.conv5.lc
        x = self.relu5(x)

        x = self.conv6(x)
        x[x < 0] -= self.conv6.lc
        x[x > 0] += self.conv6.lc
        x = self.relu6(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x[x < 0] -= self.fc1.lc
        x[x > 0] += self.fc1.lc
        x = self.relu7(x)

        x = self.fc2(x)

        return x 

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, Conv2dX) or isinstance(layer, LinearX):
                lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc

class CIFAR100_8C2F_ReLUx(nn.Module):

    def __init__(self, init=1.0, 
                 power_iter=5, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(CIFAR100_8C2F_ReLUx, self).__init__()
        self.conv1 = Conv2dX(3, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu1 = ReLU_x(torch.Size([1, 64, 32, 32]), init)
        self.conv2 = Conv2dX(64, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu2 = ReLU_x(torch.Size([1, 64, 32, 32]), init)
        self.conv3 = Conv2dX(64, 64, 4, stride=2, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu3 = ReLU_x(torch.Size([1, 64, 15, 15]), init)
        self.conv4 = Conv2dX(64, 128, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu4 = ReLU_x(torch.Size([1, 128, 15, 15]), init)
        self.conv5 = Conv2dX(128, 128, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu5 = ReLU_x(torch.Size([1, 128, 15, 15]), init)
        self.conv6 = Conv2dX(128, 128, 4, stride=2, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu6 = ReLU_x(torch.Size([1, 128, 6, 6]), init)
        self.conv7 = Conv2dX(128, 256, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu7 = ReLU_x(torch.Size([1, 256, 6, 6]), init)
        self.conv8 = Conv2dX(256, 256, 4, stride=2, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu8 = ReLU_x(torch.Size([1, 256, 2, 2]), init)
        
        self.flatten = Flatten()
        
        self.fc1 = LinearX(1024, 256, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu9 = ReLU_x(torch.Size([1, 256]), init)
        self.fc2 = LinearX(256, 100, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        x = self.relu8(self.conv8(x))
        
        x = self.flatten(x)

        x = self.relu9(self.fc1(x))
        x = self.fc2(x)

        return x 
        
    def forward_lip(self, x):
        x = self.conv1(x)
        x[x < 0] -= self.conv1.lc
        x[x > 0] += self.conv1.lc
        x = self.relu1(x)

        x = self.conv2(x)
        x[x < 0] -= self.conv2.lc
        x[x > 0] += self.conv2.lc
        x = self.relu2(x)

        x = self.conv3(x)
        x[x < 0] -= self.conv3.lc
        x[x > 0] += self.conv3.lc
        x = self.relu3(x)

        x = self.conv4(x)
        x[x < 0] -= self.conv4.lc
        x[x > 0] += self.conv4.lc
        x = self.relu4(x)

        x = self.conv5(x)
        x[x < 0] -= self.conv5.lc
        x[x > 0] += self.conv5.lc
        x = self.relu5(x)

        x = self.conv6(x)
        x[x < 0] -= self.conv6.lc
        x[x > 0] += self.conv6.lc
        x = self.relu6(x)

        x = self.conv7(x)
        x[x < 0] -= self.conv7.lc
        x[x > 0] += self.conv7.lc
        x = self.relu7(x)

        x = self.conv8(x)
        x[x < 0] -= self.conv8.lc
        x[x > 0] += self.conv8.lc
        x = self.relu8(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x[x < 0] -= self.fc1.lc
        x[x > 0] += self.fc1.lc
        x = self.relu9(x)

        x = self.fc2(x)

        return x 

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, Conv2dX) or isinstance(layer, LinearX):
                lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc

class TinyImageNet_8C2F_ReLUx(nn.Module):

    def __init__(self, init=1.0, 
                 power_iter=5, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(TinyImageNet_8C2F_ReLUx, self).__init__()
        self.conv1 = Conv2dX(3, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu1 = ReLU_x(torch.Size([1, 64, 64, 64]), init)
        self.conv2 = Conv2dX(64, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu2 = ReLU_x(torch.Size([1, 64, 64, 64]), init)
        self.conv3 = Conv2dX(64, 64, 4, stride=2, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu3 = ReLU_x(torch.Size([1, 64, 31, 31]), init)
        self.conv4 = Conv2dX(64, 128, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu4 = ReLU_x(torch.Size([1, 128, 31, 31]), init)
        self.conv5 = Conv2dX(128, 128, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu5 = ReLU_x(torch.Size([1, 128, 31, 31]), init)
        self.conv6 = Conv2dX(128, 128, 4, stride=2, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu6 = ReLU_x(torch.Size([1, 128, 14, 14]), init)
        self.conv7 = Conv2dX(128, 256, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu7 = ReLU_x(torch.Size([1, 256, 14, 14]), init)
        self.conv8 = Conv2dX(256, 256, 4, stride=2, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu8 = ReLU_x(torch.Size([1, 256, 6, 6]), init)
        
        self.flatten = Flatten()
        
        self.fc1 = LinearX(9216, 256, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta, relu_after=True)
        self.relu9 = ReLU_x(torch.Size([1, 256]), init)
        self.fc2 = LinearX(256, 200, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        x = self.relu8(self.conv8(x))
        
        x = self.flatten(x)

        x = self.relu9(self.fc1(x))
        x = self.fc2(x)

        return x 
        
    def forward_lip(self, x):
        x = self.conv1(x)
        x[x < 0] -= self.conv1.lc
        x[x > 0] += self.conv1.lc
        x = self.relu1(x)

        x = self.conv2(x)
        x[x < 0] -= self.conv2.lc
        x[x > 0] += self.conv2.lc
        x = self.relu2(x)

        x = self.conv3(x)
        x[x < 0] -= self.conv3.lc
        x[x > 0] += self.conv3.lc
        x = self.relu3(x)

        x = self.conv4(x)
        x[x < 0] -= self.conv4.lc
        x[x > 0] += self.conv4.lc
        x = self.relu4(x)

        x = self.conv5(x)
        x[x < 0] -= self.conv5.lc
        x[x > 0] += self.conv5.lc
        x = self.relu5(x)

        x = self.conv6(x)
        x[x < 0] -= self.conv6.lc
        x[x > 0] += self.conv6.lc
        x = self.relu6(x)

        x = self.conv7(x)
        x[x < 0] -= self.conv7.lc
        x[x > 0] += self.conv7.lc
        x = self.relu7(x)

        x = self.conv8(x)
        x[x < 0] -= self.conv8.lc
        x[x > 0] += self.conv8.lc
        x = self.relu8(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x[x < 0] -= self.fc1.lc
        x[x > 0] += self.fc1.lc
        x = self.relu9(x)

        x = self.fc2(x)

        return x 

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, Conv2dX) or isinstance(layer, LinearX):
                lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc


class TinyImageNet_8C2F_CLMaxMin(nn.Module):

    def __init__(self, init=1.0, 
                 power_iter=5, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2):
        super(TinyImageNet_8C2F_CLMaxMin, self).__init__()
        self.conv1 = Conv2dX(3, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp1 = ClampGroupSort(torch.Size([1, 32, 64, 64]), 0.4*2, -0.4*2)
        self.conv2 = Conv2dX(64, 64, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp2 = ClampGroupSort(torch.Size([1, 32, 64, 64]), 0.5*2, -0.5*2)
        self.conv3 = Conv2dX(64, 64, 4, stride=2, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp3 = ClampGroupSort(torch.Size([1, 32, 31, 31]), 0.7*2, -0.7*2)
        self.conv4 = Conv2dX(64, 128, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp4 = ClampGroupSort(torch.Size([1, 64, 31, 31]), 0.7*2, -0.7*2)
        self.conv5 = Conv2dX(128, 128, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp5 = ClampGroupSort(torch.Size([1, 64, 31, 31]), 0.7*2, -0.7*2)
        self.conv6 = Conv2dX(128, 128, 4, stride=2, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp6 = ClampGroupSort(torch.Size([1, 64, 14, 14]), 0.8*2, -0.8*2)
        self.conv7 = Conv2dX(128, 256, 3, stride=1, padding=1, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp7 = ClampGroupSort(torch.Size([1, 128, 14, 14]), 1.0*2, -1.0*2)
        self.conv8 = Conv2dX(256, 256, 4, stride=2, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp8 = ClampGroupSort(torch.Size([1, 128, 6, 6]), 1.5*2, -1.5*2)
        
        self.flatten = Flatten()
        
        self.fc1 = LinearX(9216, 256, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)
        self.clamp9 = ClampGroupSort(torch.Size([1, 128]), 0.8*2, -0.8*2)
        self.fc2 = LinearX(256, 200, power_iter=power_iter, lmbda=lmbda, 
                             lc_gamma=lc_gamma, lc_alpha=lc_alpha, lr=lr, eta=eta)


    def forward(self, x):
        x = self.clamp1(self.conv1(x))
        x = self.clamp2(self.conv2(x))
        x = self.clamp3(self.conv3(x))
        x = self.clamp4(self.conv4(x))
        x = self.clamp5(self.conv5(x))
        x = self.clamp6(self.conv6(x))
        x = self.clamp7(self.conv7(x))
        x = self.clamp8(self.conv8(x))
        
        x = self.flatten(x)

        x = self.clamp9(self.fc1(x))
        x = self.fc2(x)

        return x 

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, Conv2dX) or isinstance(layer, LinearX):
                lc *= layer.lipschitz()
        torch.cuda.empty_cache()
        return lc

# model = MNIST_4C3F_ReLUx()
# inp = torch.randn(2, 1, 28, 28)
# out = model(inp)
# print ("MNIST_4C3F_ReLUx", sum(p.numel() for p in model.parameters()))

# model = MNIST_4C3F_ReLU()
# inp = torch.randn(2, 1, 28, 28)
# out = model(inp)
# print ("MNIST_4C3F_ReLU", sum(p.numel() for p in model.parameters()))

# model = CIFAR10_4C3F_ReLUx()
# inp = torch.randn(2, 3, 32, 32)
# out = model(inp)
# print ("CIFAR10_4C3F_ReLUx", sum(p.numel() for p in model.parameters()))

# model = CIFAR10_6C2F_ReLUx()
# inp = torch.randn(2, 3, 32, 32)
# out = model(inp)
# print ("CIFAR10_6C2F_ReLUx", sum(p.numel() for p in model.parameters()))

# model = CIFAR10_6C2F_CLMaxMin()
# inp = torch.randn(2, 3, 32, 32)
# out = model(inp)
# print ("CIFAR10_6C2F_CLMaxMin", sum(p.numel() for p in model.parameters()))

# model = CIFAR10_6C2F_ReLU()
# inp = torch.randn(2, 3, 32, 32)
# out = model(inp)
# print ("CIFAR10_6C2F_ReLU", sum(p.numel() for p in model.parameters()))

# model = TinyImageNet_8C2F_ReLUx()
# inp = torch.randn(2, 3, 64, 64)
# out = model(inp)
# print ("TinyImageNet_8C2F_ReLUx", sum(p.numel() for p in model.parameters()))

# model = TinyImageNet_8C2F_CLMaxMin()
# inp = torch.randn(2, 3, 64, 64)
# out = model(inp)
# print ("TinyImageNet_8C2F_CLMaxMin", sum(p.numel() for p in model.parameters()))