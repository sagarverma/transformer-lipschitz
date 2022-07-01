import math 

import torch
import torch.nn as nn 


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

class MNIST_4C3F_ReLUx(nn.Module):
    
    def __init__(self):
        super(MNIST_4C3F_ReLUx, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.relu1 = ReLU_x(torch.Size([1, 32, 28, 28]))
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.relu2 = ReLU_x(torch.Size([1, 32, 14, 14]))
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu3 = ReLU_x(torch.Size([1, 64, 14, 14]))
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.relu4 = ReLU_x(torch.Size([1, 64, 7, 7]))
        
        self.flatten = Flatten()
        
        self.fc1 = nn.Linear(64*7*7,512)
        self.relu5 = ReLU_x(torch.Size([1, 512]))
        self.fc2 = nn.Linear(512,512)
        self.relu6 = ReLU_x(torch.Size([1, 512]))
        self.fc3 = nn.Linear(512,10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

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

class MNIST_4C3F_ReLU(nn.Module):
    
    def __init__(self):
        super(MNIST_4C3F_ReLU, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        
        self.flatten = Flatten()
        
        self.fc1 = nn.Linear(64*7*7,512)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(512,512)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(512,10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

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
class CIFAR10_4C3F_ReLUx(nn.Module):

    def __init__(self):
        super(CIFAR10_4C3F_ReLUx, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.relu1 = ReLU_x(torch.Size([1, 32, 32, 32]))
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.relu2 = ReLU_x(torch.Size([1, 32, 16, 16]))
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu3 = ReLU_x(torch.Size([1, 64, 16, 16]))
        self.conv4 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.relu4 = ReLU_x(torch.Size([1, 64, 8, 8]))
        
        self.flatten = Flatten()
        
        self.fc1 = nn.Linear(64*8*8,512)
        self.relu5 = ReLU_x(torch.Size([1, 512]))
        self.fc2 = nn.Linear(512,512)
        self.relu6 = ReLU_x(torch.Size([1, 512]))
        self.fc3 = nn.Linear(512,10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

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

class CIFAR10_C6F2_ReLUx(nn.Module):
    def __init__(self, init=2.0):
        super(CIFAR10_C6F2_ReLUx, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.relu1 = ReLU_x(torch.Size([1, 32, 32, 32]), init)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.relu2 = ReLU_x(torch.Size([1, 32, 32, 32]), init)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.relu3 = ReLU_x(torch.Size([1, 32, 16, 16]), init)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu4 = ReLU_x(torch.Size([1, 64, 16, 16]), init)
        self.conv5 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.relu5 = ReLU_x(torch.Size([1, 64, 16, 16]), init)
        self.conv6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.relu6 = ReLU_x(torch.Size([1, 64, 8, 8]), init)
        
        self.flatten = Flatten()
        
        self.fc1 = nn.Linear(4096,512)
        self.relu7 = ReLU_x(torch.Size([1, 512]), init)
        self.fc2 = nn.Linear(512,10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

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

class CIFAR10_C6F2_CLMaxMin(nn.Module):

    def __init__(self, init=2.0):
        super(CIFAR10_C6F2_CLMaxMin, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.clamp1 = ClampGroupSort(torch.Size([1, 16, 32, 32]), 0.1, -0.1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.clamp2 = ClampGroupSort(torch.Size([1, 16, 32, 32]), 0.1, -0.15)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2, padding=1)
        self.clamp4 = ClampGroupSort(torch.Size([1, 16, 16, 16]), 0.15, -0.15)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.clamp4 = ClampGroupSort(torch.Size([1, 32, 16, 16]), 0.15, -0.15)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.clamp5 = ClampGroupSort(torch.Size([1, 32, 16, 16]), 0.2, -0.2)
        self.conv6 = nn.Conv2d(64, 64, 4, stride=2, padding=1)
        self.clamp6 = ClampGroupSort(torch.Size([1, 32, 8, 8]), 0.3, -0.3)
        
        self.flatten = Flatten()
        
        self.fc1 = nn.Linear(4096,512)
        self.clamp7 = ClampGroupSort(torch.Size([1, 256]), 1.2, -1.2)
        self.fc2 = nn.Linear(512,10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

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

# class CIFAR10_C6F2_ReLU(nn.Module):
#     # cifar10 standard relu

# class TinyImageNet_8C2F_ReLUx(nn.Module):


# model = MNIST_4C3F_ReLUx()
# inp = torch.randn(2, 1, 28, 28)
# print (model(inp).shape)