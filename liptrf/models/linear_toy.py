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
    return torch.from_numpy(truncnorm.rvs(-2, 2, size=shape)).float()


class LinearX(nn.Module):
    def __init__(self, input, output, iter=5, lmbda=2.5, relax=1, lr=1, eta=1e-7):
        super(LinearX, self).__init__()
        self.input = input
        self.weight = nn.Parameter(torch.Tensor(output, input))
        self.rand_x = nn.Parameter(trunc(self.input), requires_grad=False)
        self.iter = iter
        self.lmbda = lmbda
        self.lc = 1.0
        self.relax = relax
        self.eta = eta
        self.lr = lr

        nn.init.orthogonal_(self.weight)

        def hook(self, input, output):
            self.inp = input[0].detach()
            self.out = output.detach()

        self.register_forward_hook(hook) 

    def forward(self, x):
        return F.linear(x, self.weight)

    def lipschitz(self):
        for i in range(self.iter):
            x = l2_normalize(self.rand_x)
            x_p = F.linear(x, self.weight) 
            self.rand_x = nn.Parameter(F.linear(x_p, self.weight.T), requires_grad=False)

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

    def prox(self):
        self.prox_weight = self.weight.clone() / self.relax
        self.proj_weight = 2 * self.prox_weight - self.weight
        self.proj_weight_n = self.proj_weight.clone()

    def proj(self):
        # print (self.inp.shape, self.proj_weight_n.shape)
        cjn = torch.mean(torch.sum(F.linear(self.inp, self.proj_weight_n) - self.out)**2 - self.eta)  
        if cjn > 0:
            cj = torch.mean(torch.sum(F.linear(self.inp, self.proj_weight) - self.out)**2 - self.eta)
            if len(self.inp.shape) == 3:
                num = 2 * torch.sum(torch.einsum("bnjd,bnci->bndc", 
                                    (F.linear(self.inp, self.proj_weight_n) - self.out).unsqueeze(-2), 
                                    self.inp.unsqueeze(-1)), dim=[0, 1])
            else:
                num = 2 * torch.sum(torch.einsum("bjd,bci->bdc", 
                                    (F.linear(self.inp, self.proj_weight_n) - self.out).unsqueeze(-2), 
                                    self.inp.unsqueeze(-1)), dim=0)
            den = torch.norm(num, 'fro')**2
            del_wn = cj * num / den 
            pi_n =  torch.trace((self.proj_weight - self.proj_weight_n).T @ del_wn)
            mu_n = torch.norm(self.proj_weight - self.proj_weight_n, 'fro')**2
            vu_n = torch.norm(del_wn, 'fro')**2 
            chi_n = mu_n * vu_n - pi_n**2 
            
            if chi_n == 0 and pi_n >= 0:
                self.proj_weight_n = self.proj_weight_n - del_wn
            elif chi_n > 0 and (pi_n * vu_n) >= chi_n:
                self.proj_weight_n = self.proj_weight - (1  + pi_n/vu_n) * del_wn
            else:
                self.proj_weight_n = self.proj_weight + vu_n / chi_n * (pi_n * (self.proj_weight - self.proj_weight_n) - mu_n * del_wn)

    def update(self):
        self.weight += self.lr * (self.proj_weight - self.prox_weight)


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