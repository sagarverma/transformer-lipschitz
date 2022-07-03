from scipy.stats import truncnorm 

import torch
import torch.nn as nn
import torch.nn.functional as F 


def l2_normalize(x):
    return x / (torch.sqrt(torch.sum(x**2.)) + 1e-9)

def trunc(shape):
    return torch.from_numpy(truncnorm.rvs(0.5, 1, size=shape)).float()


class Conv2dX(nn.Module):
    def __init__(self, in_channels, out_channels, 
                        kernel_size, stride=1, padding=0,
                        iter=5, lmbda=2.5, relax=1, lr=1, eta=1e-7):
        super(Conv2dX, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = None
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
        return  F.conv2d(x, weight=self.weight, bias=self.bias, 
                        stride=self.stride, padding=self.padding)

    def lipschitz(self):
        rand_x = trunc((1, self.in_channels, 32, 32)).cuda()
        for i in range(self.iter):
            x = l2_normalize(rand_x)
            x_p = F.conv2d(x, self.weight, 
                           stride=self.stride, 
                           padding=self.padding) 
            rand_x = F.conv_transpose2d(x_p, self.weight, 
                                        stride=self.stride, 
                                        padding=self.padding)

        Wx = F.conv2d(rand_x, self.weight, 
                      stride=self.stride, padding=self.padding)
        self.lc = torch.sqrt(
                        torch.abs(torch.sum(Wx**2.)) / 
                        (torch.abs(torch.sum(rand_x**2.)) + 1e-9)).data.cpu()
        del x, x_p
        torch.cuda.empty_cache()
        return self.lc

    def apply_spec(self):
        fc = self.weight.clone().detach()
        fc = fc * 1 / (max(1, self.lc / self.lmbda))
        self.weight = nn.Parameter(fc)
        del fc
        torch.cuda.empty_cache()

    def prox(self):
        self.lipschitz()
        self.prox_weight = self.weight.clone().detach() / self.relax
        self.proj_weight = 2 * self.prox_weight - self.weight.clone().detach()
        self.proj_weight_0 = self.proj_weight.clone()

    def proj(self):
        if torch.norm(self.proj_weight_n-self.proj_weight, 'fro') < self.eta * torch.norm(self.weight, 'fro'):
            return 

        z = F.linear(self.inp, self.proj_weight_n) - self.out
        if len(z.shape) == 3:
            cjn = torch.mean(torch.sum(z**2, dim=[0, 1]) - self.eta)
        else:
            cjn = torch.mean(torch.sum(z**2, dim=0) - self.eta)

        del_wn = torch.zeros(self.proj_weight_n.shape)
        if cjn > 0:
            if len(self.inp.shape) == 3:
                num = 2 * torch.sum(torch.einsum("bnjd,bnci->bndc", 
                                    z.unsqueeze(-2), 
                                    self.inp.unsqueeze(-1)), dim=[0, 1])
            else:
                num = 2 * torch.sum(torch.einsum("bjd,bci->bdc", 
                                    z.unsqueeze(-2), 
                                    self.inp.unsqueeze(-1)), dim=0)
            num = num / self.out.shape[-1]
            den = torch.norm(num, 'fro')**2
            del_wn = -cjn * num / den 
        
        L = torch.sum(del_wn**2)
        if L > 1e-22:
            cW = self.proj_weight - self.proj_weight_n

            pi_n =  -1 * (cW.T.flatten().unsqueeze(0) @ del_wn.flatten().unsqueeze(1))
            mu_n = torch.norm(cW, p=2)**2
            vu_n = torch.norm(del_wn, p=2)**2 
            chi_n = mu_n * vu_n - pi_n**2 

            if chi_n < 0:
                chi_n = 0

            # print (del_wn.max(), vu_n, chi_n, pi_n, mu_n)
            if (chi_n == 0) and (pi_n >= 0):
                self.proj_weight_n = self.proj_weight_n + del_wn
            elif (chi_n > 0) and ((pi_n * vu_n) >= chi_n):
                self.proj_weight_n = self.proj_weight + (1  + pi_n/vu_n) * del_wn
            elif (chi_n > 0) and ((pi_n * vu_n) < chi_n):
                self.proj_weight_n = self.proj_weight_n + vu_n / chi_n * (pi_n * cW - mu_n * del_wn)
            else:
                raise Exception("Error")

    def update(self):
        self.proj_weight = self.proj_weight_n
        self.weight = nn.Parameter(self.weight + self.lr * (self.proj_weight - self.prox_weight))



model = Conv2dX(1, 32, 3).cuda()
inp = torch.randn(4, 1, 32, 32).cuda()
out = model(inp)
print (out.shape)
print (model.lipschitz())