from scipy.stats import truncnorm 

import numpy as np
from scipy import linalg

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
                power_iter=5, lmbda=2.5, lc_alpha=0.1,
                lc_gamma=0.1, lr=1, eta=1e-7, relu_after=False):
        super(Conv2dX, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size)))
        self.bias = None
        self.power_iter = power_iter
        self.lmbda = lmbda
        self.lc = 1.0
        self.lc_gamma = lc_gamma
        self.lc_alpha = lc_alpha
        self.eta = eta
        self.lr = lr
        self.prox_done = False 
        self.proj_done = False
        self.relu_after = relu_after

        nn.init.orthogonal_(self.weight)

        def hook(self, input, output):
            self.inp = input[0].detach()
            self.out = output.detach()

        def back_hook(self, grad_input, grad_output):
            self.out -= grad_output[0]

        self.register_forward_hook(hook) 
        self.register_backward_hook(back_hook)

    def forward(self, x):
        return  F.conv2d(x, weight=self.weight, bias=self.bias, 
                        stride=self.stride, padding=self.padding)

    def lipschitz(self):
        rand_x = trunc((1, self.in_channels, 32, 32)).type_as(self.weight)
        for _ in range(self.power_iter):
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
        self.weight_old = self.weight_t#.clone().detach()
        # soft thersholding (L1Norm prox)
        wt = torch.abs(self.weight_t) - self.lc_gamma #- self.lc_alpha
        # wt[wt > 0] += self.lc_alpha
        self.prox_weight = (wt * (wt > 0)) * torch.sign(self.weight_t)
        
        # prox lip
        # wt = torch.abs(self.weight_t)
        # wt[wt < self.lc_alpha * self.lc] -= self.lc_gamma
        # self.prox_weight = wt * torch.sign(self.weight_t)
        
        
        self.proj_weight_0 = (2 * self.prox_weight - self.weight_t)#.clone().detach()
        self.proj_weight = self.proj_weight_0#.clone().detach()

    def proj(self):
        inp_unf = F.unfold(self.inp, (self.kernel_size, self.kernel_size), 
                        stride=self.stride, padding=self.padding)
        out_unf = self.out.reshape((inp_unf.shape[0], self.out.shape[1], inp_unf.shape[2]))
        
        z = inp_unf.transpose(1, 2).matmul(self.proj_weight.t()).transpose(1, 2) - out_unf 
        # print (z.shape, self.inp.shape, inp_unf.shape, self.out.shape, out_unf.shape)
        if self.relu_after:
            z[out_unf == 0 & z <= 0] = 0
        fc = torch.sum(z**2, axis=(1, 2)) - self.eta
        # print (f"Fc: {fc}")
        fc = torch.mean(fc)
        # print (f"Abs Z: {torch.sum(z**2, axis=(1, 2)).max().item()} FC: {fc.item()}")
        dW = torch.zeros(self.proj_weight.shape).type_as(self.weight)

        if fc > 1e-7:
            # for k in range(self.out.shape[0]):
            #     dW += (z[k, :].unsqueeze(1) @ self.inp[k, :].unsqueeze(0))
            # dW /= self.out.shape[0]
            # dW *= 2
            dW = 2 * torch.mean(torch.einsum("bik,bjk->bij", z, inp_unf), dim=0)
            dW = fc * dW / torch.linalg.norm(dW)**2

            # print (f"Norm dW {torch.linalg.norm(dW)}")
            cW = self.proj_weight_0 - self.proj_weight
            # print (f"Norm cW: {torch.linalg.norm(cW)}")
            
            # pi = torch.trace(cW.T @ dW)
            pi = cW.flatten().T @ dW.flatten()
            # print (f"pi {pi}")
            mu = torch.norm(cW, "fro")**2
            vu = torch.norm(dW, "fro")**2
            chi = mu * vu - pi**2
            # print (f"chi: {chi} mu: {mu} vu: {vu} pi^2: {pi**2}")
            if chi < 0:
                # print (f"chi: {chi} mu: {mu} vu: {vu} pi^2: {pi**2}")
                chi = 0

            if (chi == 0) and (pi >= 0):
                # print (f"1 Before Norm proj_weight: {torch.linalg.norm(self.proj_weight)}")
                self.proj_weight = self.proj_weight - dW
                # print (f"1 After Norm proj_weight: {torch.linalg.norm(self.proj_weight)}")
            elif (chi > 0) and ((pi * vu) >= chi):
                # print (f"2 Before Norm proj_weight: {torch.linalg.norm(self.proj_weight)}")
                self.proj_weight = self.proj_weight_0 - (1  + pi/vu) * dW
                # print (f"2 After Norm proj_weight: {torch.linalg.norm(self.proj_weight)}")
            elif (chi > 0) and (pi * vu) < chi:
                # print (f"3 Before Norm proj_weight: {torch.linalg.norm(self.proj_weight)}")
                self.proj_weight = self.proj_weight + vu / chi * \
                                    (pi * cW - mu * dW)
                # print (f"3 After Norm proj_weight: {torch.linalg.norm(self.proj_weight)}")
            else:
                print ("error", chi)
                exit()

    def update(self):
        # print (f"Norm weight_t {torch.linalg.norm(self.weight_t)} " +
        #         f"Norm proj_weight {torch.linalg.norm(self.proj_weight)} " +
        #         f"Norm prox weight {torch.linalg.norm(self.prox_weight)}")
        self.weight_t = (self.weight_t + self.lr * (self.proj_weight - self.prox_weight))#.clone().detach()
        # print (f"Norm weight_t {torch.linalg.norm(self.weight_t)}")

    def free(self):
        self.weight_t = None 
        self.proj_weight = None 
        self.prox_weight = None 
        self.proj_weight_0 = None 
        self.weight_old = None
        self.proj_weight_old = None
        torch.cuda.empty_cache()

# torch.manual_seed(0)
# model = Conv2dX(2, 2, 3, padding=1,
#                 power_iter=10, lmbda=1, lc_gamma=0.1, lr=1.2, eta=1e-1).cuda()
# inp = torch.ones(64, 2, 32, 32).cuda()
# out = inp * 2

# crit = nn.MSELoss()
# optim = torch.optim.SGD(model.parameters(), lr=0.1)

# for i in range(100):
#     optim.zero_grad()
#     pred = model(inp)
#     loss = crit(pred, out)
#     loss.backward()
#     optim.step()

# print (i, loss.item(), model.lipschitz())

# model.eval()


# weight_t = model.weight.clone().detach()
# model.weight_t = weight_t.view(weight_t.size(0), -1)
# model.weight_old = model.weight_t.clone().detach()

# for i in range(100):
#     model.prox()
#     for j in range(2000):
#         # print (f"################# Prox epoch {i} Proj Epoch {j} #################")
#         model.proj_weight_old = model.proj_weight.clone().detach()
#         for b in range(8):
#             pred = model(inp[b*8:(b+1)*8, :])
#             model.proj()
#         if torch.linalg.norm(model.proj_weight - model.proj_weight_old) < 1e-7 * torch.linalg.norm(model.proj_weight):
#             print ('convergence')
#             break 
    
#     model.update()
#     if torch.linalg.norm(model.weight_t - model.weight_old) < 1e-4 * torch.norm(model.weight_t):
#         print ("prox conv")
#         break

# model.weight = nn.Parameter(model.prox_weight.reshape(model.weight.shape))
# pred = model(inp)
# loss = crit(pred, out)
# print (loss.item(), model.lipschitz())