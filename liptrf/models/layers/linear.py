from scipy.stats import truncnorm 

import torch
import torch.nn as nn
import torch.nn.functional as F 


def l2_normalize(x):
    return x / (torch.sqrt(torch.sum(x**2.)) + 1e-9)

def trunc(shape):
    return torch.from_numpy(truncnorm.rvs(0.5, 1, size=shape)).float()


class LinearX(nn.Module):
    def __init__(self, input, output, 
                 power_iter=5, lmbda=2.5, lc_alpha=0.1,
                 lc_gamma=0.1, lr=1, eta=1e-7):
        super(LinearX, self).__init__()
        self.input = input
        self.output = output
        self.weight = nn.Parameter(torch.empty(output, input))
        self.bias = None
        self.power_iter = power_iter
        self.lmbda = lmbda
        self.lc = 1.0
        self.lc_gamma = lc_gamma
        self.lc_alpha = lc_alpha
        self.eta = eta
        self.lr = lr

        nn.init.orthogonal_(self.weight)

        def hook(self, input, output):
            self.inp = input[0].detach()
            self.out = output.detach()

        self.register_forward_hook(hook) 

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def lipschitz(self):
        rand_x = trunc(self.input).cuda()
        for i in range(self.power_iter):
            x = l2_normalize(rand_x)
            x_p = F.linear(x, self.weight) 
            rand_x = F.linear(x_p, self.weight.T)

        self.lc = torch.sqrt(torch.abs(torch.sum(self.weight @ x)) / (torch.abs(torch.sum(x)) + 1e-9)).data.cpu()
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
        self.weight_old = self.weight_t.clone().detach()

        # soft thersholding (L1Norm prox)
        # wt = torch.abs(self.weight_t) - self.relax
        # self.prox_weight = (wt * (wt > 0)) * torch.sign(self.weight_t)
        
        # prox of infinity norm
        wt = torch.abs(self.weight_t)
        wt[wt > self.lc_alpha] -= self.lc_gamma
        self.prox_weight = wt * torch.sign(self.weight_t)

        self.proj_weight_0 = 2 * self.prox_weight - self.weight_t
        self.proj_weight = self.proj_weight_0.clone().detach()

    def proj(self):
        z = F.linear(self.inp,self.proj_weight) - self.out
        fc = torch.sum(z**2, axis=0) - self.eta
        fc = torch.mean(fc)
        aW = torch.zeros(self.proj_weight.shape).cuda()

        if fc > 0:
            tW = aW
            for k in range(self.out.shape[0]):
                tW += 2 * F.linear(z[k, :], self.inp[k, :].T)
            tW /= self.out.shape[0]
            aW = -fc * tW / torch.linalg.norm(tW)**2

        L = torch.sum(aW**2)
        
        if L > 2.2204e-16:
            cW = self.proj_weight_0 - self.proj_weight
            dW = 1e-2 * aW.clone() 

            pi = -cW.flatten().T @ dW.flatten()
            mu = torch.norm(cW)**2
            vu = torch.norm(dW)**2
            chi = mu * vu - pi**2

            if chi < 0:
                chi = 0

            if (chi == 0) and (pi >= 0):
                self.proj_weight = self.proj_weight + dW
            elif (chi > 0) and ((pi * vu) >= chi):
                self.proj_weight = self.proj_weight_0 + (1  + pi/vu) * dW
            elif (chi > 0) and (pi * vu) < chi:
                self.proj_weight = self.proj_weight + vu / chi * \
                                    (pi * cW + mu * dW)        

    def update(self):
        self.weight_t = self.weight_t + self.lr * (self.proj_weight - self.prox_weight)
        


    

model = LinearX(32, 32).cuda()
inp = torch.randn(60000, 32).cuda()
out = inp * 2

crit = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(100):
    optim.zero_grad()
    pred = model(inp)
    loss = crit(pred, out)
    loss.backward()
    optim.step()

print (i, loss.item(), model.lipschitz())

model = LinearX(32, 32, lmbda=1).cuda()
optim = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(100):
    optim.zero_grad()
    pred = model(inp)
    loss = crit(pred, out)
    model.lipschitz()
    if i > 30:
        model.apply_spec()
    loss.backward()
    optim.step()

print (i, loss.item(), model.lipschitz())


model = LinearX(32, 32, lmbda=1, lc_gamma=0.1, lc_alpha=0.01, lr=1.2, eta=1e-2).cuda()
optim = torch.optim.SGD(model.parameters(), lr=0.1)

for i in range(100):
    optim.zero_grad()
    pred = model(inp)
    loss = crit(pred, out)
    loss.backward()
    optim.step()

print (i, loss.item(), model.lipschitz())

model.eval()


model.weight_t = model.weight.clone().detach()
# model.weight_t = (torch.abs(model.weight_t) + model.relax) * torch.sign(model.weight_t)

wt = torch.abs(model.weight_t)
wt[wt > model.lc_alpha] += model.lc_gamma
model.weight_t = wt * torch.sign(model.weight_t)

for i in range(2):
    model.prox()
    for j in range(100):
        model.proj_weight_old = model.proj_weight.clone().detach()
        for b in range(8):
            pred = model(inp[b*8:(b+1)*8, :])
            model.proj()
        if torch.linalg.norm(model.proj_weight - model.proj_weight_old) < 1e-7 * torch.linalg.norm(model.proj_weight):
            break 
    
    model.update()
    if torch.linalg.norm(model.weight_t - model.weight_old) < 1e-4 * torch.norm(model.weight_t):
        break

model.weight = nn.Parameter(model.weight_t)
pred = model(inp)
loss = crit(pred, out)
print (loss.item(), model.lipschitz())