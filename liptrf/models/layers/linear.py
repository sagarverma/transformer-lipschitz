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
        self.lc = 1
        self.lc_gamma = lc_gamma
        self.lc_alpha = lc_alpha
        self.eta = eta
        self.lr = lr
        self.prox_done = False 
        self.proj_done = False
        self.relu_after = False
        # print (self.power_iter)

        nn.init.orthogonal_(self.weight)

        def hook(self, input, output):
            self.inp = input[0].detach()
            self.out = output.detach()

        self.register_forward_hook(hook) 

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def lipschitz(self):
        rand_x = trunc(self.input).type_as(self.weight)
        for _ in range(self.power_iter):
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
        self.weight_old = self.weight_t#.clone().detach()
        # soft thersholding (L1Norm prox)
        wt = torch.abs(self.weight_t) - self.lc_gamma #- self.lc_alpha
        # wt[wt > 0] += self.lc_alpha
        self.prox_weight = (wt * (wt > 0)) * torch.sign(self.weight_t)
        
        # prox lip
        # wt = torch.abs(self.weight_t)
        # wt[wt < self.lc_alpha * self.lc] -= self.lc_gamma
        # self.prox_weight = wt * torch.sign(self.weight_t)
        # print (self.prox_weight.max(), self.weight_t.max())
        self.proj_weight_0 = (2 * self.prox_weight - self.weight_t)#.clone().detach()
        self.proj_weight = self.proj_weight_0#.clone().detach()

    def proj(self):
        # print (self.inp.min(), self.inp.max(), self.out.min(), self.out.max())
        z = F.linear(self.inp, self.proj_weight) - self.out
        # print (z.shape)
        if self.relu_after:
            z[self.out == 0 & z <= 0] = 0
        fc = torch.sum(z**2, axis=1) - self.eta
        fc = torch.mean(fc)
        # print (f"Norm z: {torch.linalg.norm(z).item()} FC: {fc.item()}")
        dW = torch.zeros(self.proj_weight.shape).type_as(self.weight)

        if fc > 1e-7:
            # for k in range(self.out.shape[0]):
            #     dW += (z[k, :].unsqueeze(1) @ self.inp[k, :].unsqueeze(0))
            # dW /= self.out.shape[0]
            # dW *= 2
            if len(self.inp.shape) == 3:
                dW = 2 * torch.mean(torch.einsum("bki,bkj->bij", z, self.inp), dim=0)
            else:
                dW = 2 * torch.mean(torch.einsum("bi,bj->bij", z, self.inp), dim=0)
            dW = fc * dW / torch.linalg.norm(dW)**2

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
# model = LinearX(32, 32, power_iter=10, lmbda=1, lc_gamma=0.1, lr=1.1, eta=0).cuda()
# inp = torch.ones(20, 32).cuda()
# out = torch.ones(20, 32).cuda() * 2

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


# model.weight_t = model.weight.clone().detach()
# model.weight_old = model.weight.clone().detach()

# for i in range(100):
#     model.prox()
#     for j in range(1000):
#         # print (f"Prox epoch {i} Proj Epoch {j}")
#         model.proj_weight_old = model.proj_weight#.clone().detach()
#         for b in range(10):
#             pred = model(inp[2*b:2*(b+1), :])
#             model.proj()
#         if torch.linalg.norm(model.proj_weight - model.proj_weight_old) < 1e-7 * torch.linalg.norm(model.proj_weight):
#             print ('convergence')
#             break 
    
#     model.update()

#     if torch.linalg.norm(model.weight_t - model.weight_old) < 1e-4 * torch.norm(model.weight_t):
#         print ("prox conv")
#         break

# model.weight = nn.Parameter(model.prox_weight)
# # print (model.weight)
# pred = model(inp)
# # print (pred)
# loss = crit(pred, out)
# print (loss.item(), model.lipschitz())