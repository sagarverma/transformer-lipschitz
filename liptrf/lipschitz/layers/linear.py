import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class LinearEx(nn.Linear):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        bias: bool = True, 
        device=None, 
        sniter: int = 1
    ) -> None:
        super(LinearEx, self).__init__(in_features, out_features, bias, device)
        
        self.u = None 
        self.eps = 1e-24 
        self.device = device 
        self.sniter = sniter

    def bcp(self, mu, r, ibp_mu, ibp_r):
        ibp_mu = self.forward(ibp_mu)
        ibp_r = self.forward(ibp_r)