from typing import List, Union

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class Conv2dEx(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        sniter: int = 1
    ) -> None:
        super(Conv2dEx, self).__init__(in_channels, out_channels, kernel_size, stride, 
                                        padding, dilation, groups, bias)
        self.u = None
        self.eps = 1e-24
        self.device = device
        self.sniter = sniter
    
    def bcp(self, mu, r, ibp_mu, ibp_r):
        ibp_mu = self.forward(ibp_mu)
        ibp_r = self.forward(ibp_r)
        ibp_ub = ibp_mu + ibp_r
        ibp_lb = ibp_mu - ibp_r 

        mu_after = self.forward(mu)

        p = self.power_iteration(mu)
        
        ibp_p1 = self.weight.view(self.weight.size()[0], -1).norm(2, dim=-1)
        ibp_r1 = r.view(-1, 1, 1, 1) * torch.ones_like(mu_after)
        ibp_r1 = ibp_r1 * ibp_p1.view(1, -1, 1, 1)
        ibp_ub1 = mu_after + ibp_r1
        ibp_lb1 = mu_after - ibp_r1 
        ibp_ub = torch.min(ibp_ub, ibp_ub1)
        ibp_lb = torch.max(ibp_lb, ibp_lb1)
        ibp_mu = (ibp_ub + ibp_lb) / 2 
        ibp_r = (ibp_ub - ibp_lb) / 2 

        r = r * p

        return mu_after, r, ibp_mu, ibp_r

    def power_iteration(self, mu):
        output_padding = 0 
        if not self.u:
            self.u = torch.randn((1,*mu.size()[1:])).to(self.device)

        if self.bias is not None:
            b = torch.zeros_like(self.bias)
        else:
            b = None 

        for i in range(self.sniter):
            u1 = F.conv2d(self.u, self.weight, b, stride=self.stride, padding=self.padding)
            u1_norm = u1.norm(2)
            v = u1 / (u1_norm + self.eps)

            v1 = F.conv_transpose2d(v, self.weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

            if v1.shape != self.u.shape:
                output_padding = 1 
                v1 = F.conv_transpose2d(v, self.weight, stride=self.stride, padding=self.padding, output_padding=output_padding)
            v1_norm = v1.norm(2)
            u_old = self.u
            self.u = v1 / (v1_norm + self.eps)

            if (self.u - u_old).norm(2) < 1e-5:
                break

        out = (v*(F.conv2d(self.u, self.weight, b, stride=self.stride, padding=self.padding))).view(v.size()[0],-1).sum(1)[0]

        return out
