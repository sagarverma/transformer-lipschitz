import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F


class LipschitzMargin(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()

        # self.weight = nn.Parameter(torch.Tensor(output_shape, input_shape))

    def forward(self, y, w, epsilon, lipschitz):
        y_j, j, K_ij = self._get_yj_j_Kij(y, w, lipschitz)
        y_bot_i = y + epsilon * K_ij

        y_bot_i = torch.where(torch.eq(y, y_j), -np.infty + torch.zeros_like(y_bot_i), y_bot_i)
        y_bot = torch.max(y_bot_i, dim=1, keepdims=True)[0]

        return torch.cat([y, y_bot], dim=1)

    def _get_yj_j_Kij(self, y, w, lipschitz):
        kW = lipschitz * w.T

        j = torch.argmax(y, dim=1)
        y_j = torch.max(y, dim=1, keepdim=True)[0]
        kW_j = kW.T[j]
        kW_ij = kW_j[:,:,None] - kW[None]
        K_ij = torch.sqrt(torch.sum(kW_ij * kW_ij, dim=1))
        
        return y_j, j, K_ij 

    def lipschitz(self, y):
        y_j, j, K_ij = self._get_yj_j_Kij(y)

        return torch.where(torch.eq(y, y_j), torch.zeros_like(K_ij) - 1., K_ij)