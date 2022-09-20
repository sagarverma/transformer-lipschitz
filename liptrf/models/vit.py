import numpy as np
from scipy.special import lambertw 
from scipy.stats import truncnorm 

import torch
from torch import nn
from torch.linalg import norm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from liptrf.models.layers.linear import LinearX

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(
        self, 
        dim: int, 
        fn: nn.Module
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(
        self, 
        x: torch.tensor, 
        **kwargs
    ) -> torch.tensor:
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int, 
        dropout: float = 0.,
        lmbda: float = 1.
    ) -> None:
        super().__init__()
        self.fc1 = LinearX(dim, hidden_dim, power_iter=2, lmbda=lmbda)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = LinearX(hidden_dim, dim, power_iter=2, lmbda=lmbda)
        
     
    def forward(
        self, 
        x: torch.tensor
    ) -> torch.tensor:
        x = self.dropout(self.gelu(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        return x

    def lipschitz(self):
        lc = 1 
        for layer in self.children():
            if isinstance(layer, LinearX):
                lc *= layer.lipschitz()
            if isinstance(layer, nn.GELU):
                lc *= 1.12
        
        return lc

    def apply_spec(self):
        for layer in self.children():
            if isinstance(layer, LinearX):
                layer.apply_spec()


class L2Attention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        heads: int = 8,
        dropout: float = 0.,
        n_value: int = 1,
        lmbda: float = 1.,
        device: int = 0
    ) -> None:
        super().__init__()
        self.device = device
        assert dim % heads == 0, 'dim should be divisible by heads'
        self.dim = dim 
        self.n_value = n_value
        self.heads = heads

        dim_head = dim //  heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_q = LinearX(dim, dim, power_iter=5, lmbda=lmbda)
        self.to_v = LinearX(dim, dim, power_iter=5, lmbda=lmbda)
        self.to_out = LinearX(dim, dim, power_iter=5, lmbda=lmbda)
        self.dropout =  nn.Dropout(dropout)
         
    def forward(
        self, 
        x: torch.tensor
    ) -> torch.tensor:

        q = rearrange(self.to_q(x), 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h = self.heads)
        dots = q @ q.transpose(-2, -1)
        q_l2 = torch.pow(norm(q, dim=-1, ord=2), 2).unsqueeze(-1)
        k_l2 = torch.pow(norm(q, dim=-1, ord=2), 2).unsqueeze(-1)
        q_l2 = torch.matmul(q_l2, torch.ones(q_l2.shape).transpose(-1, -2).type_as(x))
        k_l2 = torch.matmul(torch.ones(k_l2.shape).type_as(x), k_l2.transpose(-1, -2))
        
        attn = (-1 * (q_l2 - 2 * dots + k_l2) * self.scale).softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.dropout(self.to_out(out))

    def lipschitz(self):
        N = self.n_value 
        D = self.dim 
        H = self.heads
        v1 = np.sqrt(N / (D / H))
        v2 = 4 * lambertw(N / np.exp(1)).real + 1
        v3 = torch.sqrt(self.to_q.lipschitz() + self.to_v.lipschitz()) * self.to_out.lipschitz()
        return v1 * v2 * v3

    def apply_spec(self):
        for layer in self.children():
            if isinstance(layer, LinearX):
                layer.apply_spec()

class Attention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dropout: float = 0.,
        n_value: int = 1, 
        lmbda: float = 1.,
        device: int = 1
    ) -> None:
        super().__init__()
        assert dim % heads == 0, 'dim should be divisible by heads'
        self.heads = heads

        dim_head = dim //  heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = LinearX(dim, dim * 3, power_iter=2, lmbda=lmbda)

        self.to_out = LinearX(dim, dim, power_iter=2, lmbda=lmbda)
        self.dropout =  nn.Dropout(dropout)

    def forward(
        self, 
        x: torch.tensor
    ) -> torch.tensor:
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.dropout(self.to_out(out))

    def lipschitz(self):
        return float('nan')

class Transformer(nn.Module):
    def __init__(
        self, 
        dim: int, 
        depth: int, 
        heads: int,
        mlp_ratio: int = 4,  
        dropout: float = 0., 
        attention_type: str = "DP",
        n_value: int = 1,
        lmbda: float = 1.,
        device: int = 0
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        attention = Attention 
        if attention_type == "L2":
            attention = L2Attention

        mlp_hidden_dim = int(dim * mlp_ratio)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attention(dim, heads = heads, dropout = dropout, n_value = n_value, lmbda = lmbda, device=device)),
                PreNorm(dim, FeedForward(dim, mlp_hidden_dim, dropout = dropout, lmbda = lmbda))
            ]))

    def forward(
        self, 
        x: torch.tensor
    ) -> torch.tensor:
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    def lipschitz(self):
        total = 1
        for attn, ff in self.layers:
            total *= ((attn.fn.lipschitz() + 1) * (ff.fn.lipschitz() + 1))

        return total 

    def apply_spec(self):
        for attn, ff in self.layers:
            attn.fn.apply_spec()
            ff.fn.apply_spec()

class ViT(nn.Module):
    def __init__(
        self, 
        *, 
        image_size: int, 
        patch_size: int, 
        num_classes: int, 
        dim: int, 
        depth: int, 
        heads: int,
        mlp_ratio: float = 4.,
        pool: str = 'cls', 
        channels: int = 3, 
        dropout: int = 0., 
        emb_dropout: int = 0.,
        attention_type: str = "DP",
        lmbda: float = 1.,
        device: int = 0
    ) -> None:
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.rearrange_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.to_patch_embedding = LinearX(patch_dim, dim, power_iter=2, lmbda=lmbda)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_ratio, dropout, 
                                       attention_type, num_patches, lmbda=lmbda, device=device)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_ln = nn.LayerNorm(dim)
        self.mlp_head = LinearX(dim, num_classes, power_iter=2, lmbda=lmbda)

    def forward(
        self, 
        img: torch.tensor
    ) -> torch.tensor:
        x = self.rearrange_patch(img)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_ln(x)
        return self.mlp_head(x)

    def lipschitz(self):
        v1 = self.to_patch_embedding.lipschitz()
        v2 = self.transformer.lipschitz()
        v3 = self.mlp_head.lipschitz()
        return v1 * v2 * v3

    def apply_spec(self):
        self.to_patch_embedding.apply_spec()
        self.transformer.apply_spec()
        self.mlp_head.apply_spec()