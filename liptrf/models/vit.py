import torch
from torch import nn
from torch.nn.utils import spectral_norm

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

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
        dropout: float = 0.
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(
        self, 
        x: torch.tensor
    ) -> torch.tensor:
        return self.net(x)

    def lipschitz(self):
        l1 = self.net[0].weight.weight.norm(p=2)
        l2 = self.net[3].weight.weight.norm(p=2)
        return 1.12 * l1 * l2

class L2Attention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0.,
        n_value: int = 1
    ) -> None:
        super().__init__()
        self.dim = dim 
        self.n_value = n_value
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(
        self, 
        x: torch.tensor
    ) -> torch.tensor:
        print (x.shape)
        qv = self.to_qv(x).chunk(2, dim = -1)
        print (self.to_qv(x).shape, qv[0].shape, self.to_qv.weight.shape)
        q, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qv)
        print (q.shape)
        dots = q @ q.transpose(-2, -1)
        q_l2 = torch.pow(q.norm(dim=-1, p=2), 2).unsqueeze(-1)
        k_l2 = torch.pow(q.norm(dim=-1, p=2), 2).unsqueeze(-1)
        q_l2 = torch.matmul(q_l2, torch.ones(q_l2.shape).transpose(-1, -2).cuda())
        k_l2 = torch.matmul(torch.ones(k_l2.shape).cuda(), k_l2.transpose(-1, -2))
        
        attn = (-1 * (q_l2 - 2 * dots + k_l2) * self.scale).softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def lipschitz(self):
        N = self.n_value 
        D = self.dim 
        H = self.heads
        W_Qh = self.to_qv.weight[:512, :]
        W_Vh = self.to_qv.weight[512:, :]
        W_o = self.to_out[0].weight
        v1 = (N / (D / H)) ** 0.5
        v2 = 4 * 1 / ((N-1) * torch.exp(N)) + 1
        v3 = torch.sqrt(torch.pow(W_Qh.norm(p=2), 2) * torch.pow(W_Vh.norm(p=2), 2)) * W_o.norm(p=2)

        return v1 * v2 * v3

class Attention(nn.Module):
    def __init__(
        self, 
        dim: int, 
        heads: int = 8, 
        dim_head: int = 64, 
        dropout: float = 0.,
        n_value: int = 1
    ) -> None:
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

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
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(
        self, 
        dim: int, 
        depth: int, 
        heads: int, 
        dim_head: int, 
        mlp_dim: int, 
        dropout: float = 0., 
        attention_type: str = "DP",
        n_value: int = 1
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([])
        attention = Attention 
        if attention_type == "L2":
            attention = L2Attention

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, n_value = n_value)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
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
            total *= (attn.fn.lipschitz() + 1 * ff.fn.lipschitz() + 1)

        return total 

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
        mlp_dim: int, 
        pool: str = 'cls', 
        channels: int = 3, 
        dim_head: int = 64, 
        dropout: int = 0., 
        emb_dropout: int = 0.,
        attention_type: str = "DP"
    ) -> None:
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, 
                                       dim_head, mlp_dim, dropout, 
                                       attention_type, num_patches+1)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(
        self, 
        img: torch.tensor
    ) -> torch.tensor:
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

    def lipschitz(self):
        v1 = self.to_patch_embedding[1].weight.norm(p=2)
        v2 = self.transformer.lipschitz()
        v3 = self.mlp_head[1].weight.weight.norm(p=2)

        return v1 * v2 * v3