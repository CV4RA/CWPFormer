import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiChannelAttentionLayer(nn.Module):
    def __init__(self, dim, heads=8):
        super(MultiChannelAttentionLayer, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).permute(0, 2, 1, 3), qkv)

        dots = (q @ k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)

class CascadedHashAttention(nn.Module):
    def __init__(self, dim, heads=8):
        super(CascadedHashAttention, self).__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).permute(0, 2, 1, 3), qkv)

        # Compute hash attention (split input into different heads)
        dots = (q @ k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)

class CWPFormer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072):
        super(CWPFormer, self).__init__()

        # Patch Embedding
        self.patch_embed = nn.Conv2d(3, dim, patch_size, patch_size)

        # Cross-weight Pyramid Transformer (CWPFormer)
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(
                MultiChannelAttentionLayer(dim, heads),
                CascadedHashAttention(dim, heads),
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, dim)
            ) for _ in range(depth)
        ])

        # Final classification head
        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        for layer in self.transformer_layers:
            x = layer(x) + x

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
