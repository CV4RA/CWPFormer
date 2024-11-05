import torch
import torch.nn as nn
import torch.nn.functional as F

class ViTEncoder(nn.Module):
    def __init__(self, num_layers=12, d_model=768, nhead=8):
        super(ViTEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x
