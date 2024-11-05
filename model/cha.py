import torch
import torch.nn as nn

class CHA(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CHA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hash_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 将输入特征进行线性变换
        hash_output = self.hash_layer(x)
        # 通过符号函数生成哈希代码
        hash_codes = torch.sign(hash_output)
        return hash_codes
