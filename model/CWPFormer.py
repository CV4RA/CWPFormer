import torch
import torch.nn as nn
from vit_encoder import ViTEncoder
from cha import CHA
from bayesian_similarity import BayesianSimilarity

class CWPFormer(nn.Module):
    def __init__(self, num_classes=10):
        super(CWPFormer, self).__init__()
        # 定义 ViT Encoder
        self.transformer_encoder = ViTEncoder()
        
        # 定义级联哈希注意力模块
        self.cha = CHA(input_dim=768, output_dim=64)  
        
        # 定义贝叶斯相似度计算模块
        self.bayesian_similarity = BayesianSimilarity()
        
        # 分类层
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # 通过 ViT 提取特征
        features = self.transformer_encoder(x)
        
        # 通过 CHA 进行哈希编码
        hash_codes = self.cha(features)
        
        # 通过贝叶斯学习计算相似度
        similarity_score = self.bayesian_similarity(hash_codes, hash_codes)  
        
        # 最终的分类输出
        out = self.fc(hash_codes)
        return out, similarity_score
