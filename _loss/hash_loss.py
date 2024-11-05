# hash_loss.py

import torch
import torch.nn.functional as F

class HashLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(HashLoss, self).__init__()
        self.margin = margin

    def forward(self, hash_codes, labels):
        batch_size = hash_codes.size(0)
        
        # 计算哈希码之间的 Hamming 距离
        hamming_distances = torch.cdist(hash_codes.float(), hash_codes.float(), p=0)
        
        # 计算标签之间的相似性
        label_similarity = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # 创建损失：对于相似的样本，Hamming 距离应小于 margin
        loss = torch.mean(F.relu(hamming_distances - self.margin * label_similarity))
        
        return loss
