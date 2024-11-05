import torch
import torch.nn as nn

class TotalLoss(nn.Module):
    def __init__(self, classification_loss_weight=1.0, hash_loss_weight=0.5, similarity_loss_weight=0.5):
        super(TotalLoss, self).__init__()
        self.classification_loss = ClassificationLoss()
        self.hash_loss = HashLoss()
        self.similarity_loss = SimilarityLoss()

        # 权重
        self.classification_loss_weight = classification_loss_weight
        self.hash_loss_weight = hash_loss_weight
        self.similarity_loss_weight = similarity_loss_weight

    def forward(self, outputs, labels, hash_codes, similarity_scores):
        # 计算分类损失
        classification_loss = self.classification_loss(outputs, labels)
        
        # 计算哈希损失
        hash_loss = self.hash_loss(hash_codes, labels)
        
        # 计算相似度损失
        similarity_loss = self.similarity_loss(similarity_scores, labels)
        
        # 总损失
        total_loss = (self.classification_loss_weight * classification_loss +
                      self.hash_loss_weight * hash_loss +
                      self.similarity_loss_weight * similarity_loss)
        
        return total_loss
