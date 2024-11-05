import torch
import torch.nn.functional as F

class SimilarityLoss(nn.Module):
    def __init__(self):
        super(SimilarityLoss, self).__init__()

    def forward(self, similarity_scores, labels):
        target_similarity = labels.float()

        loss = F.mse_loss(similarity_scores, target_similarity)
        return loss
