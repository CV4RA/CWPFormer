import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianSimilarity(nn.Module):
    def __init__(self):
        super(BayesianSimilarity, self).__init__()

    def forward(self, query_features, reference_features):
        cos_sim = F.cosine_similarity(query_features, reference_features, dim=1)
        return cos_sim
