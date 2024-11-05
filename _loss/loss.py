import torch
import torch.nn as nn

class ClassificationLoss(nn.Module):
    def __init__(self):
        super(ClassificationLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        loss = self.cross_entropy(outputs, labels)
        return loss
