

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError


class CrisisKANClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.15):
        super().__init__()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(5)])
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # Average logits from multiple dropout branches sharing one linear classifier.
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.fc(dropout(x))
            else:
                out += self.fc(dropout(x))
        return out / len(self.dropouts)