import torch
import torch.nn as nn

class BaselineNN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=2):
        super(BaselineNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
