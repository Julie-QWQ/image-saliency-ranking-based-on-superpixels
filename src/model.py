import torch
from torch import nn


class BranchNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc6 = nn.Linear(128, feature_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        return x


class MultiBranchNet(nn.Module):
    def __init__(self, feature_dim, mlp_hidden):
        super().__init__()
        self.branch_a = BranchNet(feature_dim)
        self.branch_b = BranchNet(feature_dim)
        self.branch_c = BranchNet(feature_dim)
        self.fc7 = nn.Linear(feature_dim * 3, mlp_hidden)
        self.head = nn.Linear(mlp_hidden, 1)

    def forward(self, xa, xb, xc):
        fa = self.branch_a(xa)
        fb = self.branch_b(xb)
        fc = self.branch_c(xc)
        z = torch.cat([fa, fb, fc], dim=1)
        z = torch.relu(self.fc7(z))
        out = self.head(z).squeeze(1)
        return out
