"""
This module defines a deep residual network for Higgs Boson classification.
It includes the `HiggsNet` class, which consists of multiple residual blocks for feature extraction
and a final linear layer for binary classification.
It also includes a `ResidualBlock` class that implements the residual connection and dropout for regularization
"""
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x):
        return x + self.block(x)


class HiggsNet(nn.Module):
    """
    Deep Residual Network for Higgs Boson Classification.

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in each layer.
        num_layers (int): Number of residual blocks.
        dropout (float): Dropout probability.
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=4, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers):
            layers.append(ResidualBlock(hidden_dim, dropout=dropout))
        layers.append(
            nn.Linear(hidden_dim, 1)
        )  # Output layer for binary classification
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)
