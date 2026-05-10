"""LaRe-Task decoder fψ_task: maps per-assignment latent factors to a scalar proxy reward."""

import torch
from torch import nn


class TaskRewardDecoder(nn.Module):
    def __init__(self, factor_dim=10, hidden_dim=64, n_layers=2):
        super().__init__()
        if n_layers <= 1:
            self.model = nn.Linear(factor_dim, 1)
        else:
            layers = [nn.Linear(factor_dim, hidden_dim), nn.ReLU()]
            for _ in range(max(0, n_layers - 2)):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers += [nn.Linear(hidden_dim, 1)]
            self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)
