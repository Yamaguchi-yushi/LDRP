"""LaRe-Path decoder fψ_path.

Maps a per-(agent, step) latent factor vector z ∈ ℝ^D to a scalar proxy reward r̂.
Trained with MSE against R_path(τ) = Σ_t Σ_i r_env_t,i.
"""

import torch
from torch import nn


class PathRewardDecoder(nn.Module):
    def __init__(self, factor_dim=10, hidden_dim=64, n_layers=3):
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
