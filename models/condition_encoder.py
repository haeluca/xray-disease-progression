import torch
import torch.nn as nn


class ConditionEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int = 128, output_dim: int = 128):
        super().__init__()
        self.num_features = num_features
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)
