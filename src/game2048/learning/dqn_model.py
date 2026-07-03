"""Modelo neural DQN para estimar valores Q das ações do 2048."""

from __future__ import annotations

import torch
from torch import nn

from game2048.constants import ACTIONS
from game2048.learning.state import BOARD_VECTOR_SIZE


class DQNModel(nn.Module):
    """Rede pequena para mapear estado 4x4 em quatro valores Q."""

    def __init__(
        self,
        input_size: int = BOARD_VECTOR_SIZE,
        hidden_size: int = 128,
        output_size: int = len(ACTIONS),
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
