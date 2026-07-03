"""Agente DQN para escolher ações usando um modelo PyTorch treinado."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from game2048.agents.base import normalize_available_actions
from game2048.core import Board
from game2048.learning.dqn_model import DQNModel
from game2048.learning.state import board_to_tensor


class DQNAgent:
    """Agente que escolhe a melhor ação válida segundo uma rede DQN."""

    name = "dqn"

    def __init__(
        self,
        model: DQNModel,
        device: torch.device | str = "cpu",
        epsilon: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon deve ficar entre 0.0 e 1.0.")

        self.model = model.to(device)
        self.device = torch.device(device)
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self.model.eval()

    def choose_action(self, board: Board, available_actions: Sequence[int]) -> int:
        """Escolhe uma ação válida usando epsilon-greedy."""
        actions = normalize_available_actions(available_actions)

        if self._rng.random() < self.epsilon:
            return self._rng.choice(actions)

        with torch.no_grad():
            state = board_to_tensor(board, device=self.device)
            q_values = self.model(state)[0]

        return _best_valid_action(q_values, actions)


def load_dqn_agent(
    model_path: str | Path,
    device: torch.device | str = "cpu",
    epsilon: float = 0.0,
) -> DQNAgent:
    """Carrega um checkpoint salvo e retorna um agente DQN pronto para uso."""
    checkpoint = torch.load(model_path, map_location=device)
    model_config: dict[str, Any] = checkpoint.get("model_config", {})
    model = DQNModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return DQNAgent(model=model, device=device, epsilon=epsilon)


def _best_valid_action(q_values: torch.Tensor, available_actions: Sequence[int]) -> int:
    """Aplica máscara nas ações inválidas e retorna o melhor índice válido."""
    masked_q_values = torch.full_like(q_values, -1_000_000_000.0)
    for action in available_actions:
        masked_q_values[action] = q_values[action]
    return int(torch.argmax(masked_q_values).item())
