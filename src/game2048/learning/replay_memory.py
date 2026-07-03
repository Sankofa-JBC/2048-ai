"""Memória de replay para treinamento DQN."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    """Uma experiência observada pelo agente durante uma jogada."""

    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    done: bool
    next_available_actions: tuple[int, ...]


class ReplayMemory:
    """Armazena experiências recentes e amostra lotes aleatórios para treino."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity <= 0:
            raise ValueError("A capacidade da memória deve ser maior que zero.")

        self._items: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, transition: Transition) -> None:
        """Adiciona uma transição à memória."""
        self._items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """Retorna uma amostra aleatória sem reposição."""
        if batch_size <= 0:
            raise ValueError("batch_size deve ser maior que zero.")
        if batch_size > len(self._items):
            raise ValueError("batch_size não pode ser maior que a memória atual.")

        return self._rng.sample(list(self._items), batch_size)

    def __len__(self) -> int:
        return len(self._items)
