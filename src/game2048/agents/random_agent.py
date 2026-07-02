"""Agente aleatório de referência para 2048.

O agente aleatório é intencionalmente simples: ele não aprende e não inspeciona
o tabuleiro. Sua função é fornecer uma base fraca, mas útil, para agentes futuros.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from game2048.agents.base import normalize_available_actions
from game2048.core import Board


class RandomAgent:
    """Escolhe uniformemente entre as ações válidas passadas pelo ambiente."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        # Um gerador aleatório dedicado mantém a aleatoriedade do agente
        # reproduzível e isolada da geração de blocos aleatórios do jogo.
        self._rng = random.Random(seed)

    def choose_action(self, board: Board, available_actions: Sequence[int]) -> int:
        """Retorna uma ação válida da lista de ações disponíveis.

        O tabuleiro faz parte da assinatura do método para que agentes futuros
        usem o mesmo formato ao tomar decisões melhores a partir do estado do jogo.
        """
        actions = normalize_available_actions(available_actions)
        return self._rng.choice(actions)
