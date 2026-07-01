"""Random baseline agent for 2048.

The random agent is intentionally simple: it does not learn and does not inspect
the board. Its job is to give us a weak but useful baseline for future agents.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from game2048.core import Board, validate_action


class RandomAgent:
    """Choose uniformly among the valid actions passed by the environment."""

    def __init__(self, seed: int | None = None) -> None:
        # A dedicated RNG keeps agent randomness reproducible and isolated from
        # the game's random tile generation.
        self._rng = random.Random(seed)

    def choose_action(self, board: Board, available_actions: Sequence[int]) -> int:
        """Return one valid action from the available action list.

        The board is part of the method signature so future agents can use the
        same shape while making smarter decisions from the game state.
        """
        actions = tuple(available_actions)
        if not actions:
            raise ValueError("RandomAgent needs at least one available action.")

        for action in actions:
            validate_action(action)

        return self._rng.choice(actions)
