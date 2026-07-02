"""Ambiente 2048 com estado para humanos e futuros agentes de IA."""

from __future__ import annotations

import random
from typing import Any

from game2048.constants import ACTIONS, BOARD_SIZE, NEW_TILE_TWO_PROBABILITY
from game2048.core import (
    Board,
    can_move,
    clone_board,
    empty_board,
    get_empty_cells,
    move_board,
    validate_board,
)


class Game2048:
    """Pequeno wrapper no estilo ambiente em torno das regras do 2048."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self._board = empty_board(BOARD_SIZE)
        self.score = 0
        self.steps = 0
        self.done = False
        self.reset()

    @property
    def board(self) -> Board:
        return clone_board(self._board)

    def reset(self) -> Board:
        self._board = empty_board(BOARD_SIZE)
        self.score = 0
        self.steps = 0
        self.done = False
        self._add_random_tile()
        self._add_random_tile()
        return self.board

    def step(self, action: int) -> tuple[Board, int, bool, dict[str, Any]]:
        """Aplica uma ação e retorna estado, recompensa, fim e informações."""
        if self.done:
            return self.board, 0, True, {
                "changed": False,
                "score": self.score,
                "reason": "game_over",
            }

        moved_board, reward, changed = move_board(self._board, action)

        if changed:
            self._board = moved_board
            self.score += reward
            self.steps += 1
            self._add_random_tile()

        self.done = not can_move(self._board)

        return self.board, reward, self.done, {
            "changed": changed,
            "score": self.score,
            "steps": self.steps,
        }

    def set_board(self, board: Board, score: int = 0) -> None:
        """Define um tabuleiro customizado, principalmente para testes e experimentos."""
        validate_board(board)
        self._board = clone_board(board)
        self.score = score
        self.steps = 0
        self.done = not can_move(self._board)

    def available_actions(self) -> list[int]:
        actions: list[int] = []
        for action in ACTIONS:
            _, _, changed = move_board(self._board, action)
            if changed:
                actions.append(action)
        return actions

    def _add_random_tile(self) -> None:
        empty_cells = get_empty_cells(self._board)
        if not empty_cells:
            return

        row, column = self._rng.choice(empty_cells)
        value = 2 if self._rng.random() < NEW_TILE_TWO_PROBABILITY else 4
        self._board[row][column] = value
