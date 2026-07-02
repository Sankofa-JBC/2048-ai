"""Agente heurístico de referência simples para 2048.

Este agente ainda não aprende. Ele é uma referência mais forte que o agente
aleatório porque avalia o tabuleiro imediato produzido por cada movimento válido.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from game2048.agents.base import normalize_available_actions
from game2048.constants import ACTIONS, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP
from game2048.core import Board, get_empty_cells, move_board, validate_board


DEFAULT_ACTION_PREFERENCE = (
    ACTION_LEFT,
    ACTION_DOWN,
    ACTION_RIGHT,
    ACTION_UP,
)


@dataclass(frozen=True)
class HeuristicWeights:
    """Pesos usados para pontuar movimentos candidatos.

    Os valores são intencionalmente conservadores. Eles tornam a heurística melhor
    que jogadas aleatórias sem transformá-la em um resolvedor manual complexo.
    """

    score_gain: int = 4
    empty_cells: int = 100
    max_tile: int = 1
    corner_max_tile: int = 2


class HeuristicAgent:
    """Escolhe o movimento válido com a melhor pontuação heurística imediata."""

    name = "heuristic"

    def __init__(
        self,
        weights: HeuristicWeights | None = None,
        action_preference: Sequence[int] = DEFAULT_ACTION_PREFERENCE,
    ) -> None:
        # A ordem de desempate mantém o agente determinístico e o direciona a
        # preservar uma estratégia de canto estável sem manter estado extra.
        self.weights = weights or HeuristicWeights()
        normalized_preference = normalize_available_actions(action_preference)
        if (
            len(normalized_preference) != len(ACTIONS)
            or set(normalized_preference) != set(ACTIONS)
        ):
            raise ValueError("A preferência de ações deve incluir cada ação do 2048 uma vez.")
        self._action_preference = normalized_preference

    def choose_action(self, board: Board, available_actions: Sequence[int]) -> int:
        """Retorna a ação que cria a melhor avaliação de tabuleiro em um passo."""
        validate_board(board)
        actions = normalize_available_actions(available_actions)
        preferred_actions = self._ordered_actions(actions)

        best_action = preferred_actions[0]
        best_score = self._score_action(board, best_action)

        for action in preferred_actions[1:]:
            candidate_score = self._score_action(board, action)
            if candidate_score > best_score:
                best_action = action
                best_score = candidate_score

        return best_action

    def _ordered_actions(self, actions: tuple[int, ...]) -> tuple[int, ...]:
        """Ordena ações pela preferência determinística de desempate configurada."""
        return tuple(
            sorted(actions, key=lambda action: self._action_preference.index(action))
        )

    def _score_action(self, board: Board, action: int) -> int:
        """Pontua o tabuleiro produzido por uma ação candidata."""
        moved_board, score_gain, changed = move_board(board, action)
        if not changed:
            return -1

        max_tile = _max_tile(moved_board)
        empty_cell_count = len(get_empty_cells(moved_board))
        corner_bonus = max_tile if _max_tile_is_in_corner(moved_board, max_tile) else 0

        return (
            score_gain * self.weights.score_gain
            + empty_cell_count * self.weights.empty_cells
            + max_tile * self.weights.max_tile
            + corner_bonus * self.weights.corner_max_tile
        )


def _max_tile(board: Board) -> int:
    """Retorna o maior bloco em um tabuleiro."""
    return max(max(row) for row in board)


def _max_tile_is_in_corner(board: Board, max_tile: int) -> bool:
    """Retorna se o maior bloco está atualmente em um dos quatro cantos."""
    last_index = len(board) - 1
    corners = (
        board[0][0],
        board[0][last_index],
        board[last_index][0],
        board[last_index][last_index],
    )
    return max_tile in corners
