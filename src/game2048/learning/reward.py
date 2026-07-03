"""Funcoes puras de reward shaping para o DQN do 2048."""

from __future__ import annotations

import math
from dataclasses import dataclass

from game2048.core import Board, get_empty_cells, validate_board


@dataclass(frozen=True)
class RewardShapingWeights:
    """Pesos de shaping para dar sinais mais densos ao DQN."""

    merge_scale: float = 1.0 / 2048.0
    empty_cell_weight: float = 0.02
    max_tile_weight: float = 0.05
    corner_max_tile_weight: float = 0.02
    monotonicity_weight: float = 0.01
    terminal_penalty: float = 0.5


def calculate_shaped_reward(
    board_before_action: Board,
    board_after_move: Board,
    merge_reward: int,
    done: bool,
    weights: RewardShapingWeights,
) -> float:
    """Combina score do merge com sinais densos de qualidade do tabuleiro."""
    validate_board(board_before_action)
    validate_board(board_after_move)

    merge_component = merge_reward * weights.merge_scale
    empty_cell_component = (
        empty_cell_count(board_after_move) - empty_cell_count(board_before_action)
    ) * weights.empty_cell_weight
    max_tile_component = max(
        0.0,
        log2_max_tile(board_after_move) - log2_max_tile(board_before_action),
    ) * weights.max_tile_weight
    corner_component = (
        float(max_tile_in_corner(board_after_move))
        - float(max_tile_in_corner(board_before_action))
    ) * weights.corner_max_tile_weight
    monotonicity_component = (
        monotonicity_score(board_after_move) - monotonicity_score(board_before_action)
    ) * weights.monotonicity_weight
    terminal_component = -weights.terminal_penalty if done else 0.0

    return (
        merge_component
        + empty_cell_component
        + max_tile_component
        + corner_component
        + monotonicity_component
        + terminal_component
    )


def empty_cell_count(board: Board) -> int:
    """Conta quantas casas vazias o tabuleiro possui."""
    validate_board(board)
    return len(get_empty_cells(board))


def log2_max_tile(board: Board) -> float:
    """Retorna log2 do maior bloco, ou zero quando o tabuleiro esta vazio."""
    validate_board(board)
    max_tile = max(max(row) for row in board)
    if max_tile <= 0:
        return 0.0
    return math.log2(max_tile)


def max_tile_in_corner(board: Board) -> bool:
    """Indica se o maior bloco atual esta em um dos cantos."""
    validate_board(board)
    last_index = len(board) - 1
    max_tile = max(max(row) for row in board)
    corners = (
        board[0][0],
        board[0][last_index],
        board[last_index][0],
        board[last_index][last_index],
    )
    return max_tile in corners


def monotonicity_score(board: Board) -> float:
    """Mede o quao monotono esta o tabuleiro em linhas e colunas."""
    validate_board(board)
    values = [[_log2_or_zero(value) for value in row] for row in board]
    line_scores = [_line_monotonicity_score(row) for row in values]
    line_scores.extend(
        _line_monotonicity_score([values[row][column] for row in range(len(values))])
        for column in range(len(values[0]))
    )
    return sum(line_scores) / len(line_scores)


def _line_monotonicity_score(values: list[float]) -> float:
    """Pontua uma linha como monotona em qualquer direcao."""
    if len(values) < 2:
        return 1.0

    non_increasing = 0
    non_decreasing = 0
    comparisons = len(values) - 1
    for index in range(comparisons):
        left = values[index]
        right = values[index + 1]
        if left >= right:
            non_increasing += 1
        if left <= right:
            non_decreasing += 1

    return max(non_increasing, non_decreasing) / comparisons


def _log2_or_zero(value: int) -> float:
    """Converte bloco em log2 preservando zeros como zero."""
    if value <= 0:
        return 0.0
    return math.log2(value)
