"""Regras puras do tabuleiro 2048.

Este módulo evita entrada, saída e aleatoriedade para ser testado diretamente
e reutilizado tanto por uma interface humana quanto por um agente de IA.
"""

from __future__ import annotations

from game2048.constants import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    ACTIONS,
)

Board = list[list[int]]


def clone_board(board: Board) -> Board:
    """Retorna uma cópia do tabuleiro para proteger o estado interno."""
    return [row[:] for row in board]


def empty_board(size: int = 4) -> Board:
    return [[0 for _ in range(size)] for _ in range(size)]


def validate_board(board: Board) -> None:
    if not board:
        raise ValueError("O tabuleiro não pode estar vazio.")

    size = len(board)
    for row in board:
        if len(row) != size:
            raise ValueError("O tabuleiro precisa ser quadrado.")
        if any(value < 0 for value in row):
            raise ValueError("Os valores do tabuleiro devem ser não negativos.")


def validate_action(action: int) -> None:
    if action not in ACTIONS:
        raise ValueError(f"Ação inválida: {action}.")


def slide_and_merge_row_left(row: list[int]) -> tuple[list[int], int]:
    """Desliza uma linha para a esquerda e junta blocos iguais uma vez por movimento."""
    tiles = [value for value in row if value != 0]
    merged: list[int] = []
    score_gain = 0
    index = 0

    while index < len(tiles):
        current = tiles[index]
        next_index = index + 1

        if next_index < len(tiles) and current == tiles[next_index]:
            merged_value = current * 2
            merged.append(merged_value)
            score_gain += merged_value
            index += 2
        else:
            merged.append(current)
            index += 1

    merged.extend([0] * (len(row) - len(merged)))
    return merged, score_gain


def move_board(board: Board, action: int) -> tuple[Board, int, bool]:
    """Retorna o tabuleiro após uma ação, a pontuação ganha e se houve mudança."""
    validate_board(board)
    validate_action(action)

    if action == ACTION_LEFT:
        moved_board, score_gain = _move_left(board)
    elif action == ACTION_RIGHT:
        reversed_board = _reverse_rows(board)
        moved_reversed_board, score_gain = _move_left(reversed_board)
        moved_board = _reverse_rows(moved_reversed_board)
    elif action == ACTION_UP:
        transposed_board = _transpose(board)
        moved_transposed_board, score_gain = _move_left(transposed_board)
        moved_board = _transpose(moved_transposed_board)
    else:
        transposed_board = _transpose(board)
        reversed_board = _reverse_rows(transposed_board)
        moved_reversed_board, score_gain = _move_left(reversed_board)
        moved_board = _transpose(_reverse_rows(moved_reversed_board))

    changed = moved_board != board
    return moved_board, score_gain, changed


def get_empty_cells(board: Board) -> list[tuple[int, int]]:
    validate_board(board)
    return [
        (row_index, column_index)
        for row_index, row in enumerate(board)
        for column_index, value in enumerate(row)
        if value == 0
    ]


def has_empty_cell(board: Board) -> bool:
    return len(get_empty_cells(board)) > 0


def can_move(board: Board) -> bool:
    validate_board(board)

    if has_empty_cell(board):
        return True

    size = len(board)
    for row in range(size):
        for column in range(size):
            current = board[row][column]
            if column + 1 < size and current == board[row][column + 1]:
                return True
            if row + 1 < size and current == board[row + 1][column]:
                return True

    return False


def _move_left(board: Board) -> tuple[Board, int]:
    moved_board: Board = []
    total_score_gain = 0

    for row in board:
        moved_row, score_gain = slide_and_merge_row_left(row)
        moved_board.append(moved_row)
        total_score_gain += score_gain

    return moved_board, total_score_gain


def _reverse_rows(board: Board) -> Board:
    return [list(reversed(row)) for row in board]


def _transpose(board: Board) -> Board:
    return [list(column) for column in zip(*board)]
