"""Conversão de estado do 2048 para entrada numérica de agentes."""

from __future__ import annotations

import math
from collections.abc import Sequence

from game2048.constants import ACTIONS
from game2048.core import Board, validate_action, validate_board

BOARD_VECTOR_SIZE = 16
DEFAULT_MAX_TILE_POWER = 16


def board_to_features(
    board: Board,
    max_tile_power: int = DEFAULT_MAX_TILE_POWER,
) -> tuple[float, ...]:
    """Converte o tabuleiro 4x4 em vetor normalizado com 16 valores.

    Blocos vazios viram `0.0`. Blocos positivos usam `log2(valor) /
    max_tile_power`, o que reduz a escala dos números e facilita o trabalho da
    rede neural.
    """
    validate_board(board)
    if max_tile_power <= 0:
        raise ValueError("max_tile_power deve ser maior que zero.")

    features: list[float] = []
    for row in board:
        for value in row:
            if value == 0:
                features.append(0.0)
            else:
                features.append(math.log2(value) / max_tile_power)

    if len(features) != BOARD_VECTOR_SIZE:
        raise ValueError("O estado do DQN espera um tabuleiro 4x4.")

    return tuple(features)


def available_action_mask(available_actions: Sequence[int]) -> tuple[bool, ...]:
    """Cria uma máscara booleana de ações válidas no formato das 4 ações."""
    available = set(available_actions)
    for action in available:
        validate_action(action)
    return tuple(action in available for action in ACTIONS)


def board_to_tensor(board: Board, device: object | None = None) -> object:
    """Converte o tabuleiro para tensor PyTorch com dimensão de batch.

    O import de `torch` fica dentro da função para não exigir PyTorch quando o
    usuário só quer jogar ou rodar os testes básicos no PC.
    """
    import torch

    return torch.tensor(
        [board_to_features(board)],
        dtype=torch.float32,
        device=device,
    )
