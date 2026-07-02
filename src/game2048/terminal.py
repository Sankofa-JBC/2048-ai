"""Funções auxiliares de terminal para visualização local leve."""

from __future__ import annotations

import os


def clear_terminal() -> None:
    """Limpa o terminal sem adicionar dependência gráfica."""
    os.system("cls" if os.name == "nt" else "clear")


def print_board(board: list[list[int]], score: int) -> None:
    """Imprime um tabuleiro 2048 legível usando apenas caracteres do terminal."""
    print("2048")
    print(f"Score: {score}")
    print()

    # A largura cresce quando blocos grandes aparecem, mantendo o tabuleiro alinhado.
    cell_width = max(4, len(str(max(max(row) for row in board))))
    horizontal_border = "+" + "+".join("-" * (cell_width + 2) for _ in board) + "+"

    print(horizontal_border)
    for row in board:
        cells = []
        for value in row:
            text = "" if value == 0 else str(value)
            cells.append(f" {text:>{cell_width}} ")
        print("|" + "|".join(cells) + "|")
        print(horizontal_border)
    print()
