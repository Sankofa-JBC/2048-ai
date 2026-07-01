"""Terminal helpers for lightweight local visualization."""

from __future__ import annotations

import os


def clear_terminal() -> None:
    """Clear the terminal without adding a graphical dependency."""
    os.system("cls" if os.name == "nt" else "clear")


def print_board(board: list[list[int]], score: int) -> None:
    """Print a readable 2048 board using plain terminal characters."""
    print("2048")
    print(f"Score: {score}")
    print()

    # The width grows when large tiles appear, so the board remains aligned.
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
