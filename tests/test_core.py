from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game2048.constants import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP
from game2048.core import can_move, move_board, slide_and_merge_row_left


class SlideAndMergeRowLeftTest(unittest.TestCase):
    def test_merges_simple_pair(self) -> None:
        row, score = slide_and_merge_row_left([2, 2, 0, 0])

        self.assertEqual(row, [4, 0, 0, 0])
        self.assertEqual(score, 4)

    def test_merges_each_tile_once(self) -> None:
        row, score = slide_and_merge_row_left([2, 2, 2, 0])

        self.assertEqual(row, [4, 2, 0, 0])
        self.assertEqual(score, 4)

    def test_merges_two_pairs(self) -> None:
        row, score = slide_and_merge_row_left([2, 2, 2, 2])

        self.assertEqual(row, [4, 4, 0, 0])
        self.assertEqual(score, 8)

    def test_keeps_different_values_separate(self) -> None:
        row, score = slide_and_merge_row_left([4, 4, 8, 8])

        self.assertEqual(row, [8, 16, 0, 0])
        self.assertEqual(score, 24)


class MoveBoardTest(unittest.TestCase):
    def test_moves_left(self) -> None:
        board = [
            [2, 0, 2, 0],
            [4, 4, 0, 0],
            [0, 0, 0, 0],
            [2, 4, 8, 16],
        ]

        moved, score, changed = move_board(board, ACTION_LEFT)

        self.assertEqual(
            moved,
            [
                [4, 0, 0, 0],
                [8, 0, 0, 0],
                [0, 0, 0, 0],
                [2, 4, 8, 16],
            ],
        )
        self.assertEqual(score, 12)
        self.assertTrue(changed)

    def test_moves_right(self) -> None:
        board = [
            [2, 0, 2, 0],
            [4, 4, 0, 0],
            [0, 0, 0, 0],
            [2, 4, 8, 16],
        ]

        moved, score, changed = move_board(board, ACTION_RIGHT)

        self.assertEqual(
            moved,
            [
                [0, 0, 0, 4],
                [0, 0, 0, 8],
                [0, 0, 0, 0],
                [2, 4, 8, 16],
            ],
        )
        self.assertEqual(score, 12)
        self.assertTrue(changed)

    def test_moves_up(self) -> None:
        board = [
            [2, 0, 0, 2],
            [2, 4, 0, 2],
            [0, 4, 8, 0],
            [0, 0, 8, 0],
        ]

        moved, score, changed = move_board(board, ACTION_UP)

        self.assertEqual(
            moved,
            [
                [4, 8, 16, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )
        self.assertEqual(score, 32)
        self.assertTrue(changed)

    def test_moves_down(self) -> None:
        board = [
            [2, 0, 0, 2],
            [2, 4, 0, 2],
            [0, 4, 8, 0],
            [0, 0, 8, 0],
        ]

        moved, score, changed = move_board(board, ACTION_DOWN)

        self.assertEqual(
            moved,
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [4, 8, 16, 4],
            ],
        )
        self.assertEqual(score, 32)
        self.assertTrue(changed)

    def test_detects_unchanged_move(self) -> None:
        board = [
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        moved, score, changed = move_board(board, ACTION_LEFT)

        self.assertEqual(moved, board)
        self.assertEqual(score, 0)
        self.assertFalse(changed)


class CanMoveTest(unittest.TestCase):
    def test_can_move_when_there_is_an_empty_cell(self) -> None:
        board = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 0],
        ]

        self.assertTrue(can_move(board))

    def test_can_move_when_adjacent_tiles_match(self) -> None:
        board = [
            [2, 4, 8, 16],
            [32, 64, 128, 256],
            [512, 1024, 2048, 4096],
            [8192, 16384, 32768, 32768],
        ]

        self.assertTrue(can_move(board))

    def test_cannot_move_when_board_is_full_and_no_tiles_match(self) -> None:
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]

        self.assertFalse(can_move(board))


if __name__ == "__main__":
    unittest.main()
