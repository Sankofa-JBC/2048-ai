from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game2048 import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, Game2048


class Game2048Test(unittest.TestCase):
    def test_reset_starts_with_two_tiles(self) -> None:
        game = Game2048(seed=1)
        board = game.board
        tiles = [value for row in board for value in row if value != 0]

        self.assertEqual(len(tiles), 2)
        self.assertTrue(all(value in (2, 4) for value in tiles))
        self.assertEqual(game.score, 0)
        self.assertFalse(game.done)

    def test_step_moves_board_and_updates_score(self) -> None:
        game = Game2048(seed=1)
        game.set_board(
            [
                [2, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        board, reward, done, info = game.step(ACTION_LEFT)

        self.assertEqual(board[0][0], 4)
        self.assertEqual(reward, 4)
        self.assertEqual(game.score, 4)
        self.assertTrue(info["changed"])
        self.assertEqual(
            info["board_before_spawn"],
            [
                [4, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
        )
        self.assertFalse(done)

    def test_unchanged_step_does_not_add_tile_or_step_count(self) -> None:
        game = Game2048(seed=1)
        game.set_board(
            [
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        before = game.board

        board, reward, done, info = game.step(ACTION_LEFT)

        self.assertEqual(board, before)
        self.assertEqual(reward, 0)
        self.assertEqual(game.steps, 0)
        self.assertFalse(info["changed"])
        self.assertEqual(info["board_before_spawn"], before)
        self.assertFalse(done)

    def test_game_over_board_is_done(self) -> None:
        game = Game2048(seed=1)
        game.set_board(
            [
                [2, 4, 2, 4],
                [4, 2, 4, 2],
                [2, 4, 2, 4],
                [4, 2, 4, 2],
            ]
        )

        board, reward, done, info = game.step(ACTION_RIGHT)

        self.assertTrue(done)
        self.assertEqual(board, game.board)
        self.assertEqual(reward, 0)
        self.assertEqual(info["reason"], "game_over")

    def test_available_actions_returns_only_effective_actions(self) -> None:
        game = Game2048(seed=1)
        game.set_board(
            [
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        self.assertEqual(game.available_actions(), [ACTION_RIGHT, ACTION_DOWN])


if __name__ == "__main__":
    unittest.main()
