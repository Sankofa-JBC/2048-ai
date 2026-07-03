from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game2048 import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP
from game2048.learning.state import available_action_mask, board_to_features


class LearningStateTest(unittest.TestCase):
    def test_board_to_features_flattens_and_normalizes_board(self) -> None:
        board = [
            [0, 2, 4, 8],
            [16, 32, 64, 128],
            [256, 512, 1024, 2048],
            [4096, 8192, 16384, 32768],
        ]

        features = board_to_features(board)

        self.assertEqual(len(features), 16)
        self.assertEqual(features[0], 0.0)
        self.assertEqual(features[1], 1 / 16)
        self.assertEqual(features[2], 2 / 16)
        self.assertEqual(features[-1], 15 / 16)

    def test_board_to_features_rejects_invalid_max_tile_power(self) -> None:
        with self.assertRaises(ValueError):
            board_to_features([[0, 0, 0, 0] for _ in range(4)], max_tile_power=0)

    def test_available_action_mask_marks_valid_actions(self) -> None:
        mask = available_action_mask([ACTION_LEFT, ACTION_DOWN])

        self.assertEqual(
            mask,
            (
                False,
                False,
                True,
                True,
            ),
        )

    def test_available_action_mask_accepts_all_actions(self) -> None:
        mask = available_action_mask([ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT])

        self.assertEqual(mask, (True, True, True, True))


if __name__ == "__main__":
    unittest.main()
