from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game2048.learning.reward import (
    RewardShapingWeights,
    calculate_shaped_reward,
    max_tile_in_corner,
    monotonicity_score,
)


class RewardShapingTest(unittest.TestCase):
    def test_shaped_reward_rewards_empty_space_and_bigger_tile(self) -> None:
        before = [
            [2, 2, 4, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
        after = [
            [4, 4, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        shaped_reward = calculate_shaped_reward(
            board_before_action=before,
            board_after_move=after,
            merge_reward=4,
            done=False,
            weights=RewardShapingWeights(),
        )

        self.assertGreater(shaped_reward, 0.0)

    def test_shaped_reward_penalizes_terminal_state(self) -> None:
        board = [
            [2, 4, 2, 4],
            [4, 2, 4, 2],
            [2, 4, 2, 4],
            [4, 2, 4, 2],
        ]

        shaped_reward = calculate_shaped_reward(
            board_before_action=board,
            board_after_move=board,
            merge_reward=0,
            done=True,
            weights=RewardShapingWeights(terminal_penalty=1.0),
        )

        self.assertLess(shaped_reward, 0.0)

    def test_max_tile_in_corner_detects_corner_bonus(self) -> None:
        board = [
            [16, 8, 4, 2],
            [8, 4, 2, 0],
            [4, 2, 0, 0],
            [2, 0, 0, 0],
        ]

        self.assertTrue(max_tile_in_corner(board))

    def test_monotonicity_score_prefers_ordered_board(self) -> None:
        monotonic_board = [
            [16, 8, 4, 2],
            [8, 4, 2, 0],
            [4, 2, 0, 0],
            [2, 0, 0, 0],
        ]
        noisy_board = [
            [2, 16, 4, 8],
            [4, 2, 8, 0],
            [0, 4, 2, 16],
            [8, 0, 4, 2],
        ]

        self.assertGreater(monotonicity_score(monotonic_board), monotonicity_score(noisy_board))


if __name__ == "__main__":
    unittest.main()
