from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game2048 import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, Agent, RandomAgent


class RandomAgentTest(unittest.TestCase):
    def test_agent_matches_base_protocol(self) -> None:
        agent = RandomAgent(seed=1)

        self.assertIsInstance(agent, Agent)

    def test_choose_action_returns_one_available_action(self) -> None:
        agent = RandomAgent(seed=1)

        action = agent.choose_action(
            board=[
                [2, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            available_actions=[ACTION_RIGHT, ACTION_DOWN],
        )

        self.assertIn(action, [ACTION_RIGHT, ACTION_DOWN])

    def test_seed_makes_choices_reproducible(self) -> None:
        first_agent = RandomAgent(seed=42)
        second_agent = RandomAgent(seed=42)
        actions = [ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]
        board = [
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        first_choices = [first_agent.choose_action(board, actions) for _ in range(10)]
        second_choices = [second_agent.choose_action(board, actions) for _ in range(10)]

        self.assertEqual(first_choices, second_choices)

    def test_choose_action_rejects_empty_action_list(self) -> None:
        agent = RandomAgent(seed=1)

        with self.assertRaises(ValueError):
            agent.choose_action(
                board=[
                    [2, 4, 2, 4],
                    [4, 2, 4, 2],
                    [2, 4, 2, 4],
                    [4, 2, 4, 2],
                ],
                available_actions=[],
            )

    def test_choose_action_rejects_invalid_action(self) -> None:
        agent = RandomAgent(seed=1)

        with self.assertRaises(ValueError):
            agent.choose_action(
                board=[
                    [2, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                available_actions=[99],
            )


if __name__ == "__main__":
    unittest.main()
