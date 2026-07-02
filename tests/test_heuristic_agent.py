from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game2048 import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, Agent, HeuristicAgent


class HeuristicAgentTest(unittest.TestCase):
    def test_agent_matches_base_protocol(self) -> None:
        agent = HeuristicAgent()

        self.assertIsInstance(agent, Agent)

    def test_choose_action_prefers_better_immediate_merge(self) -> None:
        agent = HeuristicAgent()
        board = [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]

        action = agent.choose_action(
            board=board,
            available_actions=[ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN],
        )

        self.assertEqual(action, ACTION_LEFT)

    def test_choose_action_rejects_empty_action_list(self) -> None:
        agent = HeuristicAgent()

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

    def test_constructor_rejects_incomplete_action_preference(self) -> None:
        with self.assertRaises(ValueError):
            HeuristicAgent(action_preference=[ACTION_LEFT])

    def test_constructor_rejects_duplicate_action_preference(self) -> None:
        with self.assertRaises(ValueError):
            HeuristicAgent(
                action_preference=[ACTION_LEFT, ACTION_LEFT, ACTION_RIGHT, ACTION_DOWN]
            )


if __name__ == "__main__":
    unittest.main()
