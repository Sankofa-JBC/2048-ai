from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game2048.learning.replay_memory import ReplayMemory, Transition


def make_transition(index: int) -> Transition:
    """Cria uma transição pequena para testar a memória sem depender do jogo."""
    state = tuple([float(index)] * 16)
    next_state = tuple([float(index + 1)] * 16)
    return Transition(
        state=state,
        action=index % 4,
        reward=float(index),
        next_state=next_state,
        done=False,
        next_available_actions=(0, 1),
    )


class ReplayMemoryTest(unittest.TestCase):
    def test_push_and_sample_transitions(self) -> None:
        memory = ReplayMemory(capacity=10, seed=1)
        for index in range(5):
            memory.push(make_transition(index))

        sample = memory.sample(3)

        self.assertEqual(len(memory), 5)
        self.assertEqual(len(sample), 3)
        self.assertTrue(all(isinstance(item, Transition) for item in sample))

    def test_capacity_discards_oldest_items(self) -> None:
        memory = ReplayMemory(capacity=2, seed=1)
        memory.push(make_transition(1))
        memory.push(make_transition(2))
        memory.push(make_transition(3))

        sample = memory.sample(2)
        rewards = sorted(item.reward for item in sample)

        self.assertEqual(rewards, [2.0, 3.0])

    def test_rejects_invalid_capacity(self) -> None:
        with self.assertRaises(ValueError):
            ReplayMemory(capacity=0)

    def test_rejects_invalid_sample_size(self) -> None:
        memory = ReplayMemory(capacity=2, seed=1)
        memory.push(make_transition(1))

        with self.assertRaises(ValueError):
            memory.sample(0)

        with self.assertRaises(ValueError):
            memory.sample(2)


if __name__ == "__main__":
    unittest.main()
