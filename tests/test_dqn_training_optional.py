from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch nao instalado")
class DQNTrainingOptionalTest(unittest.TestCase):
    def test_training_can_resume_from_checkpoint(self) -> None:
        from game2048.learning.training import DQNTrainingConfig, train_dqn

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            first_checkpoint = temporary_path / "first.pt"
            resumed_checkpoint = temporary_path / "resumed.pt"

            first_result = train_dqn(
                DQNTrainingConfig(
                    episodes=1,
                    seed=123,
                    batch_size=1,
                    min_replay_size=10_000,
                    memory_capacity=10_000,
                    save_path=str(first_checkpoint),
                    best_save_path=None,
                    metrics_path=None,
                    checkpoint_dir=None,
                    max_steps_per_episode=20,
                )
            )
            resumed_result = train_dqn(
                DQNTrainingConfig(
                    episodes=1,
                    seed=123,
                    batch_size=1,
                    min_replay_size=10_000,
                    memory_capacity=10_000,
                    save_path=str(resumed_checkpoint),
                    best_save_path=None,
                    metrics_path=None,
                    checkpoint_dir=None,
                    max_steps_per_episode=20,
                    resume_from=str(first_checkpoint),
                )
            )

        self.assertEqual(len(first_result.episodes), 1)
        self.assertEqual(len(resumed_result.episodes), 2)
        self.assertEqual(resumed_result.episodes[0].episode, 1)
        self.assertEqual(resumed_result.episodes[-1].episode, 2)

    def test_resume_uses_hidden_size_saved_in_checkpoint(self) -> None:
        from game2048.learning.training import DQNTrainingConfig, train_dqn

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            first_checkpoint = temporary_path / "first.pt"
            resumed_checkpoint = temporary_path / "resumed.pt"

            train_dqn(
                DQNTrainingConfig(
                    episodes=1,
                    seed=123,
                    batch_size=1,
                    min_replay_size=10_000,
                    memory_capacity=10_000,
                    hidden_size=8,
                    save_path=str(first_checkpoint),
                    best_save_path=None,
                    metrics_path=None,
                    checkpoint_dir=None,
                    max_steps_per_episode=20,
                )
            )
            resumed_result = train_dqn(
                DQNTrainingConfig(
                    episodes=1,
                    seed=123,
                    batch_size=1,
                    min_replay_size=10_000,
                    memory_capacity=10_000,
                    hidden_size=16,
                    save_path=str(resumed_checkpoint),
                    best_save_path=None,
                    metrics_path=None,
                    checkpoint_dir=None,
                    max_steps_per_episode=20,
                    resume_from=str(first_checkpoint),
                )
            )

        self.assertEqual(resumed_result.config.hidden_size, 8)
        self.assertEqual(len(resumed_result.episodes), 2)


if __name__ == "__main__":
    unittest.main()
