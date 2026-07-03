from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@unittest.skipIf(importlib.util.find_spec("torch") is None, "PyTorch nao instalado")
class DQNModelOptionalTest(unittest.TestCase):
    def test_model_returns_one_q_value_per_action(self) -> None:
        import torch

        from game2048.learning.dqn_model import DQNModel

        model = DQNModel()
        output = model(torch.zeros((1, 16), dtype=torch.float32))

        self.assertEqual(tuple(output.shape), (1, 4))


if __name__ == "__main__":
    unittest.main()
