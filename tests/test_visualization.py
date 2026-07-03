from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game2048.comparison import AgentComparisonRow
from game2048.visualization import (
    build_comparison_plot_data,
    load_training_plot_data,
)


class VisualizationTest(unittest.TestCase):
    def test_build_comparison_plot_data_extracts_names_and_scores(self) -> None:
        rows = (
            AgentComparisonRow("random", 2, 50.0, 80, 20, 30.0, 64),
            AgentComparisonRow("dqn", 2, 100.0, 150, 50, 40.0, 128),
        )

        data = build_comparison_plot_data(rows)

        self.assertEqual(data["agent_names"], ["random", "dqn"])
        self.assertEqual(data["average_scores"], [50.0, 100.0])

    def test_load_training_plot_data_reads_episode_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics_path = Path(temporary_directory) / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "episode": 1,
                                "score": 100,
                                "rolling_average_score": 100.0,
                                "max_tile": 16,
                            },
                            {
                                "episode": 2,
                                "score": 200,
                                "rolling_average_score": 150.0,
                                "max_tile": 32,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            data = load_training_plot_data(metrics_path)

        self.assertEqual(data["episode_numbers"], [1.0, 2.0])
        self.assertEqual(data["scores"], [100.0, 200.0])
        self.assertEqual(data["rolling_average_scores"], [100.0, 150.0])

    def test_load_training_plot_data_returns_none_when_missing(self) -> None:
        self.assertIsNone(load_training_plot_data("missing_metrics.json"))


if __name__ == "__main__":
    unittest.main()
