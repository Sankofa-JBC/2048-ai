from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game2048 import RandomAgent
from game2048.evaluation import evaluate_agent, write_report


class EvaluationTest(unittest.TestCase):
    def test_evaluate_agent_collects_episode_and_summary_metrics(self) -> None:
        report = evaluate_agent(
            agent_factory=lambda seed: RandomAgent(seed=seed),
            games=3,
            seed=42,
            agent_name="random",
        )

        self.assertEqual(report.games, 3)
        self.assertEqual(len(report.episodes), 3)
        self.assertEqual(report.agent_name, "random")
        self.assertGreater(report.average_score, 0)
        self.assertGreaterEqual(report.best_score, report.worst_score)
        self.assertGreater(report.best_max_tile, 0)
        self.assertIn(64, report.tile_reach_rates())

    def test_evaluate_agent_rejects_invalid_game_count(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_agent(
                agent_factory=lambda seed: RandomAgent(seed=seed),
                games=0,
            )

    def test_write_report_exports_json(self) -> None:
        report = evaluate_agent(
            agent_factory=lambda seed: RandomAgent(seed=seed),
            games=2,
            seed=7,
            agent_name="random",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "report.json"

            write_report(report, output_path)

            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(data["summary"]["agent_name"], "random")
        self.assertEqual(len(data["episodes"]), 2)
        self.assertIn("64", data["tile_reach_rates"])

    def test_write_report_exports_csv(self) -> None:
        report = evaluate_agent(
            agent_factory=lambda seed: RandomAgent(seed=seed),
            games=2,
            seed=7,
            agent_name="random",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "report.csv"

            write_report(report, output_path)

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["agent_name"], "random")
        self.assertIn("reached_64", rows[0])

    def test_write_report_rejects_unknown_extension(self) -> None:
        report = evaluate_agent(
            agent_factory=lambda seed: RandomAgent(seed=seed),
            games=1,
            seed=7,
            agent_name="random",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "report.txt"

            with self.assertRaises(ValueError):
                write_report(report, output_path)


if __name__ == "__main__":
    unittest.main()
