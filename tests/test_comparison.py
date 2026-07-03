from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from game2048.comparison import (
    AgentComparisonRow,
    build_comparison_rows,
    build_final_report_data,
    compare_dqn_against_baselines,
    format_comparison_table,
    load_training_summary,
    write_comparison_csv,
    write_final_report,
)
from game2048.evaluation import EpisodeResult, EvaluationReport


class ComparisonTest(unittest.TestCase):
    def test_build_comparison_rows_uses_evaluation_summary(self) -> None:
        report = _make_report("random", [20, 40, 60])

        rows = build_comparison_rows((report,))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].agent_name, "random")
        self.assertEqual(rows[0].games, 3)
        self.assertEqual(rows[0].average_score, 40)
        self.assertEqual(rows[0].best_score, 60)
        self.assertEqual(rows[0].worst_score, 20)

    def test_compare_dqn_marks_result_as_learning_when_it_beats_random(self) -> None:
        rows = (
            AgentComparisonRow("random", 10, 100.0, 200, 40, 80.0, 64),
            AgentComparisonRow("heuristic", 10, 1000.0, 1800, 500, 200.0, 512),
            AgentComparisonRow("dqn", 10, 300.0, 800, 80, 120.0, 128),
        )

        verdict = compare_dqn_against_baselines(rows)

        self.assertEqual(verdict["status"], "aprendizado_inicial")
        self.assertFalse(verdict["satisfactory"])
        self.assertEqual(verdict["dqn_average_score"], 300.0)

    def test_compare_dqn_marks_result_as_satisfactory_when_it_beats_heuristic(
        self,
    ) -> None:
        rows = (
            AgentComparisonRow("random", 10, 100.0, 200, 40, 80.0, 64),
            AgentComparisonRow("heuristic", 10, 1000.0, 1800, 500, 200.0, 512),
            AgentComparisonRow("dqn", 10, 1200.0, 2000, 700, 220.0, 1024),
        )

        verdict = compare_dqn_against_baselines(rows)

        self.assertEqual(verdict["status"], "satisfatorio")
        self.assertTrue(verdict["satisfactory"])

    def test_write_final_report_exports_json_with_training_summary(self) -> None:
        rows = (AgentComparisonRow("dqn", 2, 150.0, 200, 100, 50.0, 128),)

        with tempfile.TemporaryDirectory() as temporary_directory:
            metrics_path = Path(temporary_directory) / "metrics.json"
            output_path = Path(temporary_directory) / "final_report.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "best_rolling_average_score": 160.0,
                        "best_save_path": "models/best.pt",
                        "config": {"episodes": 2},
                        "episodes": [
                            {
                                "score": 100,
                                "rolling_average_score": 100.0,
                            },
                            {
                                "score": 200,
                                "rolling_average_score": 150.0,
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            write_final_report(
                rows=rows,
                output_path=output_path,
                training_metrics_path=metrics_path,
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(data["comparison"][0]["agent_name"], "dqn")
        self.assertEqual(data["training_summary"]["episodes"], 2)
        self.assertEqual(data["training_summary"]["last_score"], 200)
        self.assertEqual(data["verdict"]["status"], "nao_satisfatorio")

    def test_write_comparison_csv_exports_one_row_per_agent(self) -> None:
        rows = (
            AgentComparisonRow("random", 2, 50.0, 80, 20, 30.0, 64),
            AgentComparisonRow("dqn", 2, 100.0, 150, 50, 40.0, 128),
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "comparison.csv"

            write_comparison_csv(rows, output_path)

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                exported_rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(exported_rows), 2)
        self.assertEqual(exported_rows[0]["agent_name"], "random")
        self.assertEqual(exported_rows[1]["best_max_tile"], "128")

    def test_format_comparison_table_includes_required_columns(self) -> None:
        rows = (AgentComparisonRow("dqn", 2, 100.0, 150, 50, 40.0, 128),)

        table = format_comparison_table(rows)

        self.assertIn("Score medio", table)
        self.assertIn("Melhor score", table)
        self.assertIn("Pior score", table)
        self.assertIn("dqn", table)

    def test_load_training_summary_returns_none_for_missing_file(self) -> None:
        summary = load_training_summary("missing_metrics.json")

        self.assertIsNone(summary)

    def test_build_final_report_data_keeps_metadata(self) -> None:
        rows = (AgentComparisonRow("dqn", 2, 100.0, 150, 50, 40.0, 128),)

        data = build_final_report_data(rows, metadata={"seed": 42})

        self.assertEqual(data["metadata"]["seed"], 42)
        self.assertEqual(data["comparison"][0]["agent_name"], "dqn")


def _make_report(agent_name: str, scores: list[int]) -> EvaluationReport:
    """Cria relatorios pequenos sem rodar partidas reais."""
    episodes = tuple(
        EpisodeResult(
            agent_name=agent_name,
            episode=index + 1,
            seed=index,
            score=score,
            steps=10 + index,
            max_tile=64,
        )
        for index, score in enumerate(scores)
    )
    return EvaluationReport(agent_name=agent_name, episodes=episodes)


if __name__ == "__main__":
    unittest.main()
