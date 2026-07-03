from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from game2048.cli.run_experiment import _resolve_resume_checkpoint


class RunExperimentCliTest(unittest.TestCase):
    def test_compare_only_flow_exports_report_without_dqn(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project_dir = Path(temporary_directory)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "game2048.cli.run_experiment",
                    "--project-dir",
                    str(project_dir),
                    "--games",
                    "2",
                    "--skip-train",
                    "--allow-missing-dqn",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            report_path = project_dir / "results/final_report.json"
            comparison_path = project_dir / "results/final_comparison.csv"

            self.assertIn("Comparacao final dos agentes", completed.stdout)
            self.assertTrue(report_path.exists())
            self.assertTrue(comparison_path.exists())

            report_data = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report_data["verdict"]["status"], "sem_dqn")
            self.assertEqual(len(report_data["comparison"]), 2)

    def test_auto_resume_uses_best_checkpoint_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            best_model_path = Path(temporary_directory) / "dqn_2048_best.pt"
            best_model_path.write_bytes(b"checkpoint")

            resume_path, resumed_automatically = _resolve_resume_checkpoint(
                requested_path=None,
                best_model_path=best_model_path,
                skip_train=False,
                force_fresh=False,
            )

        self.assertEqual(resume_path, best_model_path)
        self.assertTrue(resumed_automatically)

    def test_force_fresh_ignores_existing_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            best_model_path = Path(temporary_directory) / "dqn_2048_best.pt"
            best_model_path.write_bytes(b"checkpoint")

            resume_path, resumed_automatically = _resolve_resume_checkpoint(
                requested_path=None,
                best_model_path=best_model_path,
                skip_train=False,
                force_fresh=True,
            )

        self.assertIsNone(resume_path)
        self.assertFalse(resumed_automatically)


if __name__ == "__main__":
    unittest.main()
