"""Geracao opcional de graficos para resultados de treino e comparacao."""

from __future__ import annotations

import json
from pathlib import Path

from game2048.comparison import AgentComparisonRow


def build_comparison_plot_data(
    rows: list[AgentComparisonRow] | tuple[AgentComparisonRow, ...],
) -> dict[str, list[float | str]]:
    """Prepara dados agregados para grafico de comparacao entre agentes."""
    return {
        "agent_names": [row.agent_name for row in rows],
        "average_scores": [row.average_score for row in rows],
        "best_scores": [float(row.best_score) for row in rows],
    }


def load_training_plot_data(
    metrics_path: str | Path | None,
) -> dict[str, list[float]] | None:
    """Extrai pontos do historico de treino para grafico."""
    if metrics_path is None:
        return None

    path = Path(metrics_path)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = data.get("episodes", [])
    if not episodes:
        return None

    return {
        "episode_numbers": [float(item["episode"]) for item in episodes],
        "scores": [float(item["score"]) for item in episodes],
        "rolling_average_scores": [
            float(item["rolling_average_score"]) for item in episodes
        ],
        "max_tiles": [float(item["max_tile"]) for item in episodes],
    }


def write_experiment_plots(
    rows: list[AgentComparisonRow] | tuple[AgentComparisonRow, ...],
    output_dir: str | Path,
    training_metrics_path: str | Path | None = None,
) -> tuple[str, ...]:
    """Gera PNGs de comparacao e treino quando matplotlib estiver disponivel."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return ()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_files: list[str] = []

    comparison_plot_path = output_path / "comparison_scores.png"
    comparison_data = build_comparison_plot_data(rows)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(
        comparison_data["agent_names"],
        comparison_data["average_scores"],
        color=["#d97706", "#0f766e", "#2563eb"][: len(rows)],
    )
    axis.set_title("Score medio por agente")
    axis.set_ylabel("Score medio")
    figure.tight_layout()
    figure.savefig(comparison_plot_path, dpi=160)
    plt.close(figure)
    generated_files.append(str(comparison_plot_path))

    training_data = load_training_plot_data(training_metrics_path)
    if training_data is not None:
        training_plot_path = output_path / "training_progress.png"
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(
            training_data["episode_numbers"],
            training_data["scores"],
            label="score",
            alpha=0.35,
        )
        axis.plot(
            training_data["episode_numbers"],
            training_data["rolling_average_scores"],
            label="media movel",
            linewidth=2.0,
        )
        axis.set_title("Evolucao do treino DQN")
        axis.set_xlabel("Episodio")
        axis.set_ylabel("Score")
        axis.legend()
        figure.tight_layout()
        figure.savefig(training_plot_path, dpi=160)
        plt.close(figure)
        generated_files.append(str(training_plot_path))

    return tuple(generated_files)
