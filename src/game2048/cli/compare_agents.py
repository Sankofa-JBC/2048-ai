"""CLI instalavel para comparar agentes de 2048."""

from __future__ import annotations

import argparse
from pathlib import Path

from game2048 import HeuristicAgent, RandomAgent
from game2048.comparison import (
    build_comparison_rows,
    format_comparison_table,
    write_comparison_csv,
    write_final_report,
)
from game2048.evaluation import EvaluationReport, evaluate_agent


def main() -> None:
    """Compara os agentes principais e salva um relatorio final do experimento."""
    parser = argparse.ArgumentParser(
        description="Compara random, heuristic e DQN em um unico relatorio."
    )
    parser.add_argument("--games", type=int, default=100, help="numero de partidas")
    parser.add_argument("--seed", type=int, default=42, help="seed base")
    parser.add_argument(
        "--dqn-model",
        type=Path,
        default=Path("models/dqn_2048_best.pt"),
        help="checkpoint DQN que sera avaliado",
    )
    parser.add_argument(
        "--training-metrics",
        type=Path,
        default=Path("results/dqn_training_metrics.json"),
        help="JSON com historico do treino DQN",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/final_report.json"),
        help="relatorio final consolidado em JSON",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/final_comparison.csv"),
        help="tabela comparativa agregada em CSV",
    )
    parser.add_argument("--device", default="cpu", help="device PyTorch do DQN")
    parser.add_argument(
        "--allow-missing-dqn",
        action="store_true",
        help="permite comparar apenas os baselines quando o checkpoint DQN nao existe",
    )
    args = parser.parse_args()

    reports = [
        _evaluate_random(games=args.games, seed=args.seed),
        _evaluate_heuristic(games=args.games, seed=args.seed),
    ]

    if args.dqn_model.exists():
        reports.append(
            _evaluate_dqn(
                model_path=args.dqn_model,
                games=args.games,
                seed=args.seed,
                device=args.device,
            )
        )
    elif not args.allow_missing_dqn:
        raise SystemExit(
            "Modelo DQN nao encontrado. Treine primeiro ou use "
            "--allow-missing-dqn para comparar apenas os baselines."
        )

    rows = build_comparison_rows(reports)
    write_comparison_csv(rows, args.output_csv)
    final_report = write_final_report(
        rows=rows,
        output_path=args.output_json,
        training_metrics_path=args.training_metrics,
        metadata={
            "games": args.games,
            "seed": args.seed,
            "dqn_model": str(args.dqn_model),
        },
    )

    print("Comparacao final dos agentes")
    print(format_comparison_table(rows))
    print()
    print("Conclusao automatica")
    print(final_report["verdict"]["message"])
    print()
    print(f"Relatorio JSON: {args.output_json}")
    print(f"Tabela CSV: {args.output_csv}")


def _evaluate_random(games: int, seed: int | None) -> EvaluationReport:
    """Avalia o agente aleatorio criando uma instancia por partida."""
    return evaluate_agent(
        agent_factory=lambda episode_seed: RandomAgent(seed=episode_seed),
        games=games,
        seed=seed,
        agent_name="random",
    )


def _evaluate_heuristic(games: int, seed: int | None) -> EvaluationReport:
    """Avalia o agente heuristico deterministico."""
    return evaluate_agent(
        agent_factory=lambda episode_seed: HeuristicAgent(),
        games=games,
        seed=seed,
        agent_name="heuristic",
    )


def _evaluate_dqn(
    model_path: Path,
    games: int,
    seed: int | None,
    device: str,
) -> EvaluationReport:
    """Carrega o checkpoint DQN e avalia sem exploracao aleatoria."""
    try:
        from game2048.learning.dqn_agent import load_dqn_agent
    except ModuleNotFoundError as error:
        if error.name == "torch":
            print(
                "PyTorch nao esta instalado. No Colab, execute: "
                "pip install -e '.[learning]'"
            )
            raise SystemExit(1) from error
        raise

    agent = load_dqn_agent(
        model_path=model_path,
        device=device,
        epsilon=0.0,
    )
    return evaluate_agent(
        agent_factory=lambda episode_seed: agent,
        games=games,
        seed=seed,
        agent_name="dqn",
    )


if __name__ == "__main__":
    main()
