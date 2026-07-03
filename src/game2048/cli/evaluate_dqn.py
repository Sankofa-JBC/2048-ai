"""CLI instalavel para avaliar um checkpoint DQN."""

from __future__ import annotations

import argparse
from pathlib import Path

from game2048.evaluation import evaluate_agent, write_report


def main() -> None:
    """Avalia um checkpoint DQN salvo usando o avaliador comum do projeto."""
    parser = argparse.ArgumentParser(description="Avalia um agente DQN treinado.")
    parser.add_argument("model", help="caminho do checkpoint .pt")
    parser.add_argument("--games", type=int, default=100, help="numero de partidas")
    parser.add_argument("--seed", type=int, default=42, help="seed base")
    parser.add_argument("--device", default="cpu", help="device PyTorch")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="caminho opcional .json ou .csv para metricas",
    )
    args = parser.parse_args()

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
        model_path=args.model,
        device=args.device,
        epsilon=0.0,
    )
    report = evaluate_agent(
        agent_factory=lambda seed: agent,
        games=args.games,
        seed=args.seed,
        agent_name="dqn",
    )

    summary = report.summary_dict()
    print("Resumo da avaliacao DQN")
    print(f"Jogos: {summary['games']}")
    print(f"Score medio: {summary['average_score']:.2f}")
    print(f"Melhor score: {summary['best_score']}")
    print(f"Pior score: {summary['worst_score']}")
    print(f"Maior bloco: {summary['best_max_tile']}")
    print("Taxa de blocos alcancados")
    for target, rate in report.tile_reach_rates().items():
        print(f"{target}: {rate:.1%}")

    if args.output is not None:
        write_report(report, args.output)
        print(f"Metricas exportadas para: {args.output}")


if __name__ == "__main__":
    main()
