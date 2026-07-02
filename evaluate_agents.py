from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Mantém os scripts de topo executáveis antes de `pip install -e .`; o pacote
# continua importável corretamente depois da instalação editável.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from game2048 import HeuristicAgent, RandomAgent
from game2048.agents import Agent
from game2048.evaluation import EvaluationReport, evaluate_agent, write_report

AGENT_CHOICES = ("random", "heuristic")


def main() -> None:
    """Avalia um agente de referência e exporta métricas opcionalmente."""
    parser = argparse.ArgumentParser(description="Avalia agentes de 2048.")
    parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        required=True,
        help="agente que sera avaliado",
    )
    parser.add_argument("--games", type=int, default=100, help="numero de partidas")
    parser.add_argument("--seed", type=int, default=None, help="seed aleatoria base")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="caminho opcional de saida .json ou .csv",
    )
    args = parser.parse_args()

    report = evaluate_agent(
        agent_factory=lambda episode_seed: create_agent(args.agent, episode_seed),
        games=args.games,
        seed=args.seed,
        agent_name=args.agent,
    )

    print_report(report)

    if args.output is not None:
        write_report(report, args.output)
        print(f"\nMetricas exportadas para: {args.output}")


def create_agent(agent_name: str, seed: int | None) -> Agent:
    """Cria uma nova instância de agente para um episódio de avaliação."""
    if agent_name == "random":
        return RandomAgent(seed=seed)

    if agent_name == "heuristic":
        return HeuristicAgent()

    raise ValueError(f"Agente desconhecido: {agent_name}")


def print_report(report: EvaluationReport) -> None:
    """Imprime as principais métricas agregadas para comparação rápida."""
    summary = report.summary_dict()

    print("Resumo da avaliacao")
    print(f"Agente: {summary['agent_name']}")
    print(f"Jogos: {summary['games']}")
    print(f"Score medio: {summary['average_score']:.2f}")
    print(f"Melhor score: {summary['best_score']}")
    print(f"Pior score: {summary['worst_score']}")
    print(f"Passos medios: {summary['average_steps']:.2f}")
    print(f"Maior bloco: {summary['best_max_tile']}")
    print()
    print("Taxa de blocos alcancados")
    for target, rate in report.tile_reach_rates().items():
        print(f"{target}: {rate:.1%}")


if __name__ == "__main__":
    main()
