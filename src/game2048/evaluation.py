"""Ferramentas reutilizáveis de avaliação para agentes de 2048."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from game2048.agents import Agent
from game2048.game import Game2048

AgentFactory = Callable[[int | None], Agent]

DEFAULT_TILE_TARGETS = (64, 128, 256, 512, 1024, 2048)


@dataclass(frozen=True)
class EpisodeResult:
    """Métricas coletadas de uma partida completa de 2048."""

    agent_name: str
    episode: int
    seed: int | None
    score: int
    steps: int
    max_tile: int


@dataclass(frozen=True)
class EvaluationReport:
    """Relatório agregado para um lote de episódios avaliados."""

    agent_name: str
    episodes: tuple[EpisodeResult, ...]
    tile_targets: tuple[int, ...] = DEFAULT_TILE_TARGETS

    @property
    def games(self) -> int:
        """Retorna o número de partidas avaliadas."""
        return len(self.episodes)

    @property
    def average_score(self) -> float:
        """Retorna a pontuação final média entre todos os episódios."""
        return mean(result.score for result in self.episodes)

    @property
    def best_score(self) -> int:
        """Retorna a maior pontuação final alcançada no lote."""
        return max(result.score for result in self.episodes)

    @property
    def worst_score(self) -> int:
        """Retorna a menor pontuação final alcançada no lote."""
        return min(result.score for result in self.episodes)

    @property
    def average_steps(self) -> float:
        """Retorna o número médio de movimentos efetivos por episódio."""
        return mean(result.steps for result in self.episodes)

    @property
    def best_max_tile(self) -> int:
        """Retorna o maior bloco alcançado em qualquer episódio."""
        return max(result.max_tile for result in self.episodes)

    def tile_reach_rates(self) -> dict[int, float]:
        """Retorna a frequência com que cada bloco-alvo foi alcançado."""
        return {
            target: _fraction(
                result.max_tile >= target for result in self.episodes
            )
            for target in self.tile_targets
        }

    def summary_dict(self) -> dict[str, Any]:
        """Retorna métricas agregadas como dados simples para impressão ou JSON."""
        return {
            "agent_name": self.agent_name,
            "games": self.games,
            "average_score": self.average_score,
            "best_score": self.best_score,
            "worst_score": self.worst_score,
            "average_steps": self.average_steps,
            "best_max_tile": self.best_max_tile,
        }

    def to_json_data(self) -> dict[str, Any]:
        """Retorna um relatório estruturado que pode ser serializado em JSON."""
        return {
            "summary": self.summary_dict(),
            "tile_reach_rates": {
                str(target): rate
                for target, rate in self.tile_reach_rates().items()
            },
            "episodes": [asdict(result) for result in self.episodes],
        }


def run_episode(
    agent: Agent,
    game_seed: int | None,
    episode: int = 1,
    agent_name: str | None = None,
) -> EpisodeResult:
    """Executa uma partida completa para um agente já criado."""
    game = Game2048(seed=game_seed)
    resolved_agent_name = agent_name or agent.name

    while not game.done:
        available_actions = game.available_actions()
        if not available_actions:
            break

        action = agent.choose_action(game.board, available_actions)
        game.step(action)

    return EpisodeResult(
        agent_name=resolved_agent_name,
        episode=episode,
        seed=game_seed,
        score=game.score,
        steps=game.steps,
        max_tile=max(max(row) for row in game.board),
    )


def evaluate_agent(
    agent_factory: AgentFactory,
    games: int,
    seed: int | None = None,
    agent_name: str | None = None,
    tile_targets: Sequence[int] = DEFAULT_TILE_TARGETS,
) -> EvaluationReport:
    """Avalia uma fábrica de agentes em várias partidas independentes."""
    if games <= 0:
        raise ValueError("O número de partidas deve ser maior que zero.")

    episodes: list[EpisodeResult] = []
    for index in range(games):
        # Deslocar a seed por episódio mantém os lotes reproduzíveis e ainda dá
        # a cada partida uma sequência de blocos e aleatoriedade de agente diferente.
        episode_seed = None if seed is None else seed + index
        agent = agent_factory(episode_seed)
        result = run_episode(
            agent=agent,
            game_seed=episode_seed,
            episode=index + 1,
            agent_name=agent_name,
        )
        episodes.append(result)

    resolved_agent_name = agent_name or episodes[0].agent_name
    return EvaluationReport(
        agent_name=resolved_agent_name,
        episodes=tuple(episodes),
        tile_targets=tuple(tile_targets),
    )


def write_report(report: EvaluationReport, output_path: str | Path) -> None:
    """Escreve um relatório em JSON ou CSV conforme a extensão do arquivo."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".json":
        _write_json_report(report, path)
        return

    if path.suffix.lower() == ".csv":
        _write_csv_report(report, path)
        return

    raise ValueError("O caminho de saída deve terminar com .json ou .csv.")


def _write_json_report(report: EvaluationReport, path: Path) -> None:
    """Persiste o relatório completo em formato JSON estruturado."""
    path.write_text(
        json.dumps(report.to_json_data(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_csv_report(report: EvaluationReport, path: Path) -> None:
    """Persiste métricas por episódio em um CSV amigável para planilhas."""
    reach_fields = [f"reached_{target}" for target in report.tile_targets]
    fieldnames = [
        "agent_name",
        "episode",
        "seed",
        "score",
        "steps",
        "max_tile",
        *reach_fields,
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for result in report.episodes:
            row = asdict(result)
            for target in report.tile_targets:
                row[f"reached_{target}"] = int(result.max_tile >= target)
            writer.writerow(row)


def _fraction(values: Iterable[bool]) -> float:
    """Retorna a proporção de valores verdadeiros em uma sequência não vazia."""
    materialized_values = tuple(values)
    if not materialized_values:
        return 0.0
    return sum(1 for value in materialized_values if value) / len(materialized_values)
