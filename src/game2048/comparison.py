"""Relatorios consolidados para comparar agentes de 2048."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from game2048.evaluation import EvaluationReport


@dataclass(frozen=True)
class AgentComparisonRow:
    """Linha agregada com as principais metricas de um agente."""

    agent_name: str
    games: int
    average_score: float
    best_score: int
    worst_score: int
    average_steps: float
    best_max_tile: int

    @classmethod
    def from_report(cls, report: EvaluationReport) -> "AgentComparisonRow":
        """Cria uma linha de comparacao a partir do avaliador comum."""
        summary = report.summary_dict()
        return cls(
            agent_name=summary["agent_name"],
            games=summary["games"],
            average_score=summary["average_score"],
            best_score=summary["best_score"],
            worst_score=summary["worst_score"],
            average_steps=summary["average_steps"],
            best_max_tile=summary["best_max_tile"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Retorna dados simples para serializar em JSON ou CSV."""
        return asdict(self)


def build_comparison_rows(
    reports: list[EvaluationReport] | tuple[EvaluationReport, ...],
) -> tuple[AgentComparisonRow, ...]:
    """Transforma varios relatorios de avaliacao em linhas comparaveis."""
    return tuple(AgentComparisonRow.from_report(report) for report in reports)


def build_final_report_data(
    rows: list[AgentComparisonRow] | tuple[AgentComparisonRow, ...],
    training_metrics_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta o relatorio final usado pelo Colab e por analises locais."""
    materialized_rows = tuple(rows)
    training_summary = load_training_summary(training_metrics_path)
    verdict = compare_dqn_against_baselines(materialized_rows)
    return {
        "metadata": metadata or {},
        "training_summary": training_summary,
        "comparison": [row.to_dict() for row in materialized_rows],
        "verdict": verdict,
        "training_progress": summarize_training_progress(training_summary),
        "recommendation": build_recommendation(
            verdict=verdict,
            training_summary=training_summary,
        ),
    }


def write_final_report(
    rows: list[AgentComparisonRow] | tuple[AgentComparisonRow, ...],
    output_path: str | Path,
    training_metrics_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Salva o relatorio final em JSON e retorna os dados gravados."""
    report_data = build_final_report_data(
        rows=rows,
        training_metrics_path=training_metrics_path,
        metadata=metadata,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report_data


def write_comparison_csv(
    rows: list[AgentComparisonRow] | tuple[AgentComparisonRow, ...],
    output_path: str | Path,
) -> None:
    """Salva uma tabela CSV agregada, com uma linha por agente."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "agent_name",
        "games",
        "average_score",
        "best_score",
        "worst_score",
        "average_steps",
        "best_max_tile",
    ]

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def format_comparison_table(
    rows: list[AgentComparisonRow] | tuple[AgentComparisonRow, ...],
) -> str:
    """Formata a comparacao como tabela de texto para terminal ou Colab."""
    headers = (
        "Agente",
        "Jogos",
        "Score medio",
        "Melhor score",
        "Pior score",
        "Passos medios",
        "Maior bloco",
    )
    records = [
        (
            row.agent_name,
            str(row.games),
            f"{row.average_score:.2f}",
            str(row.best_score),
            str(row.worst_score),
            f"{row.average_steps:.2f}",
            str(row.best_max_tile),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(record[index]) for record in records))
        if records
        else len(headers[index])
        for index in range(len(headers))
    ]

    def format_record(record: tuple[str, ...]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(record)
        )

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join([format_record(headers), separator, *map(format_record, records)])


def compare_dqn_against_baselines(
    rows: list[AgentComparisonRow] | tuple[AgentComparisonRow, ...],
) -> dict[str, Any]:
    """Classifica o resultado do DQN em relacao aos baselines do projeto."""
    rows_by_name = {row.agent_name.lower(): row for row in rows}
    dqn = rows_by_name.get("dqn")
    random_agent = rows_by_name.get("random")
    heuristic = rows_by_name.get("heuristic")

    if dqn is None:
        return {
            "status": "sem_dqn",
            "satisfactory": False,
            "message": "O agente DQN nao foi avaliado neste relatorio.",
        }

    if heuristic is not None and dqn.average_score >= heuristic.average_score:
        return {
            "status": "satisfatorio",
            "satisfactory": True,
            "message": (
                "Satisfatorio: o DQN igualou ou superou o agente heuristico "
                "na pontuacao media."
            ),
            **_baseline_scores(dqn, random_agent, heuristic),
        }

    if random_agent is not None and dqn.average_score > random_agent.average_score:
        return {
            "status": "aprendizado_inicial",
            "satisfactory": False,
            "message": (
                "Aprendizado inicial: o DQN superou o agente aleatorio, "
                "mas ainda nao superou o heuristico."
            ),
            **_baseline_scores(dqn, random_agent, heuristic),
        }

    return {
        "status": "nao_satisfatorio",
        "satisfactory": False,
        "message": (
            "Nao satisfatorio: o DQN ainda nao superou o agente aleatorio "
            "na pontuacao media."
        ),
        **_baseline_scores(dqn, random_agent, heuristic),
    }


def load_training_summary(metrics_path: str | Path | None) -> dict[str, Any] | None:
    """Extrai um resumo compacto do historico de treino DQN."""
    if metrics_path is None:
        return None

    path = Path(metrics_path)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    episodes = data.get("episodes", [])
    if not episodes:
        return {
            "episodes": 0,
            "best_rolling_average_score": data.get("best_rolling_average_score"),
            "best_save_path": data.get("best_save_path"),
            "config": data.get("config"),
        }

    first_episode = episodes[0]
    last_episode = episodes[-1]
    return {
        "episodes": len(episodes),
        "first_score": first_episode.get("score"),
        "last_score": last_episode.get("score"),
        "first_rolling_average_score": first_episode.get("rolling_average_score"),
        "last_rolling_average_score": last_episode.get("rolling_average_score"),
        "best_rolling_average_score": data.get("best_rolling_average_score"),
        "best_save_path": data.get("best_save_path"),
        "config": data.get("config"),
    }


def summarize_training_progress(
    training_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Resume se o treino parece estar melhorando, parado ou indisponivel."""
    if training_summary is None:
        return None

    first_average = training_summary.get("first_rolling_average_score")
    last_average = training_summary.get("last_rolling_average_score")
    if first_average is None or last_average is None:
        return {
            "status": "insufficient_data",
            "message": "Nao ha dados suficientes para medir a evolucao do treino.",
        }

    delta = float(last_average) - float(first_average)
    if delta > 50.0:
        return {
            "status": "improving",
            "delta_rolling_average_score": delta,
            "message": "O treino mostra melhora da media movel de score.",
        }

    if delta < -50.0:
        return {
            "status": "regressing",
            "delta_rolling_average_score": delta,
            "message": "O treino piorou em relacao ao inicio da media movel.",
        }

    return {
        "status": "flat",
        "delta_rolling_average_score": delta,
        "message": "O treino esta praticamente estagnado na media movel.",
    }


def build_recommendation(
    verdict: dict[str, Any],
    training_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Gera uma recomendacao pratica a partir do resultado final do experimento."""
    progress = summarize_training_progress(training_summary)
    verdict_status = verdict["status"]

    if verdict_status == "sem_dqn":
        return {
            "action": "train_first",
            "message": "Treine o DQN antes de comparar os agentes.",
        }

    if verdict_status == "satisfatorio":
        return {
            "action": "validate_more",
            "message": (
                "Resultado bom. Valide com mais jogos, por exemplo 500 ou 1000, "
                "para confirmar estabilidade."
            ),
        }

    if verdict_status == "aprendizado_inicial":
        return {
            "action": "continue_training",
            "message": (
                "O DQN ja superou o aleatorio. Continue do checkpoint atual com "
                "mais 3000 a 5000 episodios."
            ),
        }

    if progress is not None and progress["status"] == "improving":
        return {
            "action": "continue_training",
            "message": (
                "Ainda nao superou o aleatorio, mas ha sinal de melhora. Continue "
                "do checkpoint atual com mais 3000 a 5000 episodios."
            ),
        }

    return {
        "action": "review_training_setup",
        "message": (
            "O DQN ainda nao superou o aleatorio e o treino parece fraco ou "
            "estagnado. Rode mais episodios ou revise os hiperparametros."
        ),
    }


def _baseline_scores(
    dqn: AgentComparisonRow,
    random_agent: AgentComparisonRow | None,
    heuristic: AgentComparisonRow | None,
) -> dict[str, float | None]:
    """Anexa scores medios que explicam a classificacao do resultado."""
    return {
        "dqn_average_score": dqn.average_score,
        "random_average_score": (
            None if random_agent is None else random_agent.average_score
        ),
        "heuristic_average_score": None if heuristic is None else heuristic.average_score,
    }
