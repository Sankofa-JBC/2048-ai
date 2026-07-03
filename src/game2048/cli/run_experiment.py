"""Runner unico para treino, avaliacao e comparacao no Colab."""

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
from game2048.visualization import write_experiment_plots


def main() -> None:
    """Executa treino opcional e comparacao final em um unico comando."""
    parser = argparse.ArgumentParser(
        description="Treina ou retoma um DQN e gera a comparacao final."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=_default_project_dir(),
        help="raiz do repositorio onde modelos e resultados serao salvos",
    )
    parser.add_argument("--episodes", type=int, default=1000, help="episodios de treino")
    parser.add_argument("--games", type=int, default=200, help="partidas para avaliacao")
    parser.add_argument("--seed", type=int, default=42, help="seed base")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="checkpoint existente para continuar o treino",
    )
    parser.add_argument(
        "--force-fresh",
        action="store_true",
        help="ignora checkpoints existentes e reinicia o treino do zero",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="pula o treino e apenas compara um checkpoint existente",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device PyTorch, por exemplo cpu ou cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="tamanho do batch",
    )
    parser.add_argument(
        "--min-replay-size",
        type=int,
        default=500,
        help="experiencias minimas antes de treinar",
    )
    parser.add_argument(
        "--memory-capacity",
        type=int,
        default=20_000,
        help="capacidade da replay memory",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="taxa de aprendizado",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="tamanho das camadas ocultas do DQN",
    )
    parser.add_argument(
        "--epsilon-decay-episodes",
        type=int,
        default=3_000,
        help="episodios para decair epsilon",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0 / 2048.0,
        help="escala base aplicada ao reward de merge",
    )
    parser.add_argument(
        "--empty-cell-reward-weight",
        type=float,
        default=0.02,
        help="peso do shaping por aumento de casas vazias",
    )
    parser.add_argument(
        "--max-tile-reward-weight",
        type=float,
        default=0.05,
        help="peso do shaping por aumento do maior bloco",
    )
    parser.add_argument(
        "--corner-max-tile-reward-weight",
        type=float,
        default=0.02,
        help="peso do shaping por manter o maior bloco em um canto",
    )
    parser.add_argument(
        "--monotonicity-reward-weight",
        type=float,
        default=0.01,
        help="peso do shaping por tabuleiro mais monotono",
    )
    parser.add_argument(
        "--terminal-penalty",
        type=float,
        default=0.5,
        help="penalidade aplicada ao perder a partida",
    )
    parser.add_argument(
        "--allow-missing-dqn",
        action="store_true",
        help="permite gerar comparacao so com baselines quando nao houver DQN",
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    models_dir = project_dir / "models"
    results_dir = project_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = models_dir / "dqn_2048_best.pt"
    model_path = models_dir / "dqn_2048.pt"
    metrics_path = results_dir / "dqn_training_metrics.json"
    comparison_json_path = results_dir / "final_report.json"
    comparison_csv_path = results_dir / "final_comparison.csv"

    requested_resume_from = _resolve_optional_path(args.resume_from, project_dir)
    resume_from, resumed_automatically = _resolve_resume_checkpoint(
        requested_path=requested_resume_from,
        best_model_path=best_model_path,
        skip_train=args.skip_train,
        force_fresh=args.force_fresh,
    )
    if not args.skip_train:
        _run_training(
            episodes=args.episodes,
            seed=args.seed,
            batch_size=args.batch_size,
            min_replay_size=args.min_replay_size,
            memory_capacity=args.memory_capacity,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            epsilon_decay_episodes=args.epsilon_decay_episodes,
            reward_scale=args.reward_scale,
            empty_cell_reward_weight=args.empty_cell_reward_weight,
            max_tile_reward_weight=args.max_tile_reward_weight,
            corner_max_tile_reward_weight=args.corner_max_tile_reward_weight,
            monotonicity_reward_weight=args.monotonicity_reward_weight,
            terminal_penalty=args.terminal_penalty,
            output=model_path,
            best_output=best_model_path,
            metrics_output=metrics_path,
            resume_from=resume_from,
            device=args.device,
        )
    elif resume_from is not None:
        best_model_path = resume_from

    reports = [
        _evaluate_random(games=args.games, seed=args.seed),
        _evaluate_heuristic(games=args.games, seed=args.seed),
    ]

    if best_model_path.exists():
        reports.append(
            _evaluate_dqn(
                model_path=best_model_path,
                games=args.games,
                seed=args.seed,
                device="cpu" if args.device is None else args.device,
            )
        )
    elif not args.allow_missing_dqn:
        raise SystemExit(
            "Modelo DQN nao encontrado. Rode o treino primeiro ou use --allow-missing-dqn."
        )

    rows = build_comparison_rows(reports)
    write_comparison_csv(rows, comparison_csv_path)
    final_report = write_final_report(
        rows=rows,
        output_path=comparison_json_path,
        training_metrics_path=metrics_path,
        metadata={
            "project_dir": str(project_dir),
            "games": args.games,
            "seed": args.seed,
            "episodes_requested": args.episodes,
            "trained_in_this_run": not args.skip_train,
            "resume_from": None if resume_from is None else str(resume_from),
            "resumed_automatically": resumed_automatically,
        },
    )
    generated_plots = write_experiment_plots(
        rows=rows,
        output_dir=results_dir,
        training_metrics_path=metrics_path,
    )

    print()
    print("Comparacao final dos agentes")
    print(format_comparison_table(rows))
    print()
    print("Conclusao automatica")
    print(final_report["verdict"]["message"])
    training_progress = final_report.get("training_progress")
    if training_progress is not None:
        print(training_progress["message"])
    recommendation = final_report.get("recommendation")
    if recommendation is not None:
        print(recommendation["message"])
    print()
    if resume_from is not None:
        resume_label = "automaticamente" if resumed_automatically else "explicitamente"
        print(f"Checkpoint retomado {resume_label}: {resume_from}")
    elif args.force_fresh:
        print("Treino iniciado do zero por --force-fresh.")
    print()
    print(f"Projeto: {project_dir}")
    print(f"Modelo DQN: {best_model_path}")
    print(f"Relatorio JSON: {comparison_json_path}")
    print(f"Tabela CSV: {comparison_csv_path}")
    if generated_plots:
        print("Graficos:")
        for plot_path in generated_plots:
            print(f"  {plot_path}")


def _default_project_dir() -> Path:
    """Resolve a raiz do repositorio a partir do proprio modulo instalado."""
    return Path(__file__).resolve().parents[3]


def _resolve_optional_path(path: Path | None, project_dir: Path) -> Path | None:
    """Resolve caminhos relativos em relacao a raiz do projeto."""
    if path is None:
        return None
    if path.is_absolute():
        return path
    return project_dir / path


def _resolve_resume_checkpoint(
    requested_path: Path | None,
    best_model_path: Path,
    skip_train: bool,
    force_fresh: bool,
) -> tuple[Path | None, bool]:
    """Escolhe qual checkpoint deve ser usado para retomada do experimento."""
    if requested_path is not None:
        return requested_path, False

    if force_fresh:
        return None, False

    if skip_train:
        return None, False

    if best_model_path.exists():
        return best_model_path, True

    return None, False


def _run_training(
    episodes: int,
    seed: int,
    batch_size: int,
    min_replay_size: int,
    memory_capacity: int,
    learning_rate: float,
    hidden_size: int,
    epsilon_decay_episodes: int,
    reward_scale: float,
    empty_cell_reward_weight: float,
    max_tile_reward_weight: float,
    corner_max_tile_reward_weight: float,
    monotonicity_reward_weight: float,
    terminal_penalty: float,
    output: Path,
    best_output: Path,
    metrics_output: Path,
    resume_from: Path | None,
    device: str | None,
) -> None:
    """Executa treino DQN com caminhos resolvidos para o projeto."""
    try:
        from game2048.learning.training import DQNTrainingConfig, train_dqn
    except ModuleNotFoundError as error:
        if error.name == "torch":
            print(
                "PyTorch nao esta instalado. No Colab, execute: "
                "pip install -e '.[learning]'"
            )
            raise SystemExit(1) from error
        raise

    result = train_dqn(
        DQNTrainingConfig(
            episodes=episodes,
            seed=seed,
            batch_size=batch_size,
            min_replay_size=min_replay_size,
            memory_capacity=memory_capacity,
            learning_rate=learning_rate,
            hidden_size=hidden_size,
            epsilon_decay_episodes=epsilon_decay_episodes,
            reward_scale=reward_scale,
            empty_cell_reward_weight=empty_cell_reward_weight,
            max_tile_reward_weight=max_tile_reward_weight,
            corner_max_tile_reward_weight=corner_max_tile_reward_weight,
            monotonicity_reward_weight=monotonicity_reward_weight,
            terminal_penalty=terminal_penalty,
            save_path=str(output),
            best_save_path=str(best_output),
            metrics_path=str(metrics_output),
            resume_from=None if resume_from is None else str(resume_from),
            device=device,
        )
    )
    final_stats = result.episodes[-1]
    print()
    print("Treino concluido")
    print(f"Episodios registrados: {len(result.episodes)}")
    print(f"Score final: {final_stats.score}")
    print(f"Maior bloco final: {final_stats.max_tile}")
    print(f"Melhor media movel: {result.best_rolling_average_score:.2f}")
    print(f"Melhor checkpoint: {best_output}")


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

    agent = load_dqn_agent(model_path=model_path, device=device, epsilon=0.0)
    return evaluate_agent(
        agent_factory=lambda episode_seed: agent,
        games=games,
        seed=seed,
        agent_name="dqn",
    )


if __name__ == "__main__":
    main()
