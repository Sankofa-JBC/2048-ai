"""CLI instalavel para treino DQN do jogo 2048."""

from __future__ import annotations

import argparse


def main() -> None:
    """Executa um treino DQN curto ou longo conforme os argumentos."""
    parser = argparse.ArgumentParser(description="Treina um agente DQN para 2048.")
    parser.add_argument("--episodes", type=int, default=200, help="numero de episodios")
    parser.add_argument("--seed", type=int, default=42, help="seed base")
    parser.add_argument("--batch-size", type=int, default=64, help="tamanho do batch")
    parser.add_argument("--gamma", type=float, default=0.99, help="fator de desconto")
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
        "--hidden-size",
        type=int,
        default=256,
        help="tamanho das camadas ocultas",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="taxa de aprendizado",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=20,
        help="episodios entre atualizacoes da rede alvo",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="epsilon inicial",
    )
    parser.add_argument(
        "--epsilon-end",
        type=float,
        default=0.05,
        help="epsilon final",
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
        help="escala aplicada ao reward do jogo",
    )
    parser.add_argument(
        "--max-steps-per-episode",
        type=int,
        default=5_000,
        help="limite de movimentos por episodio",
    )
    parser.add_argument(
        "--output",
        default="models/dqn_2048.pt",
        help="caminho do checkpoint salvo",
    )
    parser.add_argument(
        "--best-output",
        default="models/dqn_2048_best.pt",
        help="checkpoint com melhor media movel",
    )
    parser.add_argument(
        "--metrics-output",
        default="results/dqn_training_metrics.json",
        help="JSON com historico de treino",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="models/checkpoints",
        help="pasta de checkpoints intermediarios",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="intervalo de episodios para checkpoint intermediario",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=50,
        help="janela da media movel do score",
    )
    parser.add_argument(
        "--no-double-dqn",
        action="store_true",
        help="usa DQN classico em vez de Double DQN",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device PyTorch, por exemplo cpu ou cuda",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="checkpoint usado para continuar o treino a partir de pesos anteriores",
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
    args = parser.parse_args()

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

    config = DQNTrainingConfig(
        episodes=args.episodes,
        seed=args.seed,
        gamma=args.gamma,
        batch_size=args.batch_size,
        min_replay_size=args.min_replay_size,
        memory_capacity=args.memory_capacity,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        target_update_interval=args.target_update_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_episodes=args.epsilon_decay_episodes,
        reward_scale=args.reward_scale,
        max_steps_per_episode=args.max_steps_per_episode,
        save_path=args.output,
        best_save_path=args.best_output,
        metrics_path=args.metrics_output,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        rolling_window=args.rolling_window,
        double_dqn=not args.no_double_dqn,
        device=args.device,
        resume_from=args.resume_from,
        empty_cell_reward_weight=args.empty_cell_reward_weight,
        max_tile_reward_weight=args.max_tile_reward_weight,
        corner_max_tile_reward_weight=args.corner_max_tile_reward_weight,
        monotonicity_reward_weight=args.monotonicity_reward_weight,
        terminal_penalty=args.terminal_penalty,
    )
    result = train_dqn(config)
    final_stats = result.episodes[-1]

    print()
    print("Treino concluido")
    print(f"Episodios: {len(result.episodes)}")
    print(f"Score final: {final_stats.score}")
    print(f"Maior bloco final: {final_stats.max_tile}")
    print(f"Melhor media movel: {result.best_rolling_average_score:.2f}")
    print(f"Checkpoint: {result.save_path}")
    if result.best_save_path is not None:
        print(f"Melhor checkpoint: {result.best_save_path}")
    if args.resume_from is not None:
        print(f"Treino retomado de: {args.resume_from}")


if __name__ == "__main__":
    main()
