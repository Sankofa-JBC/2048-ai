from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


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
        default=128,
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
        default=800,
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


if __name__ == "__main__":
    main()
