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
        "--output",
        default="models/dqn_2048.pt",
        help="caminho do checkpoint salvo",
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
        batch_size=args.batch_size,
        min_replay_size=args.min_replay_size,
        memory_capacity=args.memory_capacity,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        save_path=args.output,
        device=args.device,
    )
    result = train_dqn(config)
    final_stats = result.episodes[-1]

    print()
    print("Treino concluido")
    print(f"Episodios: {len(result.episodes)}")
    print(f"Score final: {final_stats.score}")
    print(f"Maior bloco final: {final_stats.max_tile}")
    print(f"Checkpoint: {result.save_path}")


if __name__ == "__main__":
    main()
