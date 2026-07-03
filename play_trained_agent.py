from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from game2048 import ACTION_NAMES, Game2048
from game2048.terminal import clear_terminal, print_board


def main() -> None:
    """Mostra um checkpoint DQN jogando no terminal."""
    parser = argparse.ArgumentParser(description="Assiste um agente DQN jogar 2048.")
    parser.add_argument("model", help="caminho do checkpoint .pt")
    parser.add_argument("--games", type=int, default=1, help="numero de partidas")
    parser.add_argument("--seed", type=int, default=42, help="seed base")
    parser.add_argument("--device", default="cpu", help="device PyTorch")
    parser.add_argument("--delay", type=float, default=0.08, help="pausa entre jogadas")
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="nao limpa o terminal entre jogadas",
    )
    args = parser.parse_args()

    try:
        from game2048.learning.dqn_agent import load_dqn_agent
    except ModuleNotFoundError as error:
        if error.name == "torch":
            print(
                "PyTorch nao esta instalado. Instale com: "
                "python -m pip install -e \".[learning]\""
            )
            raise SystemExit(1) from error
        raise

    agent = load_dqn_agent(args.model, device=args.device, epsilon=0.0)

    for index in range(args.games):
        score, max_tile, steps = play_one_game(
            agent=agent,
            seed=args.seed + index,
            delay=args.delay,
            clear=not args.no_clear,
        )
        print(
            f"Jogo {index + 1}: score={score}, "
            f"maior_bloco={max_tile}, passos={steps}"
        )


def play_one_game(agent: object, seed: int, delay: float, clear: bool) -> tuple[int, int, int]:
    """Executa uma partida visual no terminal."""
    game = Game2048(seed=seed)

    while not game.done:
        available_actions = game.available_actions()
        if not available_actions:
            break

        action = agent.choose_action(game.board, available_actions)
        _, reward, _, _ = game.step(action)

        if clear:
            clear_terminal()
        print_board(game.board, game.score)
        print(f"Acao: {ACTION_NAMES[action]} | recompensa: {reward}")
        time.sleep(delay)

    return game.score, max(max(row) for row in game.board), game.steps


if __name__ == "__main__":
    main()
