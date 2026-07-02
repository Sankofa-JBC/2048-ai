from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from game2048 import ACTION_NAMES, Game2048, RandomAgent
from game2048.terminal import clear_terminal, print_board


@dataclass(frozen=True)
class GameResult:
    """Resumo de um episódio concluído pelo agente aleatório."""

    score: int
    steps: int
    max_tile: int


def main() -> None:
    """Executa uma ou mais partidas completas usando o agente aleatório de referência."""
    parser = argparse.ArgumentParser(description="Executa o agente aleatorio de 2048.")
    parser.add_argument("--games", type=int, default=1, help="numero de partidas")
    parser.add_argument("--seed", type=int, default=None, help="seed aleatoria base")
    parser.add_argument(
        "--show",
        action="store_true",
        help="mostra cada movimento no terminal",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="intervalo entre movimentos exibidos, em segundos",
    )
    args = parser.parse_args()

    results = []
    for index in range(args.games):
        # Desloca a seed por partida para que lotes repetidos sejam reproduzíveis
        # sem forçar todas as partidas do lote a serem idênticas.
        seed = None if args.seed is None else args.seed + index
        result = run_single_game(seed=seed, show=args.show, delay=args.delay)
        results.append(result)
        print(
            f"Jogo {index + 1}: score={result.score}, "
            f"maior_bloco={result.max_tile}, passos={result.steps}"
        )

    print_summary(results)


def run_single_game(seed: int | None, show: bool, delay: float) -> GameResult:
    """Joga um episódio completo e retorna suas estatísticas finais."""
    game = Game2048(seed=seed)
    agent = RandomAgent(seed=seed)

    while not game.done:
        available_actions = game.available_actions()
        if not available_actions:
            break

        action = agent.choose_action(game.board, available_actions)
        _, reward, _, _ = game.step(action)

        if show:
            clear_terminal()
            print_board(game.board, game.score)
            print(f"Acao: {ACTION_NAMES[action]} | recompensa: {reward}")
            time.sleep(delay)

    return GameResult(
        score=game.score,
        steps=game.steps,
        max_tile=max(max(row) for row in game.board),
    )


def print_summary(results: list[GameResult]) -> None:
    """Imprime estatísticas agregadas de um lote de partidas concluídas."""
    if not results:
        print("Nenhum jogo executado.")
        return

    total_score = sum(result.score for result in results)
    best_score = max(result.score for result in results)
    best_tile = max(result.max_tile for result in results)
    average_score = total_score / len(results)

    print()
    print("Resumo")
    print(f"Jogos: {len(results)}")
    print(f"Score medio: {average_score:.2f}")
    print(f"Melhor score: {best_score}")
    print(f"Maior bloco: {best_tile}")


if __name__ == "__main__":
    main()
