import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from game2048 import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, Game2048
from game2048.terminal import clear_terminal, print_board

KEY_TO_ACTION = {
    "w": ACTION_UP,
    "d": ACTION_RIGHT,
    "s": ACTION_DOWN,
    "a": ACTION_LEFT,
}


def main() -> None:
    """Executa uma versão leve do 2048 jogável por humanos no terminal."""
    game = Game2048()
    message = "Use W/A/S/D para mover. Q sai."

    while True:
        clear_terminal()
        print_board(game.board, game.score)
        print(message)

        if game.done:
            print("Fim de jogo.")
            break

        command = input("Movimento: ").strip().lower()

        if command == "q":
            print("Jogo encerrado.")
            break

        action = KEY_TO_ACTION.get(command)
        if action is None:
            message = "Comando invalido. Use W/A/S/D ou Q."
            continue

        _, reward, done, info = game.step(action)
        if not info["changed"]:
            message = "Movimento sem efeito. Tente outra direcao."
        elif done:
            message = f"Voce marcou {reward} pontos neste movimento."
        else:
            message = f"+{reward} pontos neste movimento."


if __name__ == "__main__":
    main()
