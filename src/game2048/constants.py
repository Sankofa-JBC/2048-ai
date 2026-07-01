"""Constants shared by the 2048 engine and terminal interface."""

BOARD_SIZE = 4

ACTION_UP = 0
ACTION_RIGHT = 1
ACTION_DOWN = 2
ACTION_LEFT = 3

ACTIONS = (ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT)

ACTION_NAMES = {
    ACTION_UP: "up",
    ACTION_RIGHT: "right",
    ACTION_DOWN: "down",
    ACTION_LEFT: "left",
}

NEW_TILE_TWO_PROBABILITY = 0.9
