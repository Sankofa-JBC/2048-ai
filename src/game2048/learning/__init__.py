"""Ferramentas de aprendizagem para agentes de 2048.

Os módulos pesados deste pacote usam PyTorch, mas não são importados aqui para
manter o jogo e os testes leves funcionando mesmo em máquinas sem `torch`.
"""

from game2048.learning.replay_memory import ReplayMemory, Transition
from game2048.learning.state import board_to_features, available_action_mask

__all__ = [
    "ReplayMemory",
    "Transition",
    "available_action_mask",
    "board_to_features",
]
