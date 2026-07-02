"""Contrato comum de agentes usado por avaliação e treinamento."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from game2048.core import Board, validate_action


@runtime_checkable
class Agent(Protocol):
    """Protocolo para qualquer objeto capaz de escolher uma ação do 2048.

    Um protocolo é usado em vez de uma classe base abstrata para que agentes
    futuros não precisem herdar de uma classe específica. Se um objeto tiver
    `name` e `choose_action(...)`, ele pode ser avaliado pelas mesmas ferramentas.
    """

    name: str

    def choose_action(self, board: Board, available_actions: Sequence[int]) -> int:
        """Escolhe uma ação entre as ações atualmente disponíveis."""


def normalize_available_actions(available_actions: Sequence[int]) -> tuple[int, ...]:
    """Valida e congela a lista de ações antes da escolha do agente."""
    actions = tuple(available_actions)
    if not actions:
        raise ValueError("Um agente precisa de pelo menos uma ação disponível.")

    # A validação fica perto da fronteira do agente para que loops de treinamento
    # inválidos falhem imediatamente em vez de produzir dados ruins silenciosamente.
    for action in actions:
        validate_action(action)

    return actions
