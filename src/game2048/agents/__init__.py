"""Agentes capazes de escolher ações para o ambiente 2048."""

from game2048.agents.base import Agent
from game2048.agents.heuristic_agent import HeuristicAgent, HeuristicWeights
from game2048.agents.random_agent import RandomAgent

__all__ = ["Agent", "HeuristicAgent", "HeuristicWeights", "RandomAgent"]
