"""
Decision Strategy Pattern for Battle AI

This module defines the abstract interface for battle decision-making strategies.
Each strategy implementation (minimax, tree-of-thought, MCP, etc.) should inherit
from DecisionStrategy and implement the choose_move method.
"""

from abc import ABC, abstractmethod
from pokechamp.poke_env_local.environment.abstract_battle import AbstractBattle
from pokechamp.poke_env_local.player.battle_order import BattleOrder


class DecisionStrategy(ABC):
    """
    Abstract base class for battle decision-making strategies.

    Each concrete strategy (minimax, tree-of-thought, MCP, etc.) implements
    this interface to provide its own decision-making algorithm.
    """

    @abstractmethod
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Choose the next move or switch for the given battle state.

        Args:
            battle: The current battle state

        Returns:
            BattleOrder representing the chosen action (move or switch)
        """
        pass
