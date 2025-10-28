"""This module defines a random players baseline
"""

from pokechamp.poke_env_local.environment import AbstractBattle
from pokechamp.poke_env_local.player.battle_order import BattleOrder
from pokechamp.poke_env_local.player.player import Player


class RandomPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        return self.choose_random_move(battle)
