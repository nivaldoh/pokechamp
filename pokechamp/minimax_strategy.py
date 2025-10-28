"""
Minimax Tree Search Strategy

This module implements the minimax decision-making strategy for Pokémon battles.
It uses tree search with opponent modeling to find optimal moves.
"""

import json
import os
import time
from collections import OrderedDict
from copy import copy, deepcopy
from typing import Any, Callable, Optional, Tuple

import numpy as np
from pokechamp.poke_env_local.environment.abstract_battle import AbstractBattle
from pokechamp.poke_env_local.environment.battle import Battle
from pokechamp.poke_env_local.environment.move import Move
from pokechamp.poke_env_local.environment.move_category import MoveCategory
from pokechamp.poke_env_local.environment.pokemon import Pokemon
from pokechamp.poke_env_local.player.battle_order import BattleOrder
from pokechamp.poke_env_local.player.local_simulation import LocalSim, SimNode
from pokechamp.poke_env_local.data.gen_data import GenData

from pokechamp.decision_strategy import DecisionStrategy
from pokechamp.minimax_optimizer import fast_battle_evaluation, create_battle_state_hash
from pokechamp.prompts import get_number_turns_faint, get_status_num_turns_fnt


class MinimaxStrategy(DecisionStrategy):
    """
    Minimax tree search strategy for Pokémon battles.

    This strategy uses adversarial search with opponent modeling to evaluate
    future game states and choose optimal moves. It can use either LLM-based
    evaluation or fast heuristic evaluation for leaf nodes.
    """

    # Constants for heuristic evaluation
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

    def __init__(
        self,
        # Static data dependencies
        move_effect: dict,
        pokemon_move_dict: dict,
        ability_effect: dict,
        pokemon_ability_dict: dict,
        item_effect: dict,
        pokemon_item_dict: dict,
        gen: GenData,
        battle_format: str,

        # Configuration
        K: int,
        temperature: float,
        backend: str,
        use_llm_value_function: bool,
        max_depth_for_llm_eval: int,
        _dynamax_disable: bool,
        prompt_translate: Callable,

        # TODO: this is probably bad
        # Callable dependencies (from LLMPlayer)
        get_llm_action: Callable,
        io: Callable,
        create_order: Callable,
        choose_max_damage_move: Callable,
        build_local_sim: Callable,
        llm_value: Any,
        # Defaults must come last to satisfy Python's signature rules
        use_metamon_value_function: bool = True,
    ):
        """
        Initialize the Minimax strategy with all dependencies.

        Args:
            move_effect: Dictionary of move effects
            pokemon_move_dict: Dictionary mapping Pokémon to their moves
            ability_effect: Dictionary of ability effects
            pokemon_ability_dict: Dictionary mapping Pokémon to abilities
            item_effect: Dictionary of item effects
            pokemon_item_dict: Dictionary mapping Pokémon to items
            gen: Generation data
            battle_format: Battle format string (e.g., "gen9ou")
            K: Maximum search depth
            temperature: LLM temperature
            backend: LLM backend identifier
            use_llm_value_function: Whether to use LLM for leaf evaluation
            max_depth_for_llm_eval: Maximum depth to use LLM evaluation
            _dynamax_disable: Whether dynamax is disabled
            prompt_translate: Function to translate battle state to prompts
            get_llm_action: Callable to get LLM actions
            io: Callable for input/output strategy
            create_order: Callable to create battle orders
            choose_max_damage_move: Callable fallback for choosing moves
            build_local_sim: Callable to build LocalSim instances
            llm_value: LLM instance for value evaluation
        """
        # Store static data
        self.move_effect = move_effect
        self.pokemon_move_dict = pokemon_move_dict
        self.ability_effect = ability_effect
        self.pokemon_ability_dict = pokemon_ability_dict
        self.item_effect = item_effect
        self.pokemon_item_dict = pokemon_item_dict
        self.gen = gen
        self.format = battle_format

        # Store configuration
        self.K = K
        self.temperature = temperature
        self.backend = backend
        self.use_llm_value_function = use_llm_value_function
        self.use_metamon_value_function = use_metamon_value_function
        self.max_depth_for_llm_eval = max_depth_for_llm_eval
        self._dynamax_disable = _dynamax_disable
        self.prompt_translate = prompt_translate

        # Store callable dependencies
        self.get_llm_action = get_llm_action
        self.io = io
        self.create_order = create_order
        self.choose_max_damage_move = choose_max_damage_move
        self.build_local_sim = build_local_sim
        self.llm_value = llm_value

        # Initialize LLM value cache
        self._llm_value_cache_limit = 512
        self._llm_value_cache: "OrderedDict[Any, int]" = OrderedDict()

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Main entry point for choosing a move using minimax tree search.

        Args:
            battle: Current battle state

        Returns:
            BattleOrder for the chosen action
        """
        retries = 10
        return self.tree_search(retries, battle)

    def _get_cached_llm_value(self, battle: Battle) -> Optional[int]:
        """Get cached LLM evaluation for a battle state."""
        state_hash = create_battle_state_hash(battle)
        cached = self._llm_value_cache.get(state_hash)
        if cached is not None:
            self._llm_value_cache.move_to_end(state_hash)
        return cached

    def _store_llm_value(self, battle: Battle, score: int) -> None:
        """Store LLM evaluation in cache."""
        state_hash = create_battle_state_hash(battle)
        self._llm_value_cache[state_hash] = score
        self._llm_value_cache.move_to_end(state_hash)
        if len(self._llm_value_cache) > self._llm_value_cache_limit:
            self._llm_value_cache.popitem(last=False)

    def estimate_matchup(
        self,
        sim: LocalSim,
        battle: Battle,
        mon: Pokemon,
        mon_opp: Pokemon,
        is_opp: bool = False
    ) -> Tuple[Move, int]:
        """
        Estimate the best move and turns to KO for a matchup.

        Args:
            sim: LocalSim instance
            battle: Current battle state
            mon: Attacking Pokémon
            mon_opp: Defending Pokémon
            is_opp: Whether this is for opponent's perspective

        Returns:
            Tuple of (best_move, turns_to_ko)
        """
        hp_remaining = []
        moves = list(mon.moves.keys())
        if is_opp:
            moves = sim.get_opponent_current_moves(mon=mon)
        if battle.active_pokemon.species == mon.species and not is_opp:
            moves = [move.id for move in battle.available_moves]

        for move_id in moves:
            move = Move(move_id, gen=sim.gen.gen)
            t = np.inf
            if move.category == MoveCategory.STATUS:
                # Apply stat boosting effects to see if it will KO in fewer turns
                t = get_status_num_turns_fnt(mon, move, mon_opp, sim, boosts=mon._boosts.copy())
            else:
                t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
            hp_remaining.append(t)

        hp_best_index = np.argmin(hp_remaining)
        best_move = moves[hp_best_index]
        best_move_turns = hp_remaining[hp_best_index]
        best_move = Move(best_move, gen=sim.gen.gen)
        best_move = self.create_order(best_move)

        # Check special moves: tera/dyna
        # Dynamax for gen 8
        if sim.battle._data.gen == 8 and sim.battle.can_dynamax:
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen).dynamaxed
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
                    if t < best_move_turns:
                        best_move = self.create_order(move, dynamax=True)
                        best_move_turns = t
        # Terastallize for gen 9
        elif sim.battle._data.gen == 9 and sim.battle.can_tera:
            mon.terastallize()
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen)
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
                    if t < best_move_turns:
                        best_move = self.create_order(move, terastallize=True)
                        best_move_turns = t
            mon.unterastallize()

        return best_move, best_move_turns

    def dmg_calc_move(self, battle: AbstractBattle, return_move: bool = False):
        """
        Calculate the best move based on damage calculator.

        Args:
            battle: Current battle state
            return_move: Whether to return move order instead of BattleOrder

        Returns:
            Tuple of (action, turns_to_ko) or (None, inf) if no good move
        """
        sim = LocalSim(
            battle,
            self.move_effect,
            self.pokemon_move_dict,
            self.ability_effect,
            self.pokemon_ability_dict,
            self.item_effect,
            self.pokemon_item_dict,
            self.gen,
            self._dynamax_disable,
            format=self.format
        )
        best_action = None
        best_action_turns = np.inf

        if battle.available_moves and not battle.active_pokemon.fainted:
            # Try moves and find hp remaining for opponent
            mon = battle.active_pokemon
            mon_opp = battle.opponent_active_pokemon
            best_action, best_action_turns = self.estimate_matchup(sim, battle, mon, mon_opp)

        if return_move:
            if best_action is None:
                return None, best_action_turns
            return best_action.order, best_action_turns

        if best_action_turns > 4:
            return None, np.inf

        if best_action is not None:
            return best_action, best_action_turns

        # Fallback to random move
            from pokechamp.poke_env_local.player.player import Player
        return Player.choose_random_move(None, battle), 1

    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon) -> float:
        """
        Simple heuristic matchup evaluation.

        Args:
            mon: Our Pokémon
            opponent: Opponent's Pokémon

        Returns:
            Matchup score (higher is better for us)
        """
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max([mon.damage_multiplier(t) for t in opponent.types if t is not None])

        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def _get_fast_heuristic_evaluation(self, battle_state) -> int:
        """
        Fast heuristic evaluation for leaf nodes.

        Args:
            battle_state: Battle state to evaluate

        Returns:
            Score from 0-100 (higher is better for us)
        """
        try:
            player_hp = int(battle_state.active_pokemon.current_hp_fraction * 100) if battle_state.active_pokemon else 0
            opp_hp = int(battle_state.opponent_active_pokemon.current_hp_fraction * 100) if battle_state.opponent_active_pokemon else 0
            player_remaining = len([p for p in battle_state.team.values() if not p.fainted])
            opp_remaining = len([p for p in battle_state.opponent_team.values() if not p.fainted])

            # Use cached fast evaluation
            score = fast_battle_evaluation(
                player_hp, opp_hp,
                player_remaining, opp_remaining,
                battle_state.turn
            )
            # Optional debug: trace heuristic inputs
            try:
                _log_on = os.getenv("POKECHAMP_MINIMAX_LOG", "0").lower() in {"1", "true", "t", "yes", "y"}
                if _log_on:
                    print(
                        f"[Minimax] HeurEval: hp=(P:{player_hp} O:{opp_hp}) team=(P:{player_remaining} O:{opp_remaining}) turn={getattr(battle_state,'turn','?')} -> score={score}",
                        flush=True,
                    )
            except Exception:
                pass
            return score
        except:
            # Ultimate fallback to basic hp difference
            try:
                sim = LocalSim(
                    battle_state,
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    format=self.format
                )
                return sim.get_hp_diff()
            except:
                return 50  # Neutral fallback score

    def check_timeout(self, start_time, battle):
        """
        Check if search has timed out and return fallback move if so.

        Args:
            start_time: Time when search started
            battle: Current battle state

        Returns:
            Fallback move if timed out, None otherwise
        """
        if time.time() - start_time > 30:
            print('default due to time')
            move, _ = self.dmg_calc_move(battle)
            return move
        else:
            return None

    def tree_search(self, retries, battle, sim=None, return_opp=False) -> BattleOrder:
        """
        Main minimax tree search algorithm.

        Performs adversarial tree search to find the optimal move by modeling
        opponent responses and evaluating future game states.

        Args:
            retries: Number of retries for LLM calls
            battle: Current battle state
            sim: Optional LocalSim instance (for recursive calls)
            return_opp: Whether to return opponent's predicted action

        Returns:
            BattleOrder for best action (and optionally opponent action)
        """
        # Lightweight logging toggle
        _log_on = os.getenv("POKECHAMP_MINIMAX_LOG", "0").lower() in {"1", "true", "t", "yes", "y"}
        def _log(msg: str):
            if _log_on:
                try:
                    print(f"[Minimax] {msg}", flush=True)
                except Exception:
                    pass

        # Report Metamon readiness/requirement for clarity
        metamon_required = os.getenv("POKECHAMP_METAMON_REQUIRED", "1").lower() in {"1", "true", "t", "yes", "y"}
        metamon_ready = None
        if self.use_metamon_value_function:
            try:
                from pokechamp.metamon_bridge import METAMON_BRIDGE as _MB
                metamon_ready = _MB.is_ready()[0]
            except Exception:
                metamon_ready = False
        _log(
            f"Start turn={getattr(battle, 'turn', '?')} depth={self.K} "
            f"metamon={self.use_metamon_value_function} required={metamon_required} ready={metamon_ready}"
        )

        # Optionally register live context with Metamon
        try:
            if self.use_metamon_value_function:
                from pokechamp.metamon_bridge import METAMON_BRIDGE as _MB
                _MB.ensure_registered(battle)
        except Exception:
            pass

        # Generate local simulation
        root = SimNode(
            battle,
            self.move_effect,
            self.pokemon_move_dict,
            self.ability_effect,
            self.pokemon_ability_dict,
            self.item_effect,
            self.pokemon_item_dict,
            self.gen,
            self._dynamax_disable,
            depth=1,
            format=self.format,
            prompt_translate=self.prompt_translate,
            sim=sim
        )

        # Ensure root simulation has a cloned Metamon context from the live battle
        try:
            if self.use_metamon_value_function:
                from pokechamp.metamon_bridge import METAMON_BRIDGE as _MB
                # Clone once per search so child clones inherit a valid parent
                _MB.clone_context(battle, root.simulation.battle)
                if os.getenv("POKECHAMP_MINIMAX_LOG", "0").lower() in {"1", "true", "t", "yes", "y"}:
                    try:
                        print(
                            f"[Minimax] root clone parent_id={id(battle)} child_id={id(root.simulation.battle)}",
                            flush=True,
                        )
                    except Exception:
                        pass
        except Exception:
            pass

        q = [root]
        leaf_nodes = []
        start_time = time.time()

        # Leaf evaluation stats (kept minimal to avoid spam)
        _leaf_count = 0
        _leaf_metamon = 0
        _leaf_heur = 0
        _leaf_llm = 0
        _leaf_scores = []
        _leaf_samples = []  # up to 8 sample lines

        def _path_of(node: SimNode) -> str:
            # Build a short path string of player/opponent actions
            steps = []
            cur = node
            while cur is not None and getattr(cur, 'parent_node', None) is not None:
                try:
                    if getattr(cur, 'action', None) is not None:
                        steps.append(f"P:{str(cur.action.order)}")
                    if getattr(cur, 'action_opp', None) is not None:
                        steps.append(f"O:{str(cur.action_opp.order)}")
                except Exception:
                    break
                cur = cur.parent_node
            if not steps:
                return "<root>"
            return " | ".join(reversed(steps[-6:]))  # last few steps only

        decision_method = "minimax"
        while len(q) != 0:
            node = q.pop(0)
            # Choose node for expansion
            # Generate B actions
            player_actions = []
            system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move = node.simulation.get_player_prompt(return_actions=True)

            # End if terminal
            if node.simulation.is_terminal() or node.depth == self.K:
                try:
                    leaf_src = "heur"
                    if self.use_metamon_value_function:
                        try:
                            from pokechamp.metamon_bridge import METAMON_BRIDGE as _MB
                            score = _MB.score(node.simulation.battle)
                        except Exception:
                            score = None
                        if score is not None:
                            node.hp_diff = score
                            _leaf_metamon += 1
                            _leaf_scores.append(score)
                            leaf_src = "metamon"
                        else:
                            node.hp_diff = self._get_fast_heuristic_evaluation(node.simulation.battle)
                            _leaf_heur += 1
                            _leaf_scores.append(node.hp_diff)
                            leaf_src = "heur"
                    elif self.use_llm_value_function and node.depth <= self.max_depth_for_llm_eval:
                        cached_value = self._get_cached_llm_value(node.simulation.battle)
                        if cached_value is not None:
                            node.hp_diff = cached_value
                        else:
                            value_prompt = 'Evaluate the score from 1-100 based on how likely the player is to win. Higher is better. Start at 50 points.' +\
                                            'Add points based on the effectiveness of current available moves.' +\
                                            'Award points for each pokemon remaining on the player\'s team, weighted by their strength' +\
                                            'Add points for boosted status and opponent entry hazards and subtract points for status effects and player entry hazards. ' +\
                                            'Subtract points for excessive switching.' +\
                                            'Subtract points based on the effectiveness of the opponent\'s current moves, especially if they have a faster speed.' +\
                                            'Remove points for each pokemon remaining on the opponent\'s team, weighted by their strength.\n'
                            cot_prompt = 'Briefly justify your total score, up to 100 words. Then, conclude with the score in the JSON format: {"score": <total_points>}. '
                            state_prompt_io = state_prompt + value_prompt + cot_prompt
                            llm_output = self.get_llm_action(
                                system_prompt=system_prompt,
                                user_prompt=state_prompt_io,
                                model=self.backend,
                                temperature=self.temperature,
                                max_tokens=500,
                                json_format=True,
                                llm=self.llm_value
                            )
                            # Parse JSON from LLM output
                            from pokechamp.llm_player import LLMPlayer
                            llm_action_json = LLMPlayer._safe_json_to_dict(llm_output)
                            if not llm_action_json or 'score' not in llm_action_json:
                                raise ValueError('Score not found in LLM response')
                            node.hp_diff = int(float(llm_action_json['score']))
                            # Log concise LLM value result at root depth
                            if _log_on and node.depth == 1:
                                try:
                                    # show only a short preview of raw output if it's text-like
                                    preview = llm_output
                                    if isinstance(preview, str):
                                        preview = preview.strip().replace('\n', ' ')
                                        if len(preview) > 120:
                                            preview = preview[:120] + '…'
                                    _log(f"LLM leaf eval: score={node.hp_diff} preview={preview}")
                                except Exception:
                                    pass
                            self._store_llm_value(node.simulation.battle, node.hp_diff)
                        _leaf_llm += 1
                        _leaf_scores.append(node.hp_diff)
                        leaf_src = "llm"
                    else:
                        node.hp_diff = self._get_fast_heuristic_evaluation(node.simulation.battle)
                        _leaf_heur += 1
                        _leaf_scores.append(node.hp_diff)
                        leaf_src = "heur"
                except Exception as e:
                    try:
                        node.hp_diff = node.simulation.get_hp_diff()
                    except Exception:
                        node.hp_diff = 50
                    print(e)

                leaf_nodes.append(node)
                _leaf_count += 1
                if _log_on and len(_leaf_samples) < 8:
                    _leaf_samples.append(f"Leaf d={node.depth} score={node.hp_diff} src={leaf_src} path={_path_of(node)}")
                continue

            # Estimate opponent action
            try:
                action_opp, opp_turns = self.estimate_matchup(
                    node.simulation,
                    node.simulation.battle,
                    node.simulation.battle.opponent_active_pokemon,
                    node.simulation.battle.active_pokemon,
                    is_opp=True
                )
            except:
                action_opp = None
                opp_turns = np.inf

            ##############################
            # Generate player's actions  #
            ##############################
            if not node.simulation.battle.active_pokemon.fainted and len(node.simulation.battle.available_moves) > 0:
                # Get dmg calc move
                dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(node.simulation.battle)
                if dmg_calc_out is not None:
                    if dmg_calc_turns <= opp_turns:
                        try:
                            # Ask LLM to use heuristic tool or minimax search
                            tool_prompt = '''Based on the current battle state, evaluate whether to use the damage calculator tool or the minimax tree search method. Consider the following factors:

1. Damage calculator advantages:
- Quick and efficient for finding optimal damaging moves
- Useful when a clear type advantage or high-power move is available
- Effective when the opponent's is not switching and current pokemon is likely to KO opponent

2. Minimax tree search advantages:
- Can model opponent behavior and predict future moves
- Useful in complex situations with multiple viable options
- Effective when long-term strategy is crucial

3. Current battle state:
- Remaining Pokémon on each side
- Health of active Pokémon
- Type matchups
- Available moves and their effects
- Presence of status conditions or field effects

4. Uncertainty level:
- How predictable is the opponent's next move?
- Are there multiple equally viable options for your next move?

Evaluate these factors and decide which method would be more beneficial in the current situation. Output your choice in the following JSON format:

{"choice":"damage calculator"} or {"choice":"minimax"}'''

                            state_prompt_io = state_prompt + tool_prompt
                            llm_output = self.get_llm_action(
                                system_prompt=system_prompt,
                                user_prompt=state_prompt_io,
                                model=self.backend,
                                temperature=0.6,
                                max_tokens=100,
                                json_format=True,
                                actions=["damage calculator", "minimax"],
                            )
                            # Parse JSON
                            from pokechamp.llm_player import LLMPlayer
                            llm_action_json = LLMPlayer._safe_json_to_dict(llm_output)
                            choice_value = None
                            if llm_action_json is not None:
                                choice_value = llm_action_json.get('choice')

                            if isinstance(choice_value, str) and choice_value.lower() != 'minimax':
                                decision_method = "damage_calc"
                                if _log_on and node.depth == 1:
                                    try:
                                        _log(f"Decision method=damage_calc (LLM choice='{choice_value}') action={str(dmg_calc_out)}")
                                    except Exception:
                                        pass
                                if return_opp:
                                    # Use tool to save time and llm when move makes bigger difference
                                    return dmg_calc_out, action_opp
                                return dmg_calc_out
                        except:
                            print('defaulting to minimax')
                    player_actions.append(dmg_calc_out)

            # Get llm switch
            if len(node.simulation.battle.available_switches) != 0:
                state_action_prompt_switch = state_action_prompt + action_prompt_switch + '\nYou can only choose to switch this turn.\n'
                constraint_prompt_io = 'Choose the best action and your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}.\n'
                for i in range(2):
                    action_llm_switch = self.io(
                        retries,
                        system_prompt,
                        state_prompt,
                        constraint_prompt_cot,
                        constraint_prompt_io,
                        state_action_prompt_switch,
                        node.simulation.battle,
                        node.simulation
                    )
                    if _log_on and node.depth == 1 and i == 0:
                        try:
                            _log(f"LLM switch suggestion: {getattr(action_llm_switch, 'message', str(action_llm_switch))}")
                        except Exception:
                            pass
                    if len(player_actions) == 0:
                        player_actions.append(action_llm_switch)
                    elif action_llm_switch.message != player_actions[-1].message:
                        player_actions.append(action_llm_switch)

            if not node.simulation.battle.active_pokemon.fainted and len(node.simulation.battle.available_moves) > 0:
                # Get llm move
                state_action_prompt_move = state_action_prompt + action_prompt_move + '\nYou can only choose to move this turn.\n'
                constraint_prompt_io = 'Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"}.\n'
                action_llm_move = self.io(
                    retries,
                    system_prompt,
                    state_prompt,
                    constraint_prompt_cot,
                    constraint_prompt_io,
                    state_action_prompt_move,
                    node.simulation.battle,
                    node.simulation
                )
                if _log_on and node.depth == 1:
                    try:
                        _log(f"LLM move suggestion: {getattr(action_llm_move, 'message', str(action_llm_move))}")
                    except Exception:
                        pass
                if len(player_actions) == 0:
                    player_actions.append(action_llm_move)
                elif action_llm_move.message != player_actions[0].message:
                    player_actions.append(action_llm_move)

            ##############################
            # Generate opponent's action #
            ##############################
            opponent_actions = []
            tool_is_optimal = False

            if action_opp is not None:
                tool_is_optimal = True
                opponent_actions.append(self.create_order(action_opp))

            # Heuristic matchup switch action
            best_score = np.inf
            best_action = None
            for mon in node.simulation.battle.opponent_team.values():
                if mon.species == node.simulation.battle.opponent_active_pokemon.species:
                    continue
                score = self._estimate_matchup(mon, node.simulation.battle.active_pokemon)
                if score < best_score:
                    best_score = score
                    best_action = mon
            if best_action is not None:
                opponent_actions.append(self.create_order(best_action))

            # Create opponent prompt from battle sim
            system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o = node.simulation.get_opponent_prompt(system_prompt)
            action_o = self.io(
                2,
                system_prompt_o,
                state_prompt_o,
                constraint_prompt_cot_o,
                constraint_prompt_io_o,
                state_action_prompt_o,
                node.simulation.battle,
                node.simulation,
                dont_verify=True
            )
            if _log_on and node.depth == 1:
                try:
                    _log(f"LLM opponent prediction: {getattr(action_o, 'message', str(action_o))}")
                except Exception:
                    pass
            is_repeat_action_o = np.array([action_o.message == opponent_action.message for opponent_action in opponent_actions]).any()
            if not is_repeat_action_o:
                opponent_actions.append(action_o)

            # Simulate outcome
            if node.depth < self.K:
                # Compact expansion log for shallow nodes
                if _log_on and node.depth <= 2:
                    try:
                        _log(f"Expand d={node.depth} combos={len(player_actions)}x{len(opponent_actions)} -> {len(player_actions)*len(opponent_actions)} children")
                    except Exception:
                        pass
                for action_p in player_actions:
                    for action_o in opponent_actions:
                        # Build a fresh child node with a distinct LocalSim and deepcopied Battle
                        child_battle = deepcopy(node.simulation.battle)
                        node_new = SimNode(
                            child_battle,
                            self.move_effect,
                            self.pokemon_move_dict,
                            self.ability_effect,
                            self.pokemon_ability_dict,
                            self.item_effect,
                            self.pokemon_item_dict,
                            self.gen,
                            self._dynamax_disable,
                            depth=node.depth + 1,
                            format=self.format,
                            prompt_translate=self.prompt_translate,
                            sim=None,
                        )
                        # Clone Metamon context for the new battle branch
                        try:
                            if self.use_metamon_value_function:
                                from pokechamp.metamon_bridge import METAMON_BRIDGE as _MB
                                if _log_on and node.depth == 1:
                                    try:
                                        _log(
                                            f"child-create parent_id={id(node.simulation.battle)} child_id={id(node_new.simulation.battle)} P={str(getattr(action_p,'order',action_p))} O={str(getattr(action_o,'order',action_o))}"
                                        )
                                    except Exception:
                                        pass
                                _MB.clone_context(node.simulation.battle, node_new.simulation.battle)
                        except Exception:
                            pass
                        node_new.action = action_p
                        node_new.action_opp = action_o
                        node_new.parent_node = node
                        node_new.parent_action = node.action
                        node.children.append(node_new)
                        node_new.simulation.step(action_p, action_o)
                        if _log_on and node.depth == 1:
                            try:
                                _log(f"child-step child_id={id(node_new.simulation.battle)}")
                            except Exception:
                                pass
                        q.append(node_new)

        # Choose best action according to max or min rule
        root_ref = root
        root_summary = {}
        root_actions = {}
        root_opp = {}
        def get_tree_action(node: SimNode):
            # Base case: leaf node
            if len(node.children) == 0:
                return node.action, node.hp_diff, node.action_opp
            score_dict = {}
            action_dict = {}
            opp_dict = {}
            for child in node.children:
                action = str(child.action.order)
                _, score, _ = get_tree_action(child)
                if action in score_dict.keys():
                    # Minimax: worst-case over opponent replies
                    score_dict[action] = min(score, score_dict[action])
                else:
                    score_dict[action] = score
                    action_dict[action] = child.action
                    opp_dict[action] = child.action_opp
            # Choose the action that maximizes the minimax value
            best_action_str = max(score_dict.keys(), key=lambda k: score_dict[k])
            # Capture root-level summary for logging
            if _log_on and node is root_ref:
                nonlocal root_summary, root_actions, root_opp
                root_summary = score_dict.copy()
                root_actions = action_dict.copy()
                root_opp = opp_dict.copy()
            return action_dict[best_action_str], score_dict[best_action_str], opp_dict[best_action_str]

        action, best_score, action_opp = get_tree_action(root)
        end_time = time.time()

        # Release ephemeral Metamon contexts created during this search
        try:
            if self.use_metamon_value_function:
                from pokechamp.metamon_bridge import METAMON_BRIDGE as _MB
                _MB.release_ephemeral()
        except Exception:
            pass

        # Emit compact logs
        if _log_on:
            try:
                if _leaf_scores:
                    _log(f"Leaves={_leaf_count} metamon={_leaf_metamon} heur={_leaf_heur} llm={_leaf_llm} min={min(_leaf_scores)} max={max(_leaf_scores)} avg={sum(_leaf_scores)/max(1,len(_leaf_scores)):.1f}")
                for line in _leaf_samples:
                    _log(line)
                if root_summary:
                    # Log every immediate root branch score (worst-case over opp replies)
                    for a, s in sorted(root_summary.items(), key=lambda kv: kv[1], reverse=True):
                        opp = root_opp.get(a)
                        opp_s = f" vs_opp={str(opp.order)}" if opp else ""
                        _log(f"Branch {a} -> minimax={s}{opp_s}")
                # Chosen action
                _log(f"Chosen action {str(action.order)} score={best_score} vs_opp={str(action_opp.order) if action_opp else 'None'} elapsed={end_time-start_time:.2f}s")
                _log(f"Decision method={decision_method}")
            except Exception:
                pass

        if return_opp:
            return action, action_opp
        return action
