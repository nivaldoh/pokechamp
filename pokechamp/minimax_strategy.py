"""
Minimax Tree Search Strategy

This module implements the minimax decision-making strategy for Pokémon battles.
It uses tree search with opponent modeling to find optimal moves.
"""

import json
import os
import time
from time import perf_counter
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
from concurrent.futures import ThreadPoolExecutor


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
        get_strategy_text: Optional[Callable[[Battle], str]] = None,
        root_model: Optional[str] = None,
        child_model: Optional[str] = None,
        layered_system_prompt: Optional[str] = None,
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
        self.get_strategy_text = get_strategy_text
        self.root_model = root_model or backend
        self.child_model = child_model or backend
        # Separate system prompt for layered root orchestration
        self.layered_system_prompt = layered_system_prompt or (
            "You are a high-level competitive Pokémon player in gen 9 OU, making a decision during a battle."
        )

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

    # -------------------- Layered LLM Orchestration (root) --------------------
    class _LayerContext:
        def __init__(self, *, battle: Battle, sim: LocalSim,
                     system_prompt_layered: str,
                     state_prompt_ours: str, state_action_prompt_ours: str, io_player: str,
                     state_prompt_opp: str, state_action_prompt_opp: str, io_opp: str):
            self.battle = battle
            self.sim = sim
            self.system_prompt_layered = system_prompt_layered
            # Prompts per side
            self.state_prompt_ours = state_prompt_ours
            self.state_action_prompt_ours = state_action_prompt_ours
            self.io_player = io_player
            self.state_prompt_opp = state_prompt_opp
            self.state_action_prompt_opp = state_action_prompt_opp
            self.io_opp = io_opp
            # Accumulated outputs
            self.text_outputs: dict[str, str] = {}
            self.player_actions: list[BattleOrder] = []
            self.opponent_actions: list[BattleOrder] = []

        def register_text(self, tag: str, text: str) -> None:
            if not isinstance(text, str):
                return
            self.text_outputs[tag] = text.strip()

        def select_text(self, tags: list[str]) -> str:
            parts = []
            for t in tags:
                v = self.text_outputs.get(t, "")
                if isinstance(v, str) and v.strip():
                    parts.append(f"[{t}]\n{v}")
            return "\n\n".join(parts)

    class _LayerTask:
        def __init__(self, key: str, is_opp: bool, instruction: str, input_tags: list[str], parse_mode: str = "text"):
            self.key = key
            self.is_opp = is_opp
            self.instruction = instruction
            self.input_tags = input_tags
            self.parse_mode = parse_mode  # 'text' or 'action'

    class _BaseLayer:
        name = "Layer"
        parse_actions = False
        def build_tasks(self, ctx: "MinimaxStrategy._LayerContext") -> list["MinimaxStrategy._LayerTask"]:
            return []

    class _L1StrategyLayer(_BaseLayer):
        name = "L1 Strategy"
        parse_actions = False
        def build_tasks(self, ctx):
            return [
                MinimaxStrategy._LayerTask("L1.plan_ours", False, "[STRATEGY] Summarize our overall gameplan and win conditions.", []),
                MinimaxStrategy._LayerTask("L1.plan_opp", True, "[STRATEGY] Summarize opponent's likely gameplan and win conditions.", []),
                MinimaxStrategy._LayerTask("L1.speed", False, "[ANALYSIS] Identify who is likely to move first: us or the opponent.", []),
                MinimaxStrategy._LayerTask("L1.hazards_tera", False, "[STRATEGY] How should we handle hazards and Tera usage?", []),
            ]

    class _L2ThreatsLayer(_BaseLayer):
        name = "L2 Threats/Greed"
        parse_actions = False
        def build_tasks(self, ctx):
            return [
                MinimaxStrategy._LayerTask("L2.threats_ours", False, "[ANALYSIS] Identify whether we are under immediate threat or not and what would be our best response", ["L1.plan_ours", "L1.hazards_tera", "L1.speed"]),
                MinimaxStrategy._LayerTask("L2.longterm_ours", True, "[TACTICS] Identify whether our active Pokémon has a move/item that grants long-term advantage.", ["L1.plan_ours"]),
                MinimaxStrategy._LayerTask("L2.threats_opp", True, "[ANALYSIS] Identify whether the opponent is under immediate threat or not and what would be their best response.", ["L1.plan_opp", "L1.hazards_tera", "L1.speed"]),
                MinimaxStrategy._LayerTask("L2.longterm_opp", True, "[TACTICS] Identify whether the opponent's active Pokémon has a move/item that grants them long-term advantage.", ["L1.plan_opp"]),
            ]

    class _L3TacticsLayer(_BaseLayer):
        name = "L3 Tactics"
        parse_actions = False
        def build_tasks(self, ctx):
            return [
                MinimaxStrategy._LayerTask("L3.obvious_choice_ours", False, "[TACTICS] Identify whether we are forced to respond to an immediate threat or if we are safe to make a greedier choice (preferable, if available)", ["L2.threats_ours", "L2.longterm_ours"]),
                MinimaxStrategy._LayerTask("L3.obvious_choice_opp", False, "[TACTICS] Identify whether the opponent is forced to respond to an immediate threat or if they are safe to make a greedier choice (they would prefer it, if available)", ["L2.threats_ours", "L2.longterm_ours"]),
            ]

    class _L4ActionsLayer(_BaseLayer):
        name = "L4 Actions"
        parse_actions = True
        def build_tasks(self, ctx):
            return [
                MinimaxStrategy._LayerTask("L4.action_best_now", False, "Given prior summaries, output our best immediate action (JSON).", ["L3.obvious_choice_ours", "L3.obvious_choice_opp"], parse_mode="action"),
                MinimaxStrategy._LayerTask("L4.action_greedy", False, "Given prior summaries, output the best long-term action (JSON).", ["L1.plan_ours", "L3.obvious_choice_ours", "L3.obvious_choice_opp"], parse_mode="action"),
                MinimaxStrategy._LayerTask("L4.best_counter", True, "Given prior summaries, assume the opponent will choose their best option this turn and output our best option now that we have anticipated it (JSON).", ["L3.obvious_choice_opp"], parse_mode="action"),
            ]

    def _run_llm_tasks(self, retries: int, ctx: "MinimaxStrategy._LayerContext", tasks: list["MinimaxStrategy._LayerTask"], parse_actions: bool):
        # Execute a batch of tasks in parallel (primary model only)
        from concurrent.futures import ThreadPoolExecutor
        results = {}
        def _trim(s: str) -> str:
            try:
                return (s or "").strip()
            except Exception:
                return ""
        with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as ex:
            futs = {}
            for t in tasks:
                # Build user prompt per task
                # Select side-specific battle state prompt
                if not t.is_opp:
                    state_prompt_layer = ctx.state_prompt_ours
                else:
                    state_prompt_layer = ctx.state_prompt_opp
                print(f'state prompt: {state_prompt_layer}')
                # Selectively include prior outputs
                selected = ctx.select_text(t.input_tags)
                # Instruction
                instruction = t.instruction
                if parse_actions:
                    # Build state prompt with selected prior text; let io() attach action constraints
                    state_prompt_full = state_prompt_layer + ("\n\n" + selected if selected else "") + "\n" + instruction
                    # Use io() to produce BattleOrder objects (JSON parsing inside io)
                    futs[t.key] = ex.submit(
                        self.io,
                        retries,
                        ctx.system_prompt_layered,
                        state_prompt_full,
                        "",
                        (ctx.io_opp if t.is_opp else ctx.io_player),
                        (ctx.state_action_prompt_opp if t.is_opp else ctx.state_action_prompt_ours),
                        ctx.battle,
                        ctx.sim,
                        (t.is_opp),
                        None,
                        self.root_model,
                    )
                else:
                    # Analysis/text: plain free-form output
                    user_prompt = state_prompt_layer + ("\n\n" + selected if selected else "") + "\n" + instruction
                    futs[t.key] = ex.submit(
                        self.get_llm_action,
                        ctx.system_prompt_layered,
                        user_prompt,
                        self.root_model,
                        self.temperature,
                        False,
                    )
            for key, fut in futs.items():
                try:
                    res = fut.result()
                except Exception:
                    res = None
                results[key] = res
        return results

    def _run_llm_pipeline(self, retries: int, *, battle: Battle, sim: LocalSim,
                           state_prompt: str, state_action_prompt: str, io_player: str,
                           state_prompt_o: str, state_action_prompt_o: str, io_opp: str):
        # Create context
        ctx = MinimaxStrategy._LayerContext(
            battle=battle,
            sim=sim,
            system_prompt_layered=self.layered_system_prompt,
            state_prompt_ours=state_prompt,
            state_action_prompt_ours=state_action_prompt,
            io_player=io_player,
            state_prompt_opp=state_prompt_o,
            state_action_prompt_opp=state_action_prompt_o,
            io_opp=io_opp,
        )
        # Define layers
        layers = [
            MinimaxStrategy._L1StrategyLayer(),
            MinimaxStrategy._L2ThreatsLayer(),
            MinimaxStrategy._L3TacticsLayer(),
            MinimaxStrategy._L4ActionsLayer(),
        ]
        # Execute sequentially
        for layer in layers:
            tasks = layer.build_tasks(ctx)
            parse_actions = getattr(layer, "parse_actions", False)
            results = self._run_llm_tasks(retries, ctx, tasks, parse_actions)
            # Aggregate results
            if parse_actions:
                # Each result is a BattleOrder (or None)
                for key, res in results.items():
                    if res is None:
                        continue
                    try:
                        # LLMPlayer.io returns a BattleOrder
                        if isinstance(res, BattleOrder):
                            if key.endswith("opp_most_likely"):
                                if not any(getattr(x, 'message', None) == res.message for x in ctx.opponent_actions):
                                    ctx.opponent_actions.append(res)
                            else:
                                if not any(getattr(x, 'message', None) == res.message for x in ctx.player_actions):
                                    ctx.player_actions.append(res)
                    except Exception:
                        pass
            else:
                # Text aggregation: register per-key
                for key, out in results.items():
                    if isinstance(out, tuple):
                        # Some backends may return (text, flag); take first
                        out = out[0]
                    if out is None:
                        out = ""
                    ctx.register_text(key, str(out))
        return ctx.player_actions, ctx.opponent_actions

    def _parallel_gpt_root_calls(
        self,
        retries: int,
        *,
        system_prompt: str,
        state_prompt: str,
        constraint_prompt_cot: str,
        state_action_prompt: str,
        action_prompt_switch: str,
        action_prompt_move: str,
        battle: Battle,
        sim: LocalSim,
        system_prompt_o: str,
        state_prompt_o: str,
        constraint_prompt_cot_o: str,
        constraint_prompt_io_o: str,
        state_action_prompt_o: str,
        have_switch: bool,
        have_move: bool,
    ):
        """Run 7 independent root LLM calls with delayed hedging.

        For each prompt key, we:
        - Fire the primary (root_model) immediately.
        - After a brief delay, submit a secondary (child_model) if no primary result yet.
        - Accept the primary if it returns within the overall LLM budget; otherwise
          keep the secondary. We proceed as soon as all keys have at least one
          result or the budget expires.

        Returns: (player_suggestions: list[BattleOrder], opponent_suggestions: list[BattleOrder])
        """
        from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
        import threading
        from time import perf_counter
        # Determine IO constraint for our side based on availability
        conf_clause = ' Include a field "decision_confidence" containing low/med/high.'
        if have_move and have_switch:
            constraint_prompt_io_player = 'Choose the best action and your output MUST be a JSON object containing either {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}.' + conf_clause + "\n"
        elif have_move:
            constraint_prompt_io_player = 'Choose the best action and your output MUST be a JSON object like {"move":"<move_name>"}.' + conf_clause + "\n"
        else:
            constraint_prompt_io_player = 'Choose the best action and your output MUST be a JSON object like {"switch":"<switch_pokemon_name>"}.' + conf_clause + "\n"

        # These will run in LLM calls that will suggest an action to take
        ours_threat = "\n[DAMAGE] What's our best damaging move against the opponent's active Pokémon this turn?\n"
        ours_maintain = "\n[NON-DAMAGING] What's our best non-damaging move available this turn (e.g. apply a buff or debuff, heal, etc.)?\n"
        ours_longterm = "\n[DAMAGE SWITCH] Let's assume the opponent is guaranteed to switch their active Pokémon. What's our best damaging move against their most likely switch option?\n"
        ours_surprise = "\n[SWITCH] What is our best Pokémon to switch to against the opponent's active Pokémon?\n"

        opp_threat = "\n[OPP DAMAGE] What's our opponent's best damaging move available?\n"
        opp_maintain = "\n[OPP NON-DAMAGING] What's our opponent's best non-damaging move available this turn (e.g. apply a buff or debuff, heal, etc.)?\n"
        opp_longterm = "\n[OPP SWITCH] What's our opponent's best switch option available against our active Pokémon?\n"

        # Compose full prompts
        ours_prompts = [
            ("our_analysis_threat", state_action_prompt + ours_threat),
            ("our_analysis_maintain", state_action_prompt + ours_maintain),
            ("our_analysis_longterm", state_action_prompt + ours_longterm),
            ("our_analysis_surprise", state_action_prompt + ours_surprise),
        ]
        opp_prompts = [
            ("opponent_analysis_threat", state_action_prompt_o + opp_threat),
            ("opponent_analysis_maintain", state_action_prompt_o + opp_maintain),
            ("opponent_analysis_longterm", state_action_prompt_o + opp_longterm),
        ]

        player_suggestions = []
        opponent_suggestions = []

        def _trim(s: str) -> str:
            try:
                return (s or "").replace("\n", " ")[:200]
            except Exception:
                return "<unprintable>"

        # Hedging parameters
        HEDGE_DELAY = 4.0  # seconds before launching secondary
        LLM_BUDGET = 27.0  # overall time budget for all root calls in seconds

        start_t = perf_counter()
        ex = ThreadPoolExecutor(max_workers=10)
        try:
            # Track futures and winners
            # meta: future -> (key, tier, is_opp)
            meta = {}
            # winners: key -> {'res': BattleOrder, 'tier': 'primary'|'secondary'}
            winners = {}

            def _submit(key: str, prompt: str, *, is_opp: bool, tier: str):
                # Log
                if not is_opp:
                    return ex.submit(
                        self.io,
                        retries,
                        system_prompt,
                        state_prompt,
                        constraint_prompt_cot,
                        constraint_prompt_io_player,
                        prompt,
                        battle,
                        sim,
                        False,
                        None,
                        (self.root_model if tier == 'primary' else self.child_model),
                    )
                else:
                    # Append confidence clause for opponent as well without mutating caller's string
                    io_o_with_conf = (constraint_prompt_io_o or '') + conf_clause
                    return ex.submit(
                        self.io,
                        2,
                        system_prompt_o,
                        state_prompt_o,
                        constraint_prompt_cot_o,
                        io_o_with_conf,
                        prompt,
                        battle,
                        sim,
                        True,
                        None,
                        (self.root_model if tier == 'primary' else self.child_model),
                    )

            # Submit primaries
            pending = set()
            for key, prompt in ours_prompts:
                fut = _submit(key, prompt, is_opp=False, tier='primary')
                meta[fut] = (key, 'primary', False)
                pending.add(fut)
            for key, prompt in opp_prompts:
                fut = _submit(key, prompt, is_opp=True, tier='primary')
                meta[fut] = (key, 'primary', True)
                pending.add(fut)

            # After short delay, submit secondaries for keys without results
            # Wait HEDGE_DELAY for any primaries to complete
            wait(pending, timeout=HEDGE_DELAY, return_when=FIRST_COMPLETED)
            # Determine which keys still pending
            prim_keys_pending = set()
            for fut in list(pending):
                k, tier, _ = meta[fut]
                if tier == 'primary' and not fut.done():
                    prim_keys_pending.add(k)
            # Submit secondaries only for keys whose primary not done
            for key, prompt in ours_prompts:
                if key in prim_keys_pending:
                    fut = _submit(key, prompt, is_opp=False, tier='secondary')
                    meta[fut] = (key, 'secondary', False)
                    pending.add(fut)
            for key, prompt in opp_prompts:
                if key in prim_keys_pending:
                    fut = _submit(key, prompt, is_opp=True, tier='secondary')
                    meta[fut] = (key, 'secondary', True)
                    pending.add(fut)

            # Helper to accept a result (prefer primary if it arrives later within budget)
            def _accept_result(k: str, tier: str, is_opp: bool, res):
                if res is None:
                    return
                try:
                    msg = getattr(res, 'message', str(res))
                    conf = getattr(res, 'decision_confidence', None)
                    extra = f" conf={conf}" if conf is not None else ""
                    print(f"[LLM DONE] type={k} model={tier} resp='{msg}'{extra}", flush=True)
                except Exception:
                    pass
                prev = winners.get(k)
                if prev is None:
                    winners[k] = {'res': res, 'tier': tier, 'is_opp': is_opp}
                else:
                    # If previous was secondary and new is primary within budget, prefer primary
                    if prev['tier'] == 'secondary' and tier == 'primary':
                        winners[k] = {'res': res, 'tier': tier, 'is_opp': is_opp}

            # Collect until all keys have a winner or budget expires
            all_keys = {k for k, _ in ours_prompts} | {k for k, _ in opp_prompts}
            while True:
                if set(winners.keys()) == all_keys:
                    break
                remaining = LLM_BUDGET - (perf_counter() - start_t)
                if remaining <= 0.0:
                    break
                done, not_done = wait(pending, timeout=min(0.5, remaining), return_when=FIRST_COMPLETED)
                for fut in list(done):
                    pending.discard(fut)
                    k, tier, is_opp = meta.get(fut, (None, None, None))
                    if k is None:
                        continue
                    try:
                        res = fut.result()
                    except Exception:
                        res = None
                    _accept_result(k, tier, is_opp, res)

                if not done and not pending:
                    break

            # Build suggestion lists from winners (dedupe by message)
            for k, rec in winners.items():
                res = rec['res']
                is_opp = rec['is_opp']
                if res is None:
                    continue
                try:
                    if not is_opp:
                        if not any(getattr(x, 'message', None) == getattr(res, 'message', None) for x in player_suggestions):
                            player_suggestions.append(res)
                    else:
                        if not any(getattr(x, 'message', None) == getattr(res, 'message', None) for x in opponent_suggestions):
                            opponent_suggestions.append(res)
                except Exception:
                    pass

            return player_suggestions, opponent_suggestions
        finally:
            # Do not block on slow stragglers; cancel queued tasks and return immediately
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python <3.9 compatibility: cancel_futures not supported
                ex.shutdown(wait=False)

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

        # Ensure root simulation has an up-to-date Metamon context cloned from live
        try:
            if self.use_metamon_value_function:
                from pokechamp.metamon_bridge import METAMON_BRIDGE as _MB
                try:
                    if getattr(_MB, "has_context", None) and _MB.has_context(root.simulation.battle):
                        _MB.sync_context(root.simulation.battle, battle)
                        if os.getenv("POKECHAMP_MINIMAX_LOG", "0").lower() in {"1", "true", "t", "yes", "y"}:
                            try:
                                print(
                                    f"[Minimax] root sync parent_id={id(battle)} child_id={id(root.simulation.battle)}",
                                    flush=True,
                                )
                            except Exception:
                                pass
                    else:
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
                    # Fallback to clone on any unexpected error
                    try:
                        _MB.clone_context(battle, root.simulation.battle)
                    except Exception:
                        pass
        except Exception:
            pass

        q = [root]
        leaf_nodes = []
        start_time = perf_counter()
        timing_on = os.getenv("POKECHAMP_TIMING_LOG", "0").lower() in {"1", "true", "t", "yes", "y"}
        if timing_on:
            t_llm_total = 0.0
            t_expand_copy = 0.0
            t_expand_init = 0.0
            t_expand_step = 0.0
            t_score_metamon = 0.0

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
            opponent_actions = []
            system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move = node.simulation.get_player_prompt(return_actions=True)
            # Layered orchestration: do not append background strategy to system prompt

            # End if terminal
            if node.simulation.is_terminal() or node.depth == self.K:
                try:
                    leaf_src = "heur"
                    if self.use_metamon_value_function:
                        try:
                            from pokechamp.metamon_bridge import METAMON_BRIDGE as _MB
                            _t0 = perf_counter() if timing_on else None
                            score = _MB.score(node.simulation.battle)
                            if timing_on and _t0 is not None:
                                t_score_metamon += (perf_counter() - _t0)
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
                                model=self.child_model,
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
                # Throttled progress log to avoid clutter
                if _log_on and (_leaf_count % 64 == 0):
                    try:
                        _log(
                            f"Progress: leaves={_leaf_count} metamon={_leaf_metamon} heur={_leaf_heur} llm={_leaf_llm} qsize={len(q)}"
                        )
                    except Exception:
                        pass
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
                    # if dmg_calc_turns <= opp_turns:
                    if dmg_calc_turns <= 1:
                        # try:
                            # Ask LLM to use heuristic tool or minimax search
                            # tool_prompt = '''Based on the current battle state, evaluate whether to use the damage calculator tool or the minimax tree search method. Consider the following factors:

                            #     1. Damage calculator advantages:
                            #     - Useful when a clear type advantage or high-power move is available
                            #     - Effective when the opponent's is not switching and current pokemon is likely to KO opponent

                            #     2. Minimax tree search advantages:
                            #     - Can model opponent behavior and predict future moves
                            #     - Useful in complex situations with multiple viable options
                            #     - Effective when long-term strategy is crucial

                            #     3. Current battle state:
                            #     - Remaining Pokémon on each side
                            #     - Health of active Pokémon
                            #     - Type matchups
                            #     - Available moves and their effects
                            #     - Presence of status conditions or field effects

                            #     4. Uncertainty level:
                            #     - How predictable is the opponent's next move?
                            #     - Are there multiple equally viable options for your next move?

                            #     Evaluate these factors and decide which method would be more beneficial in the current situation. Output your choice in the following JSON format:

                            # {"choice":"damage calculator"} or {"choice":"minimax"}'''

                        #     state_prompt_io = state_prompt + tool_prompt
                        #     llm_output = self.get_llm_action(
                        #         system_prompt="You are a high-level competitive Pokémon Gen 9 OU player making a decision during a battle.",
                        #         user_prompt=state_prompt_io,
                        #         model=self.backend,
                        #         temperature=0.6,
                        #         max_tokens=100,
                        #         json_format=True,
                        #         actions=["damage calculator", "minimax"],
                        #     )
                        #     # Parse JSON
                        #     from pokechamp.llm_player import LLMPlayer
                        #     llm_action_json = LLMPlayer._safe_json_to_dict(llm_output)
                        #     choice_value = None
                        #     if llm_action_json is not None:
                        #         choice_value = llm_action_json.get('choice')

                        #     if isinstance(choice_value, str) and choice_value.lower() != 'minimax':
                        #         decision_method = "damage_calc"
                        #         if _log_on and node.depth == 1:
                        #             try:
                        #                 _log(f"Decision method=damage_calc (LLM choice='{choice_value}') action={str(dmg_calc_out)}")
                        #             except Exception:
                        #                 pass
                        #         if return_opp:
                        #             # Use tool to save time and llm when move makes bigger difference
                        #             return dmg_calc_out, action_opp
                        #         return dmg_calc_out
                        # except:
                        #     print('defaulting to minimax')
                        if return_opp:
                            return dmg_calc_out, action_opp
                    player_actions.append(dmg_calc_out)

            # Root: use Metamon policy priors instead of LLM suggestions
            did_parallel_root = False
            if node.depth == 1:
                try:
                    from pokechamp.metamon_bridge import METAMON_BRIDGE as _MB
                    top_n = int(os.getenv("POKECHAMP_METAMON_PRIORS_TOP_N", 3))
                    try:
                        print(f"[MetamonPriorsDBG] calling policy_top_n_traj topN={top_n} turn={getattr(node.simulation.battle,'turn','?')} battle_id={id(node.simulation.battle)}", flush=True)
                    except Exception:
                        pass
                    # Get raw indices + probs for logging
                    pairs = _MB.policy_top_n_traj(node.simulation.battle, top_n=top_n)
                    # Always-on, concise logs to understand behavior
                    try:
                        idxs = [int(i) for (i, _p) in pairs]
                        probs = [round(float(p), 4) for (_i, p) in pairs]
                        print(f"[MetamonPriors] turn={getattr(node.simulation.battle,'turn','?')} topN idx={idxs} probs={probs}", flush=True)
                    except Exception:
                        pass
                    # Map to local orders (avoid duplicate actor call)
                    priors_orders = []
                    for idx, _p in pairs:
                        try:
                            bo = _MB._action_idx_to_order_local(node.simulation.battle, int(idx))
                        except Exception:
                            bo = None
                        if bo is not None:
                            priors_orders.append(bo)
                    # Log mapped orders
                    try:
                        msgs = [getattr(o, 'message', str(o)) for o in priors_orders]
                        print(f"[MetamonPriors] orders={msgs}", flush=True)
                    except Exception:
                        pass
                    for act in priors_orders:
                        try:
                            if not any(getattr(x, 'message', None) == getattr(act, 'message', None) for x in player_actions):
                                player_actions.append(act)
                        except Exception:
                            pass
                except Exception as e:
                    # Propagate errors for explicit failure unless explicitly ignored elsewhere
                    raise

            ##############################
            # Generate opponent's action #
            ##############################
            # Build baseline opponent actions (always)
            if not did_parallel_root:
                if action_opp is not None:
                    try:
                        opponent_actions.append(self.create_order(action_opp))
                    except Exception:
                        pass

            # Add all opponent switch options (simple, no cap). We gather every
            # benched, non-fainted opponent Pokémon and create a switch order for
            # each, then deduplicate by order.message to avoid duplicates with the
            # heuristic switch below or baseline action.
            try:
                existing_msgs = {getattr(o, 'message', None) for o in opponent_actions}
                for mon in node.simulation.battle.opponent_team.values():
                    # Skip active and fainted/0-HP mons
                    if mon.species == node.simulation.battle.opponent_active_pokemon.species:
                        continue
                    try:
                        if getattr(mon, 'fainted', False) or getattr(mon, 'current_hp', 0) <= 0:
                            continue
                    except Exception:
                        continue
                    try:
                        bo = self.create_order(mon)
                        if bo is None:
                            continue
                        msg = getattr(bo, 'message', None)
                        if msg not in existing_msgs:
                            opponent_actions.append(bo)
                            existing_msgs.add(msg)
                    except Exception:
                        pass
            except Exception:
                pass

            # Heuristic matchup switch action
            best_score = np.inf
            best_action = None
            for mon in node.simulation.battle.opponent_team.values():
                # Skip current active and any fainted/inactive candidates
                if mon.species == node.simulation.battle.opponent_active_pokemon.species:
                    continue
                try:
                    if getattr(mon, 'fainted', False) or getattr(mon, 'current_hp', 0) <= 0:
                        continue
                except Exception:
                    pass
                score = self._estimate_matchup(mon, node.simulation.battle.active_pokemon)
                if score < best_score:
                    best_score = score
                    best_action = mon
            if not did_parallel_root:
                if best_action is not None:
                    try:
                        bo = self.create_order(best_action)
                        # Avoid inserting a duplicate entry if we already added this switch
                        if bo is not None:
                            msg = getattr(bo, 'message', None)
                            if not any(getattr(x, 'message', None) == msg for x in opponent_actions):
                                opponent_actions.append(bo)
                    except Exception:
                        pass

            # Simulate outcome
            print('simulate outcome')
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
                        _c0 = perf_counter() if timing_on else None
                        child_battle = deepcopy(node.simulation.battle)
                        if timing_on and _c0 is not None:
                            t_expand_copy += (perf_counter() - _c0)
                        _i0 = perf_counter() if timing_on else None
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
                            copy_battle=False,
                        )
                        if timing_on and _i0 is not None:
                            t_expand_init += (perf_counter() - _i0)
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
                        _s0 = perf_counter() if timing_on else None
                        node_new.simulation.step(action_p, action_o)
                        if timing_on and _s0 is not None:
                            t_expand_step += (perf_counter() - _s0)
                        if _log_on and node.depth == 1:
                            try:
                                _log(f"child-step child_id={id(node_new.simulation.battle)}")
                            except Exception:
                                pass
                        q.append(node_new)

        # Choose best action according to max or min rule
        print('Choose best action according to max or min rule')
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
        end_time = perf_counter()

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
                if timing_on:
                    try:
                        total = end_time - start_time
                        print(f"[Timing] turn={getattr(battle,'turn','?')} depth={self.K} t_llm={t_llm_total:.2f}s t_expand_copy={t_expand_copy:.2f}s t_expand_init={t_expand_init:.2f}s t_expand_step={t_expand_step:.2f}s t_score={t_score_metamon:.2f}s t_total={total:.2f}s", flush=True)
                    except Exception:
                        pass
            except Exception:
                pass

        if return_opp:
            return action, action_opp
        return action
