import ast
from copy import copy, deepcopy
from concurrent.futures import ThreadPoolExecutor, Future
import datetime
import json
import logging
import os
import random
import sys

import numpy as np
from pokechamp.poke_env_local.environment.abstract_battle import AbstractBattle
from pokechamp.poke_env_local.environment.battle import Battle
from pokechamp.poke_env_local.environment.double_battle import DoubleBattle
from pokechamp.poke_env_local.environment.move_category import MoveCategory
from pokechamp.poke_env_local.environment.pokemon import Pokemon
from pokechamp.poke_env_local.environment.side_condition import SideCondition
from pokechamp.poke_env_local.player.player import Player, BattleOrder
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pokechamp.poke_env_local.environment.move import Move
import time
import json
from pokechamp.poke_env_local.data.gen_data import GenData
from pokechamp.gpt_player import GPTPlayer
from pokechamp.llama_player import LLAMAPlayer
from pokechamp.openrouter_player import OpenRouterPlayer
from pokechamp.gemini_player import GeminiPlayer
from pokechamp.ollama_player import OllamaPlayer
from pokechamp.data_cache import (
    get_cached_move_effect,
    get_cached_pokemon_move_dict,
    get_cached_ability_effect,
    get_cached_pokemon_ability_dict,
    get_cached_item_effect,
    get_cached_pokemon_item_dict,
    get_cached_pokedex
)
from pokechamp.minimax_optimizer import (
    get_minimax_optimizer,
    initialize_minimax_optimization,
    fast_battle_evaluation,
    create_battle_state_hash,
    OptimizedSimNode
)
from pokechamp.poke_env_local.player.local_simulation import LocalSim, SimNode
from difflib import get_close_matches
from pokechamp.prompts import get_number_turns_faint, get_status_num_turns_fnt, state_translate, get_gimmick_motivation

DEBUG=False

class LLMPlayer(Player):
    def __init__(self,
                 battle_format,
                 api_key="",
                 backend="gpt-4-1106-preview",
                 temperature=1.0,
                 prompt_algo="io",
                 log_dir=None,
                 team=None,
                 save_replays=None,
                 account_configuration=None,
                 server_configuration=None,
                 K=2,
                 _use_strat_prompt=False,
                 prompt_translate: Callable=state_translate,
                 device=0,
                 llm_backend=None,
                 primary_model: Optional[str] = None,
                 secondary_model: Optional[str] = None,
                 log_level: int = logging.WARNING,
                 use_metamon_value_function: Optional[bool] = True,
                 ):

        super().__init__(battle_format=battle_format,
                         team=team,
                         save_replays=save_replays,
                         account_configuration=account_configuration,
                         server_configuration=server_configuration,
                         log_level=log_level)

        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._battle_last_action : Dict[AbstractBattle, Dict] = {}
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.backend = backend
        self.temperature = temperature
        self.log_dir = log_dir
        self.api_key = api_key
        self.prompt_algo = prompt_algo
        self.gen = GenData.from_format(battle_format)
        self.genNum = self.gen.gen
        self.prompt_translate = prompt_translate

        self.strategy_prompt = ""
        self.team_str = team
        self.use_strat_prompt = _use_strat_prompt
        
        # Use cached data instead of loading files repeatedly
        self.move_effect = get_cached_move_effect()
        # only used in old prompting method, replaced by statistcal sets data
        self.pokemon_move_dict = get_cached_pokemon_move_dict()
        self.ability_effect = get_cached_ability_effect()
        # only used is old prompting method
        self.pokemon_ability_dict = get_cached_pokemon_ability_dict()
        self.item_effect = get_cached_item_effect()
        # unused
        # with open(f"./poke_env/data/static/items/gen8pokemon_item_dict.json", "r") as f:
        #     self.pokemon_item_dict = json.load(f)
        self.pokemon_item_dict = get_cached_pokemon_item_dict()
        self._pokemon_dict = get_cached_pokedex(self.gen.gen)

        self.last_plan = ""
        self._recent_context: List[str] = []
        self._recent_deltas: List[Tuple[float, float]] = []  # (dealt, taken)
        self._last_turn_snapshot: Optional[Dict[str, float]] = None

        # Background team strategy (per-battle) state
        self._bg_strategy_executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=2)
        self._bg_strategy_future_by_tag: Dict[str, Future] = {}
        self._bg_strategy_text_by_tag: Dict[str, str] = {}

        if llm_backend is None:
            print(f"Initializing backend: {backend}")  # Debug logging
            if backend.startswith('ollama/'):
                # Ollama models - extract model name after 'ollama/'
                model_name = backend.replace('ollama/', '')
                print(f"Using Ollama with model: {model_name}")
                self.llm = OllamaPlayer(model=model_name, device=device)
            elif 'gpt' in backend and not backend.startswith('openai/'):
                self.llm = GPTPlayer(self.api_key)
            elif 'llama' == backend:
                self.llm = LLAMAPlayer(device=device)
            elif 'gemini' in backend:
                self.llm = GeminiPlayer(self.api_key)
            elif backend.startswith(('openai/', 'anthropic/', 'google/', 'meta/', 'mistral/', 'cohere/', 'perplexity/', 'deepseek/', 'microsoft/', 'nvidia/', 'huggingface/', 'together/', 'replicate/', 'fireworks/', 'localai/', 'vllm/', 'sagemaker/', 'vertex/', 'bedrock/', 'azure/', 'custom/')):
                # OpenRouter supports hundreds of models from various providers
                self.llm = OpenRouterPlayer(self.api_key)
            else:
                raise NotImplementedError('LLM type not implemented:', backend)
        else:
            self.llm = llm_backend
        self.llm_value = self.llm
        # Primary/secondary model routing
        self.primary_model = primary_model if primary_model is not None else backend
        self.secondary_model = secondary_model if secondary_model is not None else self.primary_model
        self.K = K      # for minimax, SC, ToT
        self.use_optimized_minimax = False
        self._minimax_initialized = False
        # Configuration for time optimization
        self.use_damage_calc_early_exit = True  # Use damage calculator to exit early when advantageous
        self.use_llm_value_function = True  # Use LLM for leaf node evaluation (vs fast heuristic)
        self.max_depth_for_llm_eval = 2  # Only use LLM evaluation for shallow depths to save time
        # Optional Metamon integration (replaces LLM eval)
        if use_metamon_value_function is None:
            use_metamon_value_function = os.getenv("POKECHAMP_USE_METAMON", "1") in {"1", "true", "True"}
        self.use_metamon_value_function = bool(use_metamon_value_function)
        if self.use_metamon_value_function:
            # Disable LLM value if Metamon is enabled, but only after successful init
            try:
                from pokechamp import metamon_bridge as _mb
                # Strict by default: require Metamon unless explicitly allowed to fall back
                required_env = os.getenv("POKECHAMP_METAMON_REQUIRED", "1").lower()
                metamon_required = required_env in {"1", "true", "t", "yes", "y"}
                _mb.enable(os.getenv("POKECHAMP_METAMON_MODEL"), strict=metamon_required)
                # If strict=False, still try to load model and check readiness
                if not metamon_required:
                    try:
                        _mb.METAMON_BRIDGE._ensure_model()
                    except Exception as _e2:
                        print(f"[MetamonBridge] lazy model load failed: {_e2}")
                ready, reason = _mb.METAMON_BRIDGE.is_ready()
                if not ready:
                    if metamon_required:
                        raise RuntimeError(f"Metamon required but not ready: {reason}")
                    else:
                        print(f"[MetamonBridge] not ready (optional): {reason}. Falling back to LLM for leaf eval.")
                        self.use_metamon_value_function = False
                        self.use_llm_value_function = True
                else:
                    # Metamon ready; turn off LLM leaf eval
                    self.use_llm_value_function = False
                    cache_dir = os.getenv("METAMON_CACHE_DIR")
                    model = os.getenv("POKECHAMP_METAMON_MODEL", "Abra")
                    head = "loaded" if getattr(_mb.METAMON_BRIDGE, "_head_loaded", False) else "untrained"
                    print(f"[MetamonBridge] enabled model={model} head={head} cache={cache_dir}")
            except Exception as _e:
                # Required path: raise and stop early
                print(f"[MetamonBridge] initialization error: {_e}")
                required_env = os.getenv("POKECHAMP_METAMON_REQUIRED", "1").lower()
                metamon_required = required_env in {"1", "true", "t", "yes", "y"}
                if metamon_required:
                    raise
                # Optional path: fallback to LLM leaves
                self.use_metamon_value_function = False
                self.use_llm_value_function = True

    def get_LLM_action(self, system_prompt, user_prompt, model, temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=200, actions=None, llm=None) -> str:
        if llm is None:
            output, _ = self.llm.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens=max_tokens, actions=actions)
        else:
            output, _ = llm.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens=max_tokens, actions=actions)
        return output

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove surrounding Markdown code fences from text."""
        if not isinstance(text, str):
            return text

        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped

        fence_end = stripped.rfind("```")
        if fence_end <= 3:
            return stripped

        first_newline = stripped.find("\n")
        if first_newline == -1:
            return stripped[3:fence_end].strip()

        return stripped[first_newline + 1:fence_end].strip()

    @staticmethod
    def _extract_json_snippet(text: str) -> Optional[str]:
        """Extract the first balanced JSON object or array contained in text."""
        if not isinstance(text, str):
            return None

        def _extract(source: str, opener: str, closer: str) -> Optional[str]:
            depth = 0
            start = -1
            for idx, char in enumerate(source):
                if char == opener:
                    if depth == 0:
                        start = idx
                    depth += 1
                elif char == closer and depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        return source[start:idx + 1]
            return None

        snippet = _extract(text, "{", "}")
        if snippet is not None:
            return snippet
        return _extract(text, "[", "]")

    @staticmethod
    def _safe_json_to_dict(payload: Any) -> Optional[Dict[str, Any]]:
        """Best-effort conversion of payload into a dictionary."""

        if payload is None:
            return None

        if isinstance(payload, dict):
            return payload

        if isinstance(payload, list):
            for item in payload:
                candidate = LLMPlayer._safe_json_to_dict(item)
                if candidate is not None:
                    return candidate
            return None

        if isinstance(payload, str):
            candidate = LLMPlayer._strip_code_fences(payload)
            if not candidate:
                return None

            try:
                return LLMPlayer._safe_json_to_dict(json.loads(candidate))
            except json.JSONDecodeError:
                snippet = LLMPlayer._extract_json_snippet(candidate)
                if snippet:
                    try:
                        return LLMPlayer._safe_json_to_dict(json.loads(snippet))
                    except json.JSONDecodeError:
                        return None
                return None

        return None

    def check_all_pokemon(self, pokemon_str: str) -> Pokemon:
        valid_pokemon = None
        if pokemon_str in self._pokemon_dict:
            valid_pokemon = pokemon_str
        else:
            closest = get_close_matches(pokemon_str, self._pokemon_dict.keys(), n=1, cutoff=0.8)
            if len(closest) > 0:
                valid_pokemon = closest[0]
        if valid_pokemon is None:
            return None
        pokemon = Pokemon(species=pokemon_str, gen=self.genNum)
        return pokemon

    def choose_move(self, battle: AbstractBattle):
        # Kick off / poll background strategy loop and adopt latest text
        try:
            self._maybe_start_bg(battle)
            self._maybe_poll_bg(battle)
            # Expose the latest strategy text for downstream prompts
            self.strategy_prompt = self._get_bg_strategy_text(battle)
        except Exception:
            pass

        sim = LocalSim(battle, 
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    self.strategy_prompt,
                    format=self.format,
                    prompt_translate=self.prompt_translate
        )
        if battle.turn <=1 and self.use_strat_prompt:
            self.strategy_prompt = sim.get_llm_system_prompt(self.format, self.llm, team_str=self.team_str, model='gpt-4o-2024-05-13')
        
        if battle.active_pokemon:
            if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
                next_action = BattleOrder(battle.available_switches[0])
                return next_action
            elif not battle.active_pokemon.fainted and len(battle.available_moves) == 1 and len(battle.available_switches) == 0:
                return self.choose_max_damage_move(battle)
        elif len(battle.available_moves) <= 1 and len(battle.available_switches) == 0:
            return self.choose_max_damage_move(battle)

        system_prompt, state_prompt, state_action_prompt = sim.state_translate(battle) # add lower case
        moves = [move.id for move in battle.available_moves]
        switches = [pokemon.species for pokemon in battle.available_switches]
        actions = [moves, switches]
        

        gimmick_output_format = ''
        if 'pokellmon' not in self.ps_client.account_configuration.username: # make sure we dont mess with pokellmon original strat
            gimmick_output_format = f'{f' or {{"dynamax":"<move_name>"}}' if battle.can_dynamax else ''}{f' or {{"terastallize":"<move_name>"}}' if battle.can_tera else ''}'

        if battle.active_pokemon.fainted or len(battle.available_moves) == 0:

            constraint_prompt_io = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the most suitable pokemon to switch by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best switch options. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
        elif len(battle.available_switches) == 0:
            constraint_prompt_io = f'''Choose the best action and your output MUST be a JSON like: {{"move":"<move_name>"}}{gimmick_output_format}\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best action options. Your output MUST be a JSON like: {"option_1":{"action":"<move>", "target":"<move_name>"}, ..., "option_k":{"action":"<move>", "target":"<move_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best action from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:"decision":{"action":"<move>", "target":"<move_name>"}\n'''
        else:
            constraint_prompt_io = f'''Choose the best action and your output MUST be a JSON like: {{"move":"<move_name>"}}{gimmick_output_format} or {{"switch":"<switch_pokemon_name>"}}\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best action options. Your output MUST be a JSON like: {"option_1":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}, ..., "option_k":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best action from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:"decision":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}\n'''

        state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_io
        state_prompt_cot = state_prompt + state_action_prompt + constraint_prompt_cot
        state_prompt_tot_1 = state_prompt + state_action_prompt + constraint_prompt_tot_1
        state_prompt_tot_2 = state_prompt + state_action_prompt + constraint_prompt_tot_2

        retries = 10
        # Chain-of-thought
        if self.prompt_algo == "io":
            return self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=actions)

        # Self-consistency with k = 3
        elif self.prompt_algo == "sc":
            return self.sc(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim)

        # Tree of thought, k = 3
        elif self.prompt_algo == "tot":
            llm_output1 = ""
            next_action = None
            for i in range(retries):
                try:
                    llm_output1 = self.get_LLM_action(system_prompt=system_prompt,
                                               user_prompt=state_prompt_tot_1,
                                               model=self.backend,
                                               temperature=self.temperature,
                                               max_tokens=200,
                                               json_format=True)
                    break
                except:
                    raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                    continue

            if llm_output1 == "":
                return self.choose_max_damage_move(battle)

            for i in range(retries):
                try:
                    llm_output2 = self.get_LLM_action(system_prompt=system_prompt,
                                               user_prompt=state_prompt_tot_2.replace("[OPTIONS]", llm_output1),
                                               model=self.backend,
                                               temperature=self.temperature,
                                               max_tokens=10000,
                                               json_format=True)

                    next_action = self.parse_new(llm_output2, battle, sim)
                    with open(f"{self.log_dir}/output.jsonl", "a") as f:
                        f.write(json.dumps({"turn": battle.turn,
                                            "system_prompt": system_prompt,
                                            "user_prompt1": state_prompt_tot_1,
                                            "user_prompt2": state_prompt_tot_2,
                                            "llm_output1": llm_output1,
                                            "llm_output2": llm_output2,
                                            "battle_tag": battle.battle_tag
                                            }) + "\n")
                    if next_action is not None:     break
                except:
                    raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                    continue

            if next_action is None:
                next_action = self.choose_max_damage_move(battle)
            return next_action

        elif self.prompt_algo == "minimax":
            try:
                # Lazy import to avoid hard dependency
                from pokechamp.minimax_strategy import MinimaxStrategy

                # Helper function to build LocalSim
                def build_local_sim(b):
                    return LocalSim(
                        b,
                        self.move_effect,
                        self.pokemon_move_dict,
                        self.ability_effect,
                        self.pokemon_ability_dict,
                        self.item_effect,
                        self.pokemon_item_dict,
                        self.gen,
                        self._dynamax_disable,
                        self.strategy_prompt,
                        format=self.format,
                        prompt_translate=self.prompt_translate
                    )

                # Instantiate strategy with dependencies
                strategy = MinimaxStrategy(
                    move_effect=self.move_effect,
                    pokemon_move_dict=self.pokemon_move_dict,
                    ability_effect=self.ability_effect,
                    pokemon_ability_dict=self.pokemon_ability_dict,
                    item_effect=self.item_effect,
                    pokemon_item_dict=self.pokemon_item_dict,
                    gen=self.gen,
                    battle_format=self.format,
                    K=self.K,
                    temperature=self.temperature,
                    backend=self.backend,
                    use_llm_value_function=self.use_llm_value_function,
                    use_metamon_value_function=self.use_metamon_value_function,
                    max_depth_for_llm_eval=self.max_depth_for_llm_eval,
                    _dynamax_disable=self._dynamax_disable,
                    prompt_translate=self.prompt_translate,
                    get_llm_action=self.get_LLM_action,
                    io=self.io,
                    create_order=self.create_order,
                    choose_max_damage_move=self.choose_max_damage_move,
                    build_local_sim=build_local_sim,
                    llm_value=self.llm_value,
                    get_strategy_text=lambda b: self._get_bg_strategy_text(b),
                    root_model=self.primary_model,
                    child_model=self.secondary_model,
                )

                # Delegate to strategy
                return strategy.choose_move(battle)
            except Exception as e:
                print(f'minimax step failed ({e}). Using dmg calc')
                print(f'Exception: {e}', 'passed')
                return self.choose_max_damage_move(battle)

        
    def io(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle: Battle, sim, dont_verify=False, actions=None, model: Optional[str] = None):
        next_action = None
        cot_prompt = 'In fewer than 3 sentences, let\'s think step by step:'
        state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_io + cot_prompt

        for i in range(retries):
            try:
                llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                            user_prompt=state_prompt_io,
                                            model=(model or self.backend),
                                            temperature=self.temperature,
                                            max_tokens=2000,
                                            # stop=["reason"],
                                            json_format=True,
                                            actions=actions)

                # load when llm does heavylifting for parsing
                if DEBUG:
                    print(f"Raw LLM output: {llm_output}")
                llm_action_json = json.loads(llm_output)
                if DEBUG:
                    print(f"Parsed JSON: {llm_action_json}")
                next_action = None

                dynamax = "dynamax" in llm_action_json.keys()
                tera = "terastallize" in llm_action_json.keys()
                is_a_move = dynamax or tera

                if "move" in llm_action_json.keys() or is_a_move:
                    if dynamax:
                        llm_move_id = llm_action_json["dynamax"].strip()
                    elif tera:
                        llm_move_id = llm_action_json["terastallize"].strip()
                    else:
                        llm_move_id = llm_action_json["move"].strip()
                    move_list = battle.available_moves
                    if dont_verify: # opponent
                        move_list = battle.opponent_active_pokemon.moves.values()
                    
                    # Debug: print available moves
                    if DEBUG:
                        print(f"LLM requested move: '{llm_move_id}'")
                        print(f"Available moves: {[move.id for move in move_list]}")
                    
                    for i, move in enumerate(move_list):
                        if move.id.lower().replace(' ', '') == llm_move_id.lower().replace(' ', ''):
                            #next_action = self.create_order(move, dynamax=sim._should_dynamax(battle), terastallize=sim._should_terastallize(battle))
                            next_action = self.create_order(move, dynamax=dynamax, terastallize=tera)
                            if DEBUG:
                                print(f"Move match found: {move.id}")
                            break
                    
                    if next_action is None and dont_verify:
                        # unseen move so just check if it is in the action prompt
                        if llm_move_id.lower().replace(' ', '') in state_action_prompt:
                            next_action = self.create_order(Move(llm_move_id.lower().replace(' ', ''), self.gen.gen), dynamax=dynamax, terastallize=tera)
                    
                    if next_action is None and DEBUG:
                        print(f"No move match found for '{llm_move_id}'")
                elif "switch" in llm_action_json.keys():
                    llm_switch_species = llm_action_json["switch"].strip()
                    switch_list = battle.available_switches
                    if dont_verify: # opponent prediction
                        observable_switches = []
                        for _, opponent_pokemon in battle.opponent_team.items():
                            if not opponent_pokemon.active:
                                observable_switches.append(opponent_pokemon)
                        switch_list = observable_switches
                    
                    # Debug: print available switches
                    if DEBUG:
                        print(f"LLM requested switch: '{llm_switch_species}'")
                        print(f"Available switches: {[pokemon.species for pokemon in switch_list]}")
                    
                    for i, pokemon in enumerate(switch_list):
                        if pokemon.species.lower().replace(' ', '') == llm_switch_species.lower().replace(' ', ''):
                            next_action = self.create_order(pokemon)
                            if DEBUG:
                                print(f"Switch match found: {pokemon.species}")
                            break
                    
                else:
                    raise ValueError('No valid action')
                
                # with open(f"{self.log_dir}/output.jsonl", "a") as f:
                #     f.write(json.dumps({"turn": battle.turn,
                #                         "system_prompt": system_prompt,
                #                         "user_prompt": state_prompt_io,
                #                         "llm_output": llm_output,
                #                         "battle_tag": battle.battle_tag
                #                         }) + "\n")
                
                if next_action is not None:
                    break
            except Exception as e:
                print(f'Exception: {e}', 'passed')
                continue
        if next_action is None:
            print('No action found. Choosing max damage move')
            try:
                print('No action found', llm_action_json, actions, dont_verify)
            except:
                pass
            print()
            # raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
            next_action = self.choose_max_damage_move(battle)
        return next_action

    # -------------------- Background team strategy helpers --------------------
    def _build_strategy_snapshot(self, battle: Battle) -> str:
        """Build a compact snapshot of both teams and current state for the strategy LLM.

        Includes:
        - Our team: species, revealed item, ability, up to 4 known moves
        - Opponent (revealed/preview): species, revealed item/ability, known moves
        - Active mons, HP/status, simple hazards
        - Tera availability for both sides
        """
        try:
            # Our team
            our_entries = []
            for mon in battle.team.values():
                try:
                    moves = list(mon.moves.keys())[:4]
                except Exception:
                    moves = []
                our_entries.append(
                    {
                        "species": getattr(mon, "species", None),
                        "item": getattr(mon, "item", None),
                        "ability": getattr(mon, "ability", None),
                        "moves": moves,
                    }
                )

            # Opponent entries (revealed or preview)
            opp_entries = []
            try:
                # Prefer opponent_team values (revealed)
                for mon in battle.opponent_team.values():
                    try:
                        moves = list(mon.moves.keys())[:4]
                    except Exception:
                        moves = []
                    opp_entries.append(
                        {
                            "species": getattr(mon, "species", None),
                            "item": getattr(mon, "item", None),
                            "ability": getattr(mon, "ability", None),
                            "moves": moves,
                        }
                    )
            except Exception:
                pass

            # If we have a team preview list exposed via opponent_team during preview,
            # the above will already include them. Otherwise, attempt a safe preview fetch.
            # (We avoid relying on non-public attrs.)

            # Active state
            def _mk_active(mon):
                if mon is None:
                    return None
                return {
                    "species": getattr(mon, "species", None),
                    "hp": getattr(mon, "current_hp", None),
                    "max_hp": getattr(mon, "max_hp", None),
                    "status": getattr(mon, "status", None).name if getattr(mon, "status", None) is not None else None,
                }

            active_our = _mk_active(getattr(battle, "active_pokemon", None))
            active_opp = _mk_active(getattr(battle, "opponent_active_pokemon", None))

            # Hazards (simple string keys)
            def _side_to_names(side):
                try:
                    return [getattr(k, "name", str(k)).lower() for k in side.keys()]
                except Exception:
                    return []

            hazards_our = _side_to_names(getattr(battle, "side_conditions", {}))
            hazards_opp = _side_to_names(getattr(battle, "opponent_side_conditions", {}))

            # Tera flags
            our_can_tera = getattr(battle, "can_tera", None) is not None
            opp_can_tera = False
            for name in ("opponent_can_terrastallize", "_opponent_can_terrastallize"):
                val = getattr(battle, name, None)
                if isinstance(val, bool):
                    opp_can_tera = val
                    break

            snap = {
                "turn": getattr(battle, "turn", None),
                "our_team": our_entries,
                "opp_team": opp_entries,
                "active": {"our": active_our, "opp": active_opp},
                "hazards": {"our": hazards_our, "opp": hazards_opp},
                "tera": {"our_can_tera": our_can_tera, "opp_can_tera": opp_can_tera},
            }
            return json.dumps(snap, ensure_ascii=False)
        except Exception:
            # Fallback: minimal string
            return "{}"

    def _submit_bg_strategy(self, battle: Battle) -> Future:
        """Submit a background LLM call to build/update the strategy text for this battle."""
        system_prompt = """
        SYSTEM
You are a Gen 9 OU battle assistant. Given a minimal snapshot, output a terse, actionable high-level gameplan that keeps long-term win conditions in mind. 
- Infer roles/sets only from revealed info. If unknown, write "unknown".
- Assume the opponent may still Terastallize unless explicitly stated otherwise; plan safe lines that hold vs common Tera types.
- Prioritize: 1) preserve/enable our win condition(s); 2) neutralize immediate threats; 3) hazards plan (set/clear/ignore); 4) momentum via pivots/safe sacks.
- Do NOT try to do math.
- Be concise and structured. No prose explanations, no move calcs, no speculation beyond revealed info.
- Hard cap: ≤ 900 characters total.

OUTPUT (strict JSON; one-liners, no extra text)
{
  "summary": "<1 sentence on board & immediate pressure>",
  "wincons": ["<primary plan>", "<backup plan>"],
  "threats": ["<top 1-3 threats by name/why in 3-5 words>"],
  "hazards_plan": "<set | clear | ignore + 3-5 words>",
  "pivot_plan": "<safe pivot or stay; how to gain tempo>",
  "tera_plan": {
    "use_now": "<yes|no|maybe>",
    "ours": "<who/type and why in ≤8 words>",
    "opp_checks": ["<if they Tera X, do Y>", "..."]
  },
  "status_effects": {
    "ours": "<1-3 status effects we can apply on ourselves or opponents>"
    "theirs": "<1-3 status effects the opponent can apply on themselves or opponents>"
  }
  "sack_order": ["<optional, 1-2 expendable mons or 'none'>"]
}

        """
        user_prompt = self._build_strategy_snapshot(battle)

        def _job():
            try:
                try:
                    print(f"[Strategy BG START] battle={getattr(battle, 'battle_tag', '?')} turn={getattr(battle, 'turn', '?')}", flush=True)
                except Exception:
                    pass
                t0 = time.time()
                out = self.get_LLM_action(system_prompt=system_prompt,
                                          user_prompt=user_prompt,
                                          model=(self.primary_model if hasattr(self, 'primary_model') else self.backend),
                                          temperature=self.temperature,
                                          max_tokens=300)
                dt = time.time() - t0
                try:
                    print(f"[Strategy BG DONE] battle={getattr(battle, 'battle_tag', '?')} (+{dt:.2f}s) len={len(out) if isinstance(out, str) else 0}", flush=True)
                except Exception:
                    pass
                return out if isinstance(out, str) else ""
            except Exception as e:
                try:
                    print(f"[Strategy BG ERROR] battle={getattr(battle, 'battle_tag', '?')} err={e}", flush=True)
                except Exception:
                    pass
                return ""

        return self._bg_strategy_executor.submit(_job)

    def _has_sufficient_team_info(self, battle: Battle) -> bool:
        try:
            our_ok = len(getattr(battle, "team", {}) or {}) >= 1
            # Opponent preview or at least revealed active/opponent_team
            opp_ok = False
            # Use opponent_team values if available
            opp_ok = opp_ok or (len(getattr(battle, "opponent_team", {}) or {}) >= 1)
            # Active opponent mon known
            opp_ok = opp_ok or (getattr(battle, "opponent_active_pokemon", None) is not None)
            return bool(our_ok and opp_ok)
        except Exception:
            return False

    def _maybe_start_bg(self, battle: Battle) -> None:
        tag = getattr(battle, "battle_tag", None)
        if tag is None:
            return
        if tag in self._bg_strategy_future_by_tag:
            return
        if not self._has_sufficient_team_info(battle):
            return
        fut = self._submit_bg_strategy(battle)
        self._bg_strategy_future_by_tag[tag] = fut

    def _maybe_poll_bg(self, battle: Battle) -> None:
        tag = getattr(battle, "battle_tag", None)
        if tag is None:
            return
        fut = self._bg_strategy_future_by_tag.get(tag)
        if fut is None:
            return
        if not fut.done():
            return
        # Adopt result
        try:
            out = fut.result()
            if isinstance(out, str) and out.strip():
                self._bg_strategy_text_by_tag[tag] = out.strip()
        except Exception:
            pass
        # Submit next job immediately
        self._bg_strategy_future_by_tag[tag] = self._submit_bg_strategy(battle)

    def _get_bg_strategy_text(self, battle: Battle) -> str:
        tag = getattr(battle, "battle_tag", None)
        if tag is None:
            return ""
        return self._bg_strategy_text_by_tag.get(tag, "")

    def sc(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim):
        actions = [self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim) for i in range(self.K)]
        action_message = [action.message for action in actions]
        _, counts = np.unique(action_message, return_counts=True)
        index = np.argmax(counts)
        return actions[index]
    
    def battle_summary(self):

        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        for tag, battle in self.battles.items():
            beat_score = 0
            for mon in battle.opponent_team.values():
                beat_score += (1-mon.current_hp_fraction)

            beat_list.append(beat_score)

            remain_score = 0
            for mon in battle.team.values():
                remain_score += mon.current_hp_fraction

            remain_list.append(remain_score)
            if battle.won:
                win_list.append(1)

            tag_list.append(tag)

        return beat_list, remain_list, win_list, tag_list

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards."""

        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle] # the return value is the delta
        self._reward_buffer[battle] = current_value

        return to_return

    def choose_max_damage_move(self, battle: Battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)
