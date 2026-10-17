"""
Global data cache for Pokemon simulation data.

This module provides a singleton cache for all static game data to avoid 
repeated file loading during battles and simulations.
"""

import json
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict

import orjson

from pokechamp.poke_env_local.paths import STATIC_DIR
from pokechamp.poke_env_local.data.gen_data import GenData


class GameDataCache:
    """Singleton cache for static game data."""
    
    _instance = None
    _data = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Initialize data only once
        if not self._data:
            self._load_all_data()
    
    def _load_all_data(self):
        """Load all static game data into memory."""
        print("🔄 Loading static game data into cache...")
        
        # Move effects and Pokemon move mappings
        try:
            with (STATIC_DIR / "moves" / "moves_effect.json").open("r") as f:
                self._data['move_effect'] = json.load(f)
        except FileNotFoundError:
            print("⚠️  moves_effect.json not found, using empty dict")
            self._data['move_effect'] = {}
            
        try:
            with (STATIC_DIR / "moves" / "gen8pokemon_move_dict.json").open("r") as f:
                self._data['pokemon_move_dict'] = json.load(f)
        except FileNotFoundError:
            print("⚠️  gen8pokemon_move_dict.json not found, using empty dict")
            self._data['pokemon_move_dict'] = {}
        
        # Ability effects and Pokemon ability mappings
        try:
            with (STATIC_DIR / "abilities" / "ability_effect.json").open("r") as f:
                self._data['ability_effect'] = json.load(f)
        except FileNotFoundError:
            print("⚠️  ability_effect.json not found, using empty dict")
            self._data['ability_effect'] = {}
            
        try:
            with (STATIC_DIR / "abilities" / "gen8pokemon_ability_dict.json").open("r") as f:
                self._data['pokemon_ability_dict'] = json.load(f)
        except FileNotFoundError:
            print("⚠️  gen8pokemon_ability_dict.json not found, using empty dict")
            self._data['pokemon_ability_dict'] = {}
        
        # Item effects
        try:
            with (STATIC_DIR / "items" / "item_effect.json").open("r") as f:
                self._data['item_effect'] = json.load(f)
        except FileNotFoundError:
            print("⚠️  item_effect.json not found, using empty dict")
            self._data['item_effect'] = {}
        
        # Pokemon item mappings (if needed)
        self._data['pokemon_item_dict'] = {}  # Currently unused
        
        print("✅ Static game data loaded into cache")
    
    def get_move_effect(self) -> Dict[str, Any]:
        """Get cached move effects data."""
        return self._data['move_effect']
    
    def get_pokemon_move_dict(self) -> Dict[str, Any]:
        """Get cached Pokemon move mappings."""
        return self._data['pokemon_move_dict']
    
    def get_ability_effect(self) -> Dict[str, Any]:
        """Get cached ability effects data."""
        return self._data['ability_effect']
    
    def get_pokemon_ability_dict(self) -> Dict[str, Any]:
        """Get cached Pokemon ability mappings."""
        return self._data['pokemon_ability_dict']
    
    def get_item_effect(self) -> Dict[str, Any]:
        """Get cached item effects data."""
        return self._data['item_effect']
    
    def get_pokemon_item_dict(self) -> Dict[str, Any]:
        """Get cached Pokemon item mappings."""
        return self._data['pokemon_item_dict']
    
    @lru_cache(maxsize=5)  # Cache up to 5 different generations
    def get_pokedex(self, gen: int) -> Dict[str, Any]:
        """Get cached Pokedex data for a specific generation."""
        cache_key = f'pokedex_gen{gen}'
        
        if cache_key not in self._data:
            try:
                with (STATIC_DIR / "pokedex" / f"gen{gen}pokedex.json").open("r") as f:
                    self._data[cache_key] = json.load(f)
                print(f"✅ Loaded gen{gen} Pokedex data")
            except FileNotFoundError:
                print(f"⚠️  gen{gen}pokedex.json not found, using empty dict")
                self._data[cache_key] = {}
        
        return self._data[cache_key]
    
    @lru_cache(maxsize=10)  # Cache up to 10 different formats
    def get_moves_set(self, format: str) -> Dict[str, Any]:
        """Get cached moves set data for a specific format."""
        cache_key = f'moves_set_{format}'
        
        if cache_key not in self._data:
            try:
                if format == 'gen9ou':
                    file_path = STATIC_DIR / 'gen9' / 'ou' / 'sets_1000.json'
                else:
                    # Add more formats as needed
                    file_path = STATIC_DIR / format / 'sets_1000.json'

                with Path(file_path).open('r') as f:
                    self._data[cache_key] = orjson.loads(f.read())
                print(f"✅ Loaded {format} moves set data")
            except FileNotFoundError:
                print(f"⚠️  {format} moves set not found, using empty dict")
                self._data[cache_key] = {}
        
        return self._data[cache_key]
    
    def clear_cache(self):
        """Clear all cached data (useful for testing)."""
        self._data.clear()
        print("🧹 Cache cleared")


# Global cache instance
_cache = GameDataCache()


def get_cached_move_effect() -> Dict[str, Any]:
    """Get cached move effects data."""
    return _cache.get_move_effect()


def get_cached_pokemon_move_dict() -> Dict[str, Any]:
    """Get cached Pokemon move mappings."""
    return _cache.get_pokemon_move_dict()


def get_cached_ability_effect() -> Dict[str, Any]:
    """Get cached ability effects data."""
    return _cache.get_ability_effect()


def get_cached_pokemon_ability_dict() -> Dict[str, Any]:
    """Get cached Pokemon ability mappings."""
    return _cache.get_pokemon_ability_dict()


def get_cached_item_effect() -> Dict[str, Any]:
    """Get cached item effects data."""
    return _cache.get_item_effect()


def get_cached_pokemon_item_dict() -> Dict[str, Any]:
    """Get cached Pokemon item mappings."""
    return _cache.get_pokemon_item_dict()


def get_cached_pokedex(gen: int) -> Dict[str, Any]:
    """Get cached Pokedex data for a specific generation."""
    return _cache.get_pokedex(gen)


def get_cached_moves_set(format: str) -> Dict[str, Any]:
    """Get cached moves set data for a specific format."""
    return _cache.get_moves_set(format)


def clear_data_cache():
    """Clear all cached data."""
    _cache.clear_cache()
