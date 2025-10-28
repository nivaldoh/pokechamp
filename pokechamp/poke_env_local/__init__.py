"""poke_env module init."""

import logging
from pathlib import Path

import pokechamp.poke_env_local.environment as environment
import pokechamp.poke_env_local.exceptions as exceptions
import pokechamp.poke_env_local.player as player
import pokechamp.poke_env_local.ps_client as ps_client
import pokechamp.poke_env_local.stats as stats
import pokechamp.poke_env_local.teambuilder as teambuilder
from pokechamp.poke_env_local.data import gen_data, to_id_str
from pokechamp.poke_env_local.exceptions import ShowdownException
from pokechamp.poke_env_local.ps_client import AccountConfiguration
from pokechamp.poke_env_local.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
    ShowdownServerConfiguration,
)
from pokechamp.poke_env_local.stats import compute_raw_stats

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
STATIC_DIR = DATA_DIR / "static"

__logger = logging.getLogger("poke-env")
__stream_handler = logging.StreamHandler()
__formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
__stream_handler.setFormatter(__formatter)
__logger.addHandler(__stream_handler)
logging.addLevelName(25, "PS_ERROR")

__all__ = [
    "AccountConfiguration",
    "LocalhostServerConfiguration",
    "ServerConfiguration",
    "ShowdownException",
    "ShowdownServerConfiguration",
    "PACKAGE_ROOT",
    "DATA_DIR",
    "STATIC_DIR",
    "compute_raw_stats",
    "environment",
    "exceptions",
    "gen_data",
    "player",
    "ps_client",
    "stats",
    "teambuilder",
    "to_id_str",
]
