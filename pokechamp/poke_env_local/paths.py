from pathlib import Path

# Standalone module for path constants to avoid circular imports.
# Any submodule can import from pokechamp.poke_env_local.paths safely.

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
STATIC_DIR = DATA_DIR / "static"

__all__ = [
    "PACKAGE_ROOT",
    "DATA_DIR",
    "STATIC_DIR",
]

