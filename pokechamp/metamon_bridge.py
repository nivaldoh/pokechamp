"""
MetamonBridge: Runtime bridge that mirrors Showdown messages/requests (both live messages as well as messages from simulated nodes?) into a
Metamon trajectory context, and provides a leaf scoring API for minimax.

Design goals
- Do not modify poke_env. Install lightweight monkey‑patches at runtime.
- Keep one context per Python Battle instance (keyed by id(battle)).
- Maintain a live root context from real server traffic; clone it for branches.
- Provide score(battle) -> float in [0, 100] using a pretrained Metamon model.

Notes
- Default model is tuned for Gen 9 and uses DefaultObservationSpace‑v1 tokens via
  TeamPreviewObservationSpace. You can override via env POKECHAMP_METAMON_MODEL.
"""

from __future__ import annotations

import copy
import os
import threading
import re
from typing import Dict, Optional, Any

import numpy as np
import traceback
import json

try:
    import torch
except Exception:  # pragma: no cover - optional at import time
    torch = None  # type: ignore

# metamon installed as an editable module from the pokechamp root folder
from metamon.env.metamon_battle import MetamonBackendBattle
from metamon.rl.pretrained import get_pretrained_model
# No win-probability head; we use critics for scoring
from metamon.interface import UniversalState, consistent_move_order, consistent_pokemon_order
from pokechamp.poke_env_local.player.player import Player as LocalPlayer
from pokechamp.metamon_normalizer import SimProtocolNormalizer


class _IncompleteObs(Exception):
    """Raised internally to indicate replay info is not complete enough yet."""
    pass


class _Ctx:
    """Per‑battle trajectory context.

    Attributes:
        mm_battle: MetamonBackendBattle that accumulates replay state from messages/requests.
        hidden: Persistent hidden state tensor for the traj encoder (shape handled by model).
        time_idx: Integer turn index for encoding sequence steps.
        ephemeral: True when created for a simulated branch.
        player_role: Optional cached role (p1/p2) for convenience.
    """

    def __init__(self, mm_battle: MetamonBackendBattle, *, hidden=None, time_idx: int = 0, ephemeral: bool = False):
        self.mm_battle = mm_battle
        self.hidden = hidden
        self.time_idx = time_idx
        self.ephemeral = ephemeral
        self.player_role: Optional[str] = None
        # Last observation source used by _build_obs_np: 'replay' | 'battle' | None
        self.last_obs_src: Optional[str] = None


class MetamonBridge:
    def __init__(self):
        self._contexts: Dict[int, _Ctx] = {}
        self._installed = False
        self._lock = threading.RLock()

        # Model/runtime fields (lazy)
        self._experiment = None
        self._agent = None
        self._device = None
        self._obs_space = None
        self._rl2_width: Optional[int] = None
        # No trajectory head; critic-based scoring only

        # Cache hooks (per-target)
        self._patched: Dict[str, bool] = {}
        self._orig_parse_message: Dict[str, Any] = {}
        self._orig_parse_request: Dict[str, Any] = {}
        # Pre-log sequence counters per battle
        self._prelog_seq: Dict[int, int] = {}

        # (Removed) Lightweight runtime stats and psutil-based memory snapshots
        # These were previously used for ad-hoc instrumentation and health logs.

        # Configuration
        self._model_name = os.getenv("POKECHAMP_METAMON_MODEL", "Abra")
        # Optional legacy head checkpoint env is ignored; critics are used for scoring
        self._winprob_ckpt = os.getenv("METAMON_WINPROB_CKPT")
        self._debug = str(os.getenv("POKECHAMP_METAMON_DEBUG", "")).lower() in {"1", "true", "t", "yes", "y"}
        # LocalSim normalization layer
        self._normalizer = SimProtocolNormalizer()
        # Trajectory mode only

    # -------- Debug helper --------
    def _dbg(self, *args: Any, **kwargs: Any) -> None:
        if self._debug:
            try:
                print("[MetamonBridge]", *args, **kwargs, flush=True)
            except Exception:
                pass

    def _dbg_exc(self, prefix: str, exc: BaseException) -> None:
        if self._debug:
            etype = type(exc).__name__
            msg = str(exc)
            tb = traceback.format_exc()
            try:
                print(f"[MetamonBridge] {prefix}: {etype}: {msg}\n{tb}", flush=True)
            except Exception:
                pass

    # -------- Failure diagnostics helpers (debug only) --------
    def _battle_snapshot(self, battle) -> Dict[str, Any]:
        try:
            act = getattr(battle, "active_pokemon", None)
            opp = getattr(battle, "opponent_active_pokemon", None)
            def _mon(m):
                if m is None:
                    return None
                try:
                    return {
                        "species": getattr(m, "species", None),
                        "base_species": getattr(m, "base_species", None),
                        "hp": getattr(m, "current_hp", None),
                        "max_hp": getattr(m, "max_hp", None),
                        "status": getattr(m, "status", None).name if getattr(m, "status", None) is not None else None,
                        "moves_len": len(getattr(m, "moves", {}) or {}),
                        "known_moves": list((getattr(m, "moves", {}) or {}).keys())[:4],
                    }
                except Exception:
                    return {"species": getattr(m, "species", None)}

            snap = {
                "battle_id": id(battle),
                "battle_tag": getattr(battle, "battle_tag", None),
                "module": getattr(battle, "__module__", None),
                "class": battle.__class__.__name__,
                "turn": getattr(battle, "turn", None),
                "in_team_preview": getattr(battle, "in_team_preview", None),
                "teampreview": getattr(battle, "_teampreview", None),
                "player_role": getattr(battle, "player_role", None),
                "opponent_role": getattr(battle, "opponent_role", None),
                "force_switch": getattr(battle, "force_switch", None),
                "force_switch_type": type(getattr(battle, "force_switch", None)).__name__,
                "reviving": getattr(battle, "reviving", None),
                "can_tera": getattr(battle, "can_tera", None) is not None,
                "has_teampreview_attr": hasattr(battle, "teampreview_opponent_team"),
                "has__teampreview_attr": hasattr(battle, "_teampreview_opponent_team"),
                "available_moves_len": len(getattr(battle, "available_moves", []) or []),
                "available_switches_len": len(getattr(battle, "available_switches", []) or []),
                "active": _mon(act),
                "opponent": _mon(opp),
            }
            try:
                sw = getattr(battle, "available_switches", []) or []
                snap["switches"] = [getattr(p, "species", None) for p in sw][:6]
            except Exception:
                pass
            return snap
        except Exception:
            return {"error": "snapshot_failed"}

    def _log_observation_debug(self, ustate: "UniversalState") -> None:
        """Attempt to compute and log base/text/tokenization stats after an obs failure."""
        if not self._debug:
            return
        try:
            obs_space = getattr(self, "_obs_space", None)
            if obs_space is None:
                return
            base = getattr(obs_space, "base_obs_space", None)
            if base is None:
                return
            base_obs = base.state_to_obs(ustate)
            keys = list(base_obs.keys())
            shapes = {k: (None if not hasattr(v, "shape") else tuple(getattr(v, "shape"))) for k, v in base_obs.items()}
            self._dbg(f"base_obs keys={keys} shapes={shapes}")
            # numbers length
            try:
                nums = base_obs.get("numbers")
                if nums is not None:
                    self._dbg(f"base numbers len={int(nums.shape[-1])} sample={np.asarray(nums)[...,:8].tolist() if np.ndim(nums)>0 else 'NA'}")
            except Exception:
                pass
            # text/tokenization
            try:
                if "text" in base_obs:
                    text = base_obs["text"]
                    try:
                        s = text.item() if hasattr(text, "item") else str(text)
                    except Exception:
                        s = str(text)
                    words = s.split()
                    self._dbg(f"base text words={len(words)} head={' '.join(words[:16])}")
                    tok_len = None
                    try:
                        tok_len = base.tokenizable.get("text")
                    except Exception:
                        pass
                    tokenizer = getattr(obs_space, "tokenizer", None)
                    if tokenizer is not None:
                        try:
                            t = tokenizer.tokenize(np.array(s, dtype=np.str_).tolist())
                            t = np.asarray(t)
                            unk = int(np.sum(t == getattr(__import__('metamon.tokenizer', fromlist=['UNKNOWN_TOKEN']), 'UNKNOWN_TOKEN', -1)))
                            self._dbg(f"tokenized len={int(t.shape[-1])} expected={tok_len} unk={unk} head={t[:16].tolist()}")
                        except Exception as e:
                            self._dbg(f"tokenizer.tokenize failed: {e}")
            except Exception:
                pass
        except Exception:
            pass

    # -------- Model loading --------
    def _ensure_model(self):
        if self._experiment is not None:
            return
        if torch is None:
            raise RuntimeError("PyTorch not available for MetamonBridge model load")
        # Load pretrained model + experiment (no training)
        pretrained = get_pretrained_model(self._model_name)
        # Keep a reference to the pretrained wrapper — it owns the observation space
        self._pretrained = pretrained  # type: ignore[attr-defined]
        self._experiment = pretrained.initialize_agent(checkpoint=None, log=False)
        # Unwrapped policy (AMAGO agent)
        self._agent = self._experiment.policy
        self._device = getattr(self._experiment, "DEVICE", torch.device("cpu"))
        # Ensure inference mode for policy and submodules (disables dropout and enables KV-cache path)
        try:
            self._agent.eval()
        except Exception:
            pass
        # Use the pretrained model's tokenized observation space
        self._obs_space = pretrained.observation_space
        # Infer rl2 width from tstep encoder (robust to architectural changes)
        extra = getattr(self._agent.tstep_encoder, "extra_emb", None)
        self._rl2_width = int(getattr(extra, "in_features", getattr(self._agent.tstep_encoder, "rl2_width", 1)))
        # One-time model summary for debugging
        try:
            base_n = int(self._obs_space.gym_space["numbers"].shape[0])
            extra = getattr(self._agent.tstep_encoder, "extra_emb", None)
            extra_dim = int(getattr(extra, "out_features", 0))
            self._dbg(
                f"model ready: obs_base_numbers={base_n} extra_emb={extra_dim} traj_encoder={type(self._agent.traj_encoder).__name__}"
            )
        except Exception:
            pass
        # Additional environment summary
        if self._debug:
            try:
                dev = str(self._device)
                has_cuda = False
                try:
                    has_cuda = bool(torch.cuda.is_available())
                except Exception:
                    pass
                try:
                    space = self._obs_space
                    cls = type(space).__name__
                    base_cls = type(getattr(space, 'base_obs_space', None)).__name__ if hasattr(space, 'base_obs_space') else None
                    gym_space = getattr(space, "gym_space", None)
                    if gym_space is not None and hasattr(gym_space, "spaces"):
                        gs = {k: (type(v).__name__, getattr(v, "shape", None)) for k, v in gym_space.spaces.items()}
                    else:
                        gs = "<unknown>"
                    tok_len = None
                    vocab = None
                    # TokenizedObservationSpace exposes tokenizer and base_obs_space
                    if hasattr(space, "base_obs_space"):
                        tok_len = getattr(space.base_obs_space, "tokenizable", {}).get("text")
                    if hasattr(space, "tokenizer"):
                        try:
                            vocab = len(space.tokenizer)
                        except Exception:
                            vocab = "<unknown>"
                    self._dbg(f"device={dev} cuda={has_cuda} obs_space={cls} base_obs={base_cls} gym={gs} token_len={tok_len} vocab={vocab}")
                except Exception:
                    pass
            except Exception:
                pass

    # -------- Hook installation --------
    def install_hooks(self):
        """Monkey‑patch both pip poke_env and local fork battle classes.

        Safe to call multiple times; installs once per target family.
        """

        def _iter_targets():
            # pip-installed poke_env
            try:
                from poke_env.environment.abstract_battle import AbstractBattle as AB1  # type: ignore
                from poke_env.environment.battle import Battle as B1  # type: ignore
                yield ("pip", AB1, B1)
            except Exception:
                pass
            # local fork used by Pokéchamp agent / LocalSim
            try:
                from pokechamp.poke_env_local.environment.abstract_battle import (
                    AbstractBattle as AB2,  # type: ignore
                )
                from pokechamp.poke_env_local.environment.battle import (
                    Battle as B2,  # type: ignore
                )
                yield ("local", AB2, B2)
            except Exception:
                pass

        for tag, _AB, _B in _iter_targets():
            if self._patched.get(tag):
                continue

            self._orig_parse_message[tag] = _AB.parse_message

            def _wrapped_parse_message(this, split_message, _orig=self._orig_parse_message[tag]):  # type: ignore
                # Run original first to keep engine behavior authoritative
                _orig(this, split_message)
                try:
                    self.on_message(this, split_message, source=tag)
                except Exception as e:
                    self._dbg(f"{tag}.on_message error: {e}")

            _AB.parse_message = _wrapped_parse_message  # type: ignore

            self._orig_parse_request[tag] = _B.parse_request

            def _wrapped_parse_request(this, request, _orig=self._orig_parse_request[tag]):  # type: ignore
                _orig(this, request)
                try:
                    self.on_request(this, request, source=tag)
                except Exception as e:
                    self._dbg(f"{tag}.on_request error: {e}")

            _B.parse_request = _wrapped_parse_request  # type: ignore

            self._patched[tag] = True

        self._installed = True

    # -------- Readiness / validation --------
    def is_ready(self) -> tuple[bool, str]:
        """Return (ready, reason) indicating whether the bridge can score.

        Checks model, device, and observation readiness for critic-based scoring.
        """
        try:
            if self._experiment is None or self._agent is None:
                return False, "model_not_loaded"
            if self._obs_space is None or self._rl2_width is None:
                return False, "observation_space_not_ready"
            return True, "ok"
        except Exception as e:
            return False, f"exception:{e}"

    # -------- Context presence helper --------
    def has_context(self, battle) -> bool:
        try:
            return id(battle) in self._contexts
        except Exception:
            return False

    # -------- Context management --------
    def _infer_gen_from_btag(self, battle_tag: str) -> int:
        # battle tag patterns commonly look like "battle-gen9ou-..."
        try:
            parts = battle_tag.split("-")
            fmt = parts[1] if len(parts) > 1 else battle_tag
            if fmt.startswith("gen"):
                rest = fmt[3:]
                m = re.match(r"(\d+)", rest)
                if m:
                    return int(m.group(1))
        except Exception:
            pass
        # default to Gen 9 for our exclusive focus
        return 9

    def ensure_registered(self, battle) -> None:
        with self._lock:
            key = id(battle)
            if key in self._contexts:
                return
            self._ensure_model()

            gen = self._infer_gen_from_btag(getattr(battle, "battle_tag", "battle-gen9ou"))
            username = getattr(battle, "player_username", getattr(battle, "username", "player"))
            mm_battle = MetamonBackendBattle(
                battle_tag=getattr(battle, "battle_tag", "battle-gen9ou"),
                username=username,
                logger=getattr(battle, "logger", None),
                save_replays=False,
                gen=gen,
            )
            # Silence backend debug dumps regardless of env
            try:
                setattr(mm_battle, "_debug", False)
            except Exception:
                pass
            # Try to capture player role from local battle if available
            try:
                role = getattr(battle, "_player_role", None)
                if role in ("p1", "p2"):
                    setattr(mm_battle, "player_role", role)
            except Exception:
                pass
            hidden = self._agent.traj_encoder.init_hidden_state(1, self._device)
            self._contexts[key] = _Ctx(mm_battle=mm_battle, hidden=hidden, time_idx=0, ephemeral=False)
            try:
                if self._debug or str(os.getenv("POKECHAMP_MINIMAX_LOG", "0")).lower() in {"1","true","t","yes","y"}:
                    print(f"[MetamonBridge] ensure_registered: created ctx id={key} ephemeral=False role={getattr(mm_battle,'player_role',None)}", flush=True)
            except Exception:
                pass

    def clone_context(self, parent_battle, child_battle) -> None:
        with self._lock:
            pkey = id(parent_battle)
            if pkey not in self._contexts:
                # parent might not be registered if Metamon disabled
                return
            ckey = id(child_battle)
            if ckey in self._contexts:
                return
            parent = self._contexts[pkey]
            # Pre-clone diag: length of parent's replay (turnlist)
            try:
                preplay = getattr(parent.mm_battle, "_mm_battle", None)
                parent_turns = len(getattr(preplay, "turnlist", []) or [])
            except Exception:
                parent_turns = 0
            gen = self._infer_gen_from_btag(
                getattr(child_battle, "battle_tag", getattr(parent.mm_battle, "battle_tag", "battle-gen9ou"))
            )
            username = getattr(child_battle, "player_username", getattr(parent.mm_battle, "player_username", "player"))
            child_mm = MetamonBackendBattle(
                battle_tag=getattr(child_battle, "battle_tag", parent.mm_battle.battle_tag),
                username=username,
                logger=getattr(child_battle, "logger", None),
                save_replays=False,
                gen=gen,
            )
            # Silence backend debug dumps regardless of env
            try:
                setattr(child_mm, "_debug", False)
            except Exception:
                pass
            # Deep copy ParsedReplay into the new backend and reset protocol
            try:
                child_mm._mm_battle = copy.deepcopy(parent.mm_battle._mm_battle)
                # Use the installed metamon package path
                try:
                    from metamon.backend.replay_parser import forward as _forward  # type: ignore
                except Exception:
                    # Fallback to older nested path if present
                    from metamon.metamon.backend.replay_parser import forward as _forward  # type: ignore
                child_mm._sim_protocol = _forward.SimProtocol(child_mm._mm_battle)
                # Propagate player role explicitly to the child backend
                try:
                    parent_role = getattr(parent.mm_battle, "player_role", None)
                    if parent_role in ("p1", "p2"):
                        setattr(child_mm, "player_role", parent_role)
                except Exception:
                    pass
                # Best-effort backfill of current-turn active HP from parent into child
                try:
                    p_replay = getattr(parent.mm_battle, "_mm_battle", None)
                    c_replay = getattr(child_mm, "_mm_battle", None)
                    if (
                        p_replay is not None
                        and c_replay is not None
                        and getattr(p_replay, "turnlist", None)
                        and getattr(c_replay, "turnlist", None)
                    ):
                        p_turn = p_replay.turnlist[-1]
                        c_turn = c_replay.turnlist[-1]
                        # helper: copy hp if known
                        def _copy_hp(src_mon, dst_mon):
                            try:
                                if src_mon is None or dst_mon is None:
                                    return
                                chp = getattr(src_mon, "current_hp", None)
                                mhp = getattr(src_mon, "max_hp", None)
                                if chp is not None:
                                    setattr(dst_mon, "current_hp", chp)
                                if mhp is not None:
                                    setattr(dst_mon, "max_hp", mhp)
                            except Exception:
                                pass
                        # Both sides actives
                        try:
                            p1_src = p_turn.get_active_pokemon(True)[0]
                            p2_src = p_turn.get_active_pokemon(False)[0]
                        except Exception:
                            p1_src = None
                            p2_src = None
                        try:
                            p1_dst = c_turn.get_active_pokemon(True)[0]
                            p2_dst = c_turn.get_active_pokemon(False)[0]
                        except Exception:
                            p1_dst = None
                            p2_dst = None
                        _copy_hp(p1_src, p1_dst)
                        _copy_hp(p2_src, p2_dst)
                except Exception as e:
                    self._dbg(f"clone_context hp backfill failed: {e}")
            except Exception as e:
                self._dbg(f"clone_context deep copy failed: {e}")

            hidden = self._clone_hidden(parent.hidden) if torch is not None else None
            ctx = _Ctx(mm_battle=child_mm, hidden=hidden, time_idx=parent.time_idx, ephemeral=True)
            try:
                # Cache role on context as well
                ctx.player_role = getattr(parent, "player_role", None)
                if ctx.player_role is None:
                    ctx.player_role = getattr(parent.mm_battle, "player_role", None)
            except Exception:
                pass
            self._contexts[ckey] = ctx
            # Clone normalizer state so LocalSim branches inherit roster info
            try:
                self._normalizer.clone_state(parent_battle, child_battle)
            except Exception:
                pass
            # Debug: log clone mapping and parent replay size
            try:
                if self._debug or str(os.getenv("POKECHAMP_MINIMAX_LOG", "0")).lower() in {"1","true","t","yes","y"}:
                    print(f"[MetamonBridge] clone_context parent_id={id(parent_battle)} -> child_id={id(child_battle)} turns={parent_turns}", flush=True)
            except Exception:
                pass

    def sync_context(self, dst_battle, src_battle) -> None:
        """Refresh an existing destination context from a source (live) context.

        Copies the parsed replay and resets the sim protocol. Propagates player_role
        and aligns hidden/time_idx for trajectory continuity.
        """
        with self._lock:
            dkey = id(dst_battle)
            skey = id(src_battle)
            if skey not in self._contexts:
                # Ensure live context exists
                try:
                    self.ensure_registered(src_battle)
                except Exception as e:
                    self._dbg(f"sync_context ensure live failed: {e}")
                    return
            if dkey not in self._contexts:
                return
            dctx = self._contexts[dkey]
            sctx = self._contexts[skey]
            try:
                # Replace parsed replay
                dctx.mm_battle._mm_battle = copy.deepcopy(sctx.mm_battle._mm_battle)
                # Reset protocol on destination
                try:
                    from metamon.backend.replay_parser import forward as _forward  # type: ignore
                except Exception:
                    from metamon.metamon.backend.replay_parser import forward as _forward  # type: ignore
                dctx.mm_battle._sim_protocol = _forward.SimProtocol(dctx.mm_battle._mm_battle)
                # Role propagation
                try:
                    role = getattr(sctx.mm_battle, "player_role", None)
                    if role in ("p1", "p2"):
                        setattr(dctx.mm_battle, "player_role", role)
                        dctx.player_role = role
                except Exception:
                    pass
                # Align hidden state and time index
                try:
                    if torch is not None:
                        dctx.hidden = self._clone_hidden(sctx.hidden)
                    dctx.time_idx = sctx.time_idx
                except Exception:
                    pass
                # Optional debug: report refresh
                try:
                    if self._debug or str(os.getenv("POKECHAMP_MINIMAX_LOG", "0")).lower() in {"1","true","t","yes","y"}:
                        turns = 0
                        try:
                            turns = len(getattr(dctx.mm_battle._mm_battle, "turnlist", []) or [])
                        except Exception:
                            turns = 0
                        print(
                            f"[MetamonBridge] sync_context src_id={id(src_battle)} -> dst_id={id(dst_battle)} turns={turns}",
                            flush=True,
                        )
                except Exception:
                    pass
            except Exception as e:
                self._dbg(f"sync_context failed: {e}")

    def _clone_hidden(self, hidden):
        if hidden is None:
            return None
        try:
            if isinstance(hidden, torch.Tensor):
                return hidden.clone()
            if isinstance(hidden, (list, tuple)):
                cloned = [self._clone_hidden(h) for h in hidden]
                return type(hidden)(cloned) if isinstance(hidden, tuple) else cloned
            if isinstance(hidden, dict):
                return {k: self._clone_hidden(v) for k, v in hidden.items()}
        except Exception:
            pass
        return copy.deepcopy(hidden)

    def release_ephemeral(self):
        with self._lock:
            to_delete = [k for k, v in self._contexts.items() if v.ephemeral]
            for k in to_delete:
                del self._contexts[k]

    def cleanup(self, battle):
        with self._lock:
            key = id(battle)
            if key in self._contexts:
                del self._contexts[key]
            try:
                self._normalizer.cleanup(battle)
            except Exception:
                pass

    # -------- Message/request mirroring --------
    def on_message(self, battle, split_message, source: Optional[str] = None):
        with self._lock:
            key = id(battle)
            ctx = self._contexts.get(key)
            if ctx is None:
                # Opportunistically register if possible
                try:
                    self.ensure_registered(battle)
                    ctx = self._contexts.get(key)
                except Exception as e:
                    self._dbg(f"ensure_registered (message) failed: {e}")
                    return
            # Forward to Metamon sim protocol
            try:
                msg = split_message
                # Normalize LocalSim messages before forwarding
                if source == "local":
                    try:
                        msg = self._normalizer.normalize(battle, split_message)
                    except Exception as e:
                        self._dbg(f"normalizer error: {e}")
                    # Seed missing actives for the current side if a move arrives before a switch
                    try:
                        if len(msg) > 3 and msg[1] == "move":
                            # Determine side from ident in msg[2] ('p1a: Name')
                            ident = str(msg[2]) if msg[2] else ""
                            side = ident.split(":")[0].strip()[:2] if ":" in ident else ""
                            if side in ("p1", "p2"):
                                replay = getattr(ctx.mm_battle, "_mm_battle", None)
                                if replay is not None and getattr(replay, "turnlist", None):
                                    turn = replay.turnlist[-1]
                                    missing = False
                                    if side == "p1":
                                        missing = not bool(getattr(turn, "active_pokemon_1", []) )
                                    else:
                                        missing = not bool(getattr(turn, "active_pokemon_2", []) )
                                    if missing:
                                        # Reconstruct details and emit a synthetic switch before the move
                                        try:
                                            name = ident.split(":",1)[1].strip()
                                        except Exception:
                                            name = ""
                                        details = self._normalizer.reconstruct_details(battle, side, name)
                                        seed = ["", "switch", ident if ident else f"{side}a: {name}", details, "100/100"]
                                        ctx.mm_battle.parse_message(seed)
                    except Exception as e:
                        self._dbg(f"seed presence error: {e}")
                ctx.mm_battle.parse_message(msg)
            except Exception as e:
                # Swallow parser exceptions to avoid affecting the engine
                self._dbg(f"parse_message error: {e} :: {split_message}")
            # Advance hidden on simulated branches at turn boundaries
            try:
                if ctx.ephemeral and torch is not None and len(split_message) > 1:
                    if split_message[1] == "turn":
                        self._advance_hidden(ctx)
                # Auto-cleanup when battle ends
                if len(split_message) > 1 and split_message[1] in {"win", "tie"}:
                    # Also remove ephemeral children on terminal events
                    self.cleanup(battle)
            except Exception as e:
                self._dbg(f"on_message post-process error: {e}")

    def _to_torch_batch(self, obs_np: Dict[str, np.ndarray], *, device) -> Dict[str, "torch.Tensor"]:
        t_obs: Dict[str, "torch.Tensor"] = {}
        expected_n: Optional[int] = None
        extra_dim: Optional[int] = None

        # Infer model expectations from the tstep encoder layers
        try:
            extra = getattr(self._agent.tstep_encoder, "extra_emb", None)
            extra_dim = int(getattr(extra, "out_features", 0))
            turn_emb = getattr(self._agent.tstep_encoder, "turn_embedding", None)
            multimodal = getattr(turn_emb, "multimodal_fuse", None)
            num_emb = getattr(multimodal, "num_emb", None)
            if num_emb is not None:
                expected_n = int(num_emb.in_features) - int(extra_dim or 0)
        except Exception:
            expected_n = None

        # Fallback to observation space declaration
        if expected_n is None:
            try:
                expected_n = int(self._obs_space.gym_space["numbers"].shape[0])
            except Exception:
                expected_n = None

        # Removed noisy per-leaf logging of obs keys/shapes to keep output concise

        for k, v in obs_np.items():
            if k == "numbers" and expected_n is not None:
                try:
                    current_n = int(v.shape[-1])
                    if current_n != expected_n:
                        if current_n > expected_n:
                            v = v[..., :expected_n]
                            self._dbg(
                                f"normalize numbers dims: slice {current_n}->{expected_n} (extra_emb={extra_dim})"
                            )
                        else:
                            pad_width = expected_n - current_n
                            pad = np.full((pad_width,), -2.0, dtype=v.dtype)
                            v = np.concatenate([v, pad], axis=-1)
                            self._dbg(
                                f"normalize numbers dims: pad {current_n}->{expected_n} (extra_emb={extra_dim})"
                            )
                except Exception as e:
                    self._dbg(f"failed to normalize numbers dims: {e}")

            if k not in {"numbers", "text_tokens"}:
                continue

            # Ensure dtypes are correct and add [B,T] dims
            if k == "numbers":
                np_arr = v.astype(np.float32, copy=False)
            else:  # text_tokens
                np_arr = v.astype(np.int32, copy=False)

            tv = torch.from_numpy(np_arr)
            t_obs[k] = tv.to(device).unsqueeze(0).unsqueeze(1)
        return t_obs


    def _advance_hidden(self, ctx: _Ctx) -> None:
        # Build observation (tolerant: replay-state if complete, else live battle)
        obs_np = self._build_obs_np(ctx)
        if obs_np is None:
            return
        obs = self._to_torch_batch(obs_np, device=self._device)

        # rl2s/times for one step
        rl2s = torch.zeros((1, 1, int(self._rl2_width)), dtype=torch.float32, device=self._device)
        # Transformer expects pos indices with shape [B, L, 1]; add last dim
        time_idxs = torch.tensor([[ctx.time_idx]], dtype=torch.long, device=self._device).unsqueeze(-1)

        with torch.no_grad():
            tstep_emb = self._agent.tstep_encoder(obs=obs, rl2s=rl2s)
            ret = self._agent.traj_encoder(seq=tstep_emb, time_idxs=time_idxs, hidden_state=ctx.hidden)
            if isinstance(ret, tuple):
                # RNN-style encoders return (emb, hidden)
                _s_rep, new_hidden = ret
                ctx.hidden = new_hidden
            else:
                # Transformer-style encoders return only the embedding sequence
                ctx.hidden = ctx.hidden
        ctx.time_idx += 1

    def on_request(self, battle, request_json, source: Optional[str] = None):
        with self._lock:
            key = id(battle)
            ctx = self._contexts.get(key)
            if ctx is None:
                # Opportunistically register if possible
                try:
                    self.ensure_registered(battle)
                    ctx = self._contexts.get(key)
                except Exception as e:
                    self._dbg(f"ensure_registered (request) failed: {e}")
                    return
            try:
                ctx.mm_battle.parse_request(request_json)
            except Exception as e:
                self._dbg(f"parse_request error: {e}")
            # Encode this turn into persistent hidden state (live context only)
            try:
                if torch is not None and not ctx.ephemeral:
                    self._advance_hidden(ctx)
                    # Instrumentation (periodic health logging) removed
            except Exception as e:
                self._dbg_exc("advance_hidden on_request error", e)

    # -------- Scoring --------
    def score(self, battle) -> Optional[float]:
        """Return scalar critic-based value for the given battle branch.

        Uses E_{a ~ pi}[Q(s, a)] over the legal discrete actions at test-time gamma.
        Returns None if the trajectory context or observation is not ready.
        """
        self._ensure_model()
        if torch is None:
            return None
        ctx = self._contexts.get(id(battle))
        if ctx is None:
            return None
        obs_np = self._build_obs_np(ctx)
        if obs_np is None:
            return None
        # Debug: basic state + obs source
        try:
            role = getattr(ctx, "player_role", None) or getattr(ctx.mm_battle, "player_role", None)
            rep = getattr(ctx.mm_battle, "_mm_battle", None)
            turn = rep.turnlist[-1] if (rep is not None and getattr(rep, "turnlist", None)) else None
            act = turn.get_active_pokemon(True)[0] if turn is not None else None
            opp = turn.get_active_pokemon(False)[0] if turn is not None else None
            def _nmhp(p):
                try:
                    return (getattr(p, "name", None) or getattr(p, "had_name", None) or "?",
                            getattr(p, "current_hp", None), getattr(p, "max_hp", None))
                except Exception:
                    return ("?", None, None)
            an, ach, amh = _nmhp(act)
            on, och, omh = _nmhp(opp)
            print(f"[MetamonScoreDBG] battle_id={id(battle)} turn_idx={ctx.time_idx} role={role} obs_src={ctx.last_obs_src} act=({an},{ach}/{amh}) opp=({on},{och}/{omh})", flush=True)
        except Exception:
            pass
        # Build torch batch
        obs = self._to_torch_batch(obs_np, device=self._device)
        rl2s = torch.zeros((1, 1, int(self._rl2_width)), dtype=torch.float32, device=self._device)
        time_idxs = torch.tensor([[ctx.time_idx]], dtype=torch.long, device=self._device).unsqueeze(-1)

        with torch.no_grad():
            tstep_emb = self._agent.tstep_encoder(obs=obs, rl2s=rl2s)
            ret = self._agent.traj_encoder(seq=tstep_emb, time_idxs=time_idxs, hidden_state=ctx.hidden)
            s_rep = ret[0] if isinstance(ret, tuple) else ret  # [1,1,D]
            # Actor probabilities via base + manual reshape to avoid actor head crash
            # (see policy_top_n_traj for rationale). Base output is [B,L,G*D].
            head = self._agent.actor
            dp0 = head.base(s_rep)
            g = int(getattr(head, 'num_gammas', 0))
            d = int(getattr(getattr(head, 'policy_dist', None), 'input_dimension', 0))
            if g <= 0 or d <= 0 or int(dp0.shape[-1]) != g * d:
                return None
            dp = dp0.contiguous().view(1, 1, g, d)
            pvec = torch.softmax(dp, dim=-1)[0, 0, -1, :]
            # Log actor stats for test-time gamma
            try:
                ps = pvec.detach().cpu().numpy()
                topk = int(min(5, ps.shape[0]))
                idxs = np.argsort(-ps)[:topk]
                vals = [float(ps[i]) for i in idxs]
                print(f"[MetamonScoreDBG] actor_g* top_idx={idxs.tolist()} top_p={[round(v,4) for v in vals]} p_sum={float(ps.sum()):.4f}", flush=True)
            except Exception:
                pass

        # Determine legal indices from the current replay snapshot
        try:
            from metamon.backend.replay_parser.replay_state import ReplayState as _ReplayState
            mm = ctx.mm_battle
            replay = getattr(mm, "_mm_battle", None)
            if replay is None or not getattr(replay, "turnlist", None):
                return None
            turn = replay.turnlist[-1]
            role = getattr(ctx, "player_role", None) or getattr(mm, "player_role", None)
            p1 = (role == "p1")
            # Basic active/opponent for safety
            active_list = turn.active_pokemon_1 if p1 else turn.active_pokemon_2
            opp_active_list = turn.active_pokemon_2 if p1 else turn.active_pokemon_1
            active_mon = active_list[0] if active_list else None
            opp_active_mon = opp_active_list[0] if opp_active_list else None
            if active_mon is None or opp_active_mon is None:
                return None
            # Build UniversalState and legal set
            state = _ReplayState(
                format=replay.format or "gen9ou",
                force_switch=bool(getattr(turn, "is_force_switch", False)),
                active_pokemon=active_mon,
                opponent_active_pokemon=opp_active_mon,
                opponent_team=(turn.pokemon_2 if p1 else turn.pokemon_1),
                available_switches=(turn.available_switches_1 if p1 else turn.available_switches_2) or [],
                player_prev_move=getattr(active_mon, "last_used_move", None),
                opponent_prev_move=getattr(opp_active_mon, "last_used_move", None),
                player_conditions=(turn.conditions_1 if p1 else turn.conditions_2),
                opponent_conditions=(turn.conditions_2 if p1 else turn.conditions_1),
                weather=getattr(turn, "weather", None),
                battle_field=getattr(turn, "battle_field", None),
                battle_won=bool(getattr(mm, "won", False)),
                battle_lost=bool(getattr(mm, "lost", False)),
                can_tera=(turn.can_tera_1 if p1 else turn.can_tera_2),
                opponent_teampreview=(turn.teampreview_2 if p1 else turn.teampreview_1),
            )
            src_ustate = 'replay'
            try:
                ustate = UniversalState.from_ReplayState(state)
            except Exception as e_rs:
                try:
                    ustate = UniversalState.from_Battle(ctx.mm_battle)
                    src_ustate = 'battle'
                    try:
                        et = type(e_rs).__name__
                        em = str(e_rs)
                        print(f"[MetamonScoreDBG] legality fallback: from_ReplayState failed ({et}: {em}); using from_Battle", flush=True)
                    except Exception:
                        pass
                except Exception:
                    return None
            from metamon.interface import UniversalAction
            # Maybe-legal indices (state-only)
            legal_set = UniversalAction.maybe_valid_actions(ustate)
            legal_indices = sorted([int(a.action_idx) for a in legal_set])
            # Definitely-legal indices via local mapping to Pokechamp battle orders
            def_indices = [i for i in legal_indices if self._action_idx_to_order_local(battle, int(i)) is not None]
            if not legal_indices:
                return None
            # Tera gating summary
            try:
                can_tera = bool(getattr(ustate, 'can_tera', False))
                tera_maybe = int(sum(1 for i in legal_indices if i >= 9))
                tera_def = int(sum(1 for i in def_indices if i >= 9))
                print(
                    f"[MetamonScoreDBG] legal maybe_n={len(legal_indices)} definite_n={len(def_indices)} can_tera={can_tera} tera_maybe={tera_maybe} tera_def={tera_def}",
                    flush=True,
                )
            except Exception:
                pass
            # Log action space + legality mapping
            try:
                D_action = int(getattr(self._agent, 'action_dim', 0))
                sample = legal_indices[:min(5, len(legal_indices))]
                mapped = []
                for j in sample:
                    try:
                        bo = self._action_idx_to_order_local(battle, int(j))
                        mapped.append(str(bo) if bo is not None else "None")
                    except Exception:
                        mapped.append("err")
                print(f"[MetamonScoreDBG] action_dim={D_action} legal_n={len(legal_indices)} sample_idx={sample} sample_orders={mapped}", flush=True)
            except Exception:
                pass
        except Exception:
            return None

        # Build one-hot actions for all discrete actions for unified Q lookup
        try:
            G = int(len(getattr(self._agent, 'gammas', [])) or 1)
            D_action = int(getattr(self._agent, 'action_dim', 0))
            if D_action <= 0:
                return None
            K_all = D_action
            a_all = torch.zeros((K_all, 1, 1, G, D_action), dtype=torch.float32, device=self._device)
            for k in range(D_action):
                a_all[k, 0, 0, :, k] = 1.0
            with torch.no_grad():
                bin_dist_all = self._agent.critics(s_rep, a_all)
                q_raw_all = self._agent.critics.bin_dist_to_raw_vals(bin_dist_all)  # [K_all,1,1,C,G,1]
            q_all = q_raw_all[:, 0, 0, :, -1, 0]  # [K_all, C]
            q_mean_all = q_all.mean(dim=1)  # [K_all]
            # Log critic stats across legal actions
            try:
                qm = q_mean_all.detach().cpu().numpy()[legal_indices]
                qmin = float(np.min(qm)) if qm.size else float('nan')
                qmax = float(np.max(qm)) if qm.size else float('nan')
                qmed = float(np.median(qm)) if qm.size else float('nan')
                print(f"[MetamonScoreDBG] critics q_mean stats min={qmin:.3f} med={qmed:.3f} max={qmax:.3f}", flush=True)
            except Exception:
                pass
            # Expected Q over maybe-legal (legacy behavior)
            idx_maybe = [i for i in legal_indices if 0 <= i < D_action]
            p_maybe = pvec[idx_maybe]
            p_maybe_sum = float(p_maybe.sum().item())
            if p_maybe_sum <= 1e-8:
                w_maybe = torch.full_like(p_maybe, 1.0 / max(1, len(idx_maybe)))
            else:
                w_maybe = p_maybe / p_maybe.sum()
            v_maybe = float((w_maybe * q_mean_all[idx_maybe]).sum().item()) if idx_maybe else float('nan')
            # Expected Q over definitely-legal
            idx_def = [i for i in def_indices if 0 <= i < D_action]
            p_def = pvec[idx_def]
            p_def_sum = float(p_def.sum().item())
            if p_def_sum <= 1e-8:
                w_def = torch.full_like(p_def, 1.0 / max(1, len(idx_def)))
            else:
                w_def = p_def / p_def.sum()
            v_def = float((w_def * q_mean_all[idx_def]).sum().item()) if idx_def else float('nan')
            # As a last resort, compute over all actions (masked by [0..D_action))
            idx_all = list(range(D_action))
            p_all = pvec[idx_all]
            p_all_sum = float(p_all.sum().item())
            if p_all_sum <= 1e-8:
                w_all = torch.full_like(p_all, 1.0 / max(1, len(idx_all)))
            else:
                w_all = p_all / p_all.sum()
            v_all = float((w_all * q_mean_all[idx_all]).sum().item()) if idx_all else float('nan')
            try:
                print(f"[MetamonScoreDBG] value maybe={v_maybe:.3f} (p_sum={p_maybe_sum:.4f}) | definite={v_def:.3f} (p_sum={p_def_sum:.4f})", flush=True)
            except Exception:
                pass
            # Use definitely-legal value; fallback to maybe, else to all-actions expectation
            if idx_def:
                return v_def
            if idx_maybe:
                return v_maybe
            return v_all
        except Exception:
            return None

    # -------- Action mapping (local battle) --------
    # IMPORTANT: Pokechamp uses a local forked poke_env (pokechamp.poke_env_local)
    # while Metamon’s code expects its own poke_env data structures. Do NOT call
    # Metamon helpers like action_idx_to_BattleOrder or definitely_valid_actions
    # with a local Battle instance — types and semantics differ and will break.
    # Always map indices to local BattleOrders via this helper using:
    #   - consistent_move_order for moves (by id strings)
    #   - consistent_pokemon_order for bench species
    # and validate legality against battle.available_moves/switches.
    def _action_idx_to_order_local(self, battle, action_idx: int) -> Optional[LocalBattleOrder]:
        """Map a Metamon discrete action index to a local BattleOrder.

        Follows the same indexing convention as metamon.interface.UniversalAction:
        - 0..3: moves (sorted by consistent_move_order over move ids)
        - 4..8: switches (sorted by consistent_pokemon_order over species names)
        - >=9: Tera variant of move indices (only valid if can_tera)
        Returns None if the index does not correspond to a currently legal action.
        """
        try:
            # Moves legality
            valid_moves = {getattr(m, 'id', None) for m in (battle.available_moves or [])}
            valid_moves.discard(None)
            # Handle recharge/struggle short-circuits
            if valid_moves == {"recharge"}:
                return LocalPlayer.create_order(battle.available_moves[0])
            if valid_moves == {"struggle"}:
                move_objs = [battle.available_moves[0]] * 4
            else:
                # Build canonical move order from mirrored replay state to align with Metamon indices
                move_objs = []
                try:
                    ctx = self._contexts.get(id(battle))
                    rs_names: list[str] = []
                    if ctx is not None:
                        from metamon.backend.replay_parser.replay_state import ReplayState as _ReplayState  # type: ignore
                        mm = ctx.mm_battle
                        replay = getattr(mm, "_mm_battle", None)
                        if replay is not None and getattr(replay, "turnlist", None):
                            turn = replay.turnlist[-1]
                            role = getattr(ctx, "player_role", None) or getattr(mm, "player_role", None)
                            p1 = (role == "p1")
                            state = _ReplayState(
                                format=replay.format or "gen9ou",
                                force_switch=bool(getattr(turn, "is_force_switch", False)),
                                active_pokemon=(turn.active_pokemon_1 if p1 else turn.active_pokemon_2)[0] if ((turn.active_pokemon_1 if p1 else turn.active_pokemon_2)) else None,
                                opponent_active_pokemon=(turn.active_pokemon_2 if p1 else turn.active_pokemon_1)[0] if ((turn.active_pokemon_2 if p1 else turn.active_pokemon_1)) else None,
                                opponent_team=(turn.pokemon_2 if p1 else turn.pokemon_1),
                                available_switches=(turn.available_switches_1 if p1 else turn.available_switches_2) or [],
                                player_prev_move=getattr(((turn.active_pokemon_1 if p1 else turn.active_pokemon_2) or [None])[0], "last_used_move", None),
                                opponent_prev_move=getattr(((turn.active_pokemon_2 if p1 else turn.active_pokemon_1) or [None])[0], "last_used_move", None),
                                player_conditions=(turn.conditions_1 if p1 else turn.conditions_2),
                                opponent_conditions=(turn.conditions_2 if p1 else turn.conditions_1),
                                weather=getattr(turn, "weather", None),
                                battle_field=getattr(turn, "battle_field", None),
                                battle_won=bool(getattr(mm, "won", False)),
                                battle_lost=bool(getattr(mm, "lost", False)),
                                can_tera=(turn.can_tera_1 if p1 else turn.can_tera_2),
                                opponent_teampreview=(turn.teampreview_2 if p1 else turn.teampreview_1),
                            )
                            ustate = UniversalState.from_ReplayState(state)
                            rs_names = [getattr(m, 'name', '') for m in ustate.player_active_pokemon.moves]
                    # Canonical move names as used by obs/actor
                    canonical = list(consistent_move_order(rs_names)) if rs_names else []
                    # Map canonical names to local Move objects by id
                    move_map = {getattr(m, 'id'): m for m in (getattr(battle.active_pokemon, 'moves', {}) or {}).values()}
                    # Preserve 4 slots; None when missing locally
                    move_objs = [(move_map.get(name) if name in move_map else None) for name in canonical]
                    # If we couldn't build from replay (no ctx/turns), fall back to local canonical order
                    if not move_objs:
                        sorted_ids = list(consistent_move_order(list(move_map.keys()))) if move_map else []
                        move_objs = [move_map[mid] for mid in sorted_ids if mid in move_map]
                except Exception:
                    # Fallback to local order on any failure
                    move_map = {getattr(m, 'id'): m for m in (getattr(battle.active_pokemon, 'moves', {}) or {}).values()}
                    sorted_ids = list(consistent_move_order(list(move_map.keys()))) if move_map else []
                    move_objs = [move_map[mid] for mid in sorted_ids if mid in move_map]

            # Switch legality
            valid_switch_species = {getattr(p, 'species', None) for p in (battle.available_switches or [])}
            valid_switch_species.discard(None)
            if not getattr(battle, 'reviving', False):
                bench = [
                    p for p in list((battle.team or {}).values())
                    if not getattr(p, 'fainted', False) and not getattr(p, 'active', False)
                ]
            else:
                bench = [
                    p for p in list((battle.team or {}).values())
                    if getattr(p, 'fainted', False) and not getattr(p, 'active', False)
                ]
            # Sort switches by species name using consistent_pokemon_order on strings
            bench_species = [getattr(p, 'species', '') for p in bench]
            sorted_species = list(consistent_pokemon_order(bench_species)) if bench_species else []
            # Map sorted species back to Pokemon objects
            species_to_poke = {getattr(p, 'species', ''): p for p in bench}

            # Handle Tera offset
            wants_tera = False
            can_tera = getattr(battle, 'can_tera', None) is not None
            if action_idx >= 9:
                wants_tera = True
                action_idx -= 9

            # Moves (0..3), only if not forced to switch
            if action_idx <= 3 and not getattr(battle, 'force_switch', False):
                if action_idx < len(move_objs):
                    mv = move_objs[action_idx]
                    # mv may be None when replay has a move we can't find locally; treat as illegal mapping for now
                    if mv is not None and getattr(mv, 'id', None) in valid_moves:
                        return LocalPlayer.create_order(mv, terastallize=bool(wants_tera and can_tera))
                # Fallthrough: invalid/missing move or illegal under current constraints
                return None

            # Switches (4..8)
            if 4 <= action_idx <= 8:
                sidx = action_idx - 4
                if sidx < len(sorted_species):
                    sp = sorted_species[sidx]
                    if sp in valid_switch_species:
                        poke = species_to_poke.get(sp)
                        if poke is not None:
                            return LocalPlayer.create_order(poke)
                return None

            return None
        except Exception:
            return None

    # -------- Policy priors (trajectory) --------
    def policy_top_n_traj(self, battle, top_n: int = 3) -> list[tuple[int, float]]:
        """Return top-N (action_idx, prob) policy priors using trajectory mode.

        Returns an empty list when the mirrored context is missing or replay is incomplete.
        Filters to actions that can be mapped to valid local BattleOrders.
        """
        self._ensure_model()
        bid = id(battle)
        try:
            print(f"[MetamonPriorsDBG] enter policy_top_n_traj battle_id={bid} topN={top_n}", flush=True)
        except Exception:
            pass
        ctx = self._contexts.get(bid)
        if ctx is None:
            print(f"[MetamonPriorsDBG] no ctx for battle_id={bid}", flush=True)
            raise RuntimeError("MetamonBridge.policy_top_n_traj: no trajectory context found for battle; ensure contexts are registered/cloned before calling.")
        try:
            print(f"[MetamonPriorsDBG] ctx found: ephemeral={ctx.ephemeral} time_idx={ctx.time_idx}", flush=True)
        except Exception:
            pass
        obs_np = self._build_obs_np(ctx)
        if obs_np is None:
            print(f"[MetamonPriorsDBG] obs_np is None (incomplete replay snapshot) last_obs_src={ctx.last_obs_src}", flush=True)
            raise RuntimeError("MetamonBridge.policy_top_n_traj: observation incomplete for current replay snapshot; context not ready.")
        try:
            nums = obs_np.get("numbers")
            toks = obs_np.get("text_tokens")
            n_shape = tuple(nums.shape) if isinstance(nums, np.ndarray) else None
            t_nnz = int(np.count_nonzero(toks)) if isinstance(toks, np.ndarray) else None
            # Compute expected numbers length from obs space; flag mismatches
            exp_n = None
            try:
                exp_n = int(self._obs_space.gym_space["numbers"].shape[0])
            except Exception:
                exp_n = None
            mismatch = (exp_n is not None and nums is not None and int(nums.shape[-1]) != exp_n)
            # OOV ratio for tokens, if tokenizer known
            oov_ratio = None
            try:
                from metamon.tokenizer import UNKNOWN_TOKEN as _UNK  # type: ignore
                if isinstance(toks, np.ndarray):
                    if toks.size > 0:
                        oov_ratio = float((toks == _UNK).sum()) / float(toks.size)
            except Exception:
                oov_ratio = None
            print(
                f"[MetamonPriorsDBG] obs numbers_shape={n_shape} expected={exp_n}{' MISMATCH' if mismatch else ''} "
                f"text_tokens_nnz={t_nnz} oov_ratio={(round(oov_ratio,4) if isinstance(oov_ratio,float) else 'NA')}",
                flush=True,
            )
        except Exception:
            pass

        # Alignment check: ensure local move/switch ordering matches replay-state ordering
        try:
            from metamon.backend.replay_parser.replay_state import ReplayState as _ReplayState
            # Build ReplayState snapshot (same as score())
            mm = ctx.mm_battle
            replay = getattr(mm, "_mm_battle", None)
            align_moves_ok = None
            align_switch_ok = None
            if replay is not None and getattr(replay, "turnlist", None):
                turn = replay.turnlist[-1]
                role = getattr(ctx, "player_role", None) or getattr(mm, "player_role", None)
                p1 = (role == "p1")
                state = _ReplayState(
                    format=replay.format or "gen9ou",
                    force_switch=bool(getattr(turn, "is_force_switch", False)),
                    active_pokemon=(turn.active_pokemon_1 if p1 else turn.active_pokemon_2)[0] if ((turn.active_pokemon_1 if p1 else turn.active_pokemon_2)) else None,
                    opponent_active_pokemon=(turn.active_pokemon_2 if p1 else turn.active_pokemon_1)[0] if ((turn.active_pokemon_2 if p1 else turn.active_pokemon_1)) else None,
                    opponent_team=(turn.pokemon_2 if p1 else turn.pokemon_1),
                    available_switches=(turn.available_switches_1 if p1 else turn.available_switches_2) or [],
                    player_prev_move=getattr(((turn.active_pokemon_1 if p1 else turn.active_pokemon_2) or [None])[0], "last_used_move", None),
                    opponent_prev_move=getattr(((turn.active_pokemon_2 if p1 else turn.active_pokemon_1) or [None])[0], "last_used_move", None),
                    player_conditions=(turn.conditions_1 if p1 else turn.conditions_2),
                    opponent_conditions=(turn.conditions_2 if p1 else turn.conditions_1),
                    weather=getattr(turn, "weather", None),
                    battle_field=getattr(turn, "battle_field", None),
                    battle_won=bool(getattr(mm, "won", False)),
                    battle_lost=bool(getattr(mm, "lost", False)),
                    can_tera=(turn.can_tera_1 if p1 else turn.can_tera_2),
                    opponent_teampreview=(turn.teampreview_2 if p1 else turn.teampreview_1),
                )
                ustate = UniversalState.from_ReplayState(state)
                # Replay-state orders (names)
                rs_move_names = [getattr(m, 'name', '') for m in ustate.player_active_pokemon.moves]
                rs_move_order = [m for m in consistent_move_order(rs_move_names)]
                rs_switch_names = [getattr(p, 'name', '') for p in ustate.available_switches]
                rs_switch_order = [s for s in consistent_pokemon_order(rs_switch_names)]
                # Local orders
                try:
                    move_map = {getattr(m, 'id'): m for m in (battle.active_pokemon.moves or {}).values()}
                except Exception:
                    move_map = {}
                local_move_ids = list(move_map.keys())
                local_move_order = [m for m in consistent_move_order(local_move_ids)]
                if len(local_move_order) == len(rs_move_order):
                    align_moves_ok = (local_move_order == rs_move_order)
                else:
                    align_moves_ok = False
                # Switches
                try:
                    bench = [
                        p for p in list((battle.team or {}).values())
                        if (not getattr(battle, 'reviving', False) and not getattr(p, 'fainted', False) and not getattr(p, 'active', False))
                        or (getattr(battle, 'reviving', False) and getattr(p, 'fainted', False) and not getattr(p, 'active', False))
                    ]
                except Exception:
                    bench = []
                local_switch_names = [getattr(p, 'species', '') for p in bench]
                local_switch_order = [s for s in consistent_pokemon_order(local_switch_names)]
                if len(local_switch_order) == len(rs_switch_order):
                    align_switch_ok = (local_switch_order == rs_switch_order)
                else:
                    align_switch_ok = False
                # Emit concise alignment summary (head of lists for readability)
                try:
                    print(
                        f"[MetamonAlign] moves_ok={align_moves_ok} local={local_move_order[:4]} replay={rs_move_order[:4]} | "
                        f"switch_ok={align_switch_ok} local={local_switch_order[:4]} replay={rs_switch_order[:4]}",
                        flush=True,
                    )
                except Exception:
                    pass
            # If we can construct the alternate obs path, compute rough diffs
            try:
                alt_obs = None
                if getattr(ctx, 'last_obs_src', None) == 'replay':
                    # Build via Battle
                    u_b = UniversalState.from_Battle(ctx.mm_battle)
                    alt_obs = self._obs_space(u_b)
                elif getattr(ctx, 'last_obs_src', None) == 'battle':
                    # Build via ReplayState
                    alt_obs = self._obs_from_replay_state(ctx)
                if alt_obs is not None:
                    n_a = alt_obs.get('numbers')
                    n_b = obs_np.get('numbers')
                    tt_a = alt_obs.get('text_tokens')
                    tt_b = obs_np.get('text_tokens')
                    l1 = None
                    ham = None
                    try:
                        if isinstance(n_a, np.ndarray) and isinstance(n_b, np.ndarray) and n_a.shape == n_b.shape:
                            l1 = float(np.sum(np.abs(n_a - n_b)))
                    except Exception:
                        l1 = None
                    try:
                        if isinstance(tt_a, np.ndarray) and isinstance(tt_b, np.ndarray) and tt_a.shape == tt_b.shape:
                            ham = int(np.sum(tt_a != tt_b))
                    except Exception:
                        ham = None
                    print(
                        f"[MetamonObsDBG] alt_obs_diff numbers_L1={(round(l1,3) if isinstance(l1,float) else 'NA')} tokens_hamming={(ham if isinstance(ham,int) else 'NA')}",
                        flush=True,
                    )
            except Exception:
                pass
        except Exception:
            pass
        obs = self._to_torch_batch(obs_np, device=self._device)
        rl2s = torch.zeros((1, 1, int(self._rl2_width)), dtype=torch.float32, device=self._device)
        time_idxs = torch.tensor([[ctx.time_idx]], dtype=torch.long, device=self._device).unsqueeze(-1)

        with torch.no_grad():
            tstep_emb = self._agent.tstep_encoder(obs=obs, rl2s=rl2s)
            ret = self._agent.traj_encoder(seq=tstep_emb, time_idxs=time_idxs, hidden_state=ctx.hidden)
            s_rep = ret[0] if isinstance(ret, tuple) else ret
            try:
                print(f"[MetamonPriorsDBG] encoders ok: s_rep_shape={tuple(s_rep.shape) if hasattr(s_rep,'shape') else None} dtype={getattr(s_rep,'dtype',None)}", flush=True)
            except Exception:
                pass
            # Bypass actor_network_forward; reshape base MLP output manually.
            head = self._agent.actor
            dp0 = head.base(s_rep)
            g = int(getattr(head, 'num_gammas', 0))
            d = int(getattr(getattr(head, 'policy_dist', None), 'input_dimension', 0))
            if int(dp0.shape[-1]) != g * d or g <= 0 or d <= 0:
                raise RuntimeError("MetamonBridge.policy_top_n_traj: actor base output shape mismatch.")
            try:
                print(f"[MetamonPriorsDBG] actor.base out_shape={tuple(dp0.shape)}", flush=True)
                print(f"[MetamonPriorsDBG] base_last_dim={int(dp0.shape[-1])} num_gammas={g}", flush=True)
            except Exception:
                pass
            dp = dp0.contiguous().view(1, 1, g, d)
            probs_vec = torch.softmax(dp, dim=-1)[0, 0, -1, :].detach().cpu().numpy().astype(np.float64)

        # Mask out locally illegal actions, renormalize, and take top-N among valid
        valid_mask = np.zeros_like(probs_vec, dtype=bool)
        for i in range(probs_vec.shape[0]):
            if self._action_idx_to_order_local(battle, int(i)) is not None:
                valid_mask[i] = True
        probs_valid = probs_vec.copy()
        probs_valid[~valid_mask] = 0.0
        s = probs_valid.sum()
        if s > 0:
            probs_valid = probs_valid / s
        # Rank valid indices by masked probability
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            raise RuntimeError("MetamonBridge.policy_top_n_traj: no locally valid actions after legality mask.")
        ranked = valid_indices[np.argsort(-probs_valid[valid_indices])]
        out: list[tuple[int, float]] = []
        for idx in ranked:
            out.append((int(idx), float(probs_valid[idx])))
            if len(out) >= max(0, int(top_n)):
                break
        if not out:
            # Fallback: pick first valid indices with tiny uniform weights
            k = int(min(top_n, valid_indices.size))
            out = [(int(i), float(1.0 / k)) for i in valid_indices[:k]]
        return out

    def policy_top_n_orders_traj(self, battle, top_n: int = 3) -> list[LocalBattleOrder]:
        """Return top-N policy priors as local BattleOrder objects (trajectory mode).

        Raises if the trajectory context/observation is not ready or if no valid
        orders can be produced.
        """
        pairs = self.policy_top_n_traj(battle, top_n=top_n)
        orders: list[LocalBattleOrder] = []
        for idx, _p in pairs:
            bo = self._action_idx_to_order_local(battle, idx)
            if bo is None:
                # This should not happen because policy_top_n_traj filters by mapping
                raise RuntimeError("MetamonBridge.policy_top_n_orders_traj: failed to map action_idx to BattleOrder.")
            orders.append(bo)
        if not orders:
            raise RuntimeError("MetamonBridge.policy_top_n_orders_traj: no orders available from policy priors.")
        return orders

    # -------- Critics (trajectory) --------
    def critics_q_traj(
        self,
        battle,
        action_indices: list[int],
    ) -> dict:
        """Evaluate Q(s,a) for given indices using the critic ensemble.

        Returns a dict with keys: 'indices', 'mean', 'var', and 'q_bottom',
        where 'q_bottom' is the 0.25 quantile across critics for each action.
        Raises on missing context or incomplete observations.
        """
        self._ensure_model()
        if not action_indices:
            raise ValueError("MetamonBridge.critics_q_traj: action_indices must be a non-empty list.")
        ctx = self._contexts.get(id(battle))
        if ctx is None:
            raise RuntimeError("MetamonBridge.critics_q_traj: no trajectory context found for battle; ensure contexts are registered/cloned before calling.")
        obs_np = self._build_obs_np(ctx)
        if obs_np is None:
            raise RuntimeError("MetamonBridge.critics_q_traj: observation incomplete for current replay snapshot; context not ready.")
        obs = self._to_torch_batch(obs_np, device=self._device)
        rl2s = torch.zeros((1, 1, int(self._rl2_width)), dtype=torch.float32, device=self._device)
        time_idxs = torch.tensor([[ctx.time_idx]], dtype=torch.long, device=self._device).unsqueeze(-1)

        with torch.no_grad():
            tstep_emb = self._agent.tstep_encoder(obs=obs, rl2s=rl2s)
            ret = self._agent.traj_encoder(seq=tstep_emb, time_idxs=time_idxs, hidden_state=ctx.hidden)
            s_rep = ret[0] if isinstance(ret, tuple) else ret  # [1,1,D]

        try:
            G = int(len(getattr(self._agent, 'gammas', [])) or 1)
            D_action = int(getattr(self._agent, 'action_dim', 0))
            if D_action <= 0:
                raise RuntimeError("MetamonBridge.critics_q_traj: invalid agent action_dim (<=0).")
            # Build one-hot action batch [K,1,1,G,D]
            K = len(action_indices)
            a = torch.zeros((K, 1, 1, G, D_action), dtype=torch.float32, device=self._device)
            for k, idx in enumerate(action_indices):
                if 0 <= int(idx) < D_action:
                    a[k, 0, 0, :, int(idx)] = 1.0
            with torch.no_grad():
                bin_dist = self._agent.critics(s_rep, a)
                q_raw = self._agent.critics.bin_dist_to_raw_vals(bin_dist)  # [K,1,1,C,G,1]
            # Select test gamma (-1), squeeze
            q = q_raw[:, 0, 0, :, -1, 0]  # [K, C]
            q_mean = q.mean(dim=1).detach().cpu().numpy().astype(np.float64).tolist()
            q_var = q.var(dim=1, unbiased=False).detach().cpu().numpy().astype(np.float64).tolist()
            out: dict = {"indices": [int(i) for i in action_indices], "mean": q_mean, "var": q_var}
            # bottom quantile per action across critics at q=0.25
            try:
                q_sorted, _ = torch.sort(q, dim=1)
                q_idx = max(0, min(q.shape[1] - 1, int(np.floor(0.25 * q.shape[1]))))
                q_bot = q_sorted[:, q_idx].detach().cpu().numpy().astype(np.float64).tolist()
                out["q_bottom"] = q_bot
            except Exception:
                out["q_bottom"] = []
            return out
        except Exception as e:
            raise RuntimeError(f"MetamonBridge.critics_q_traj failed: {e}")

    # -------- Observation builder (ReplayState) --------
    def _build_obs_np(self, ctx: _Ctx) -> Optional[Dict[str, np.ndarray]]:
        """Try replay-state obs; if incomplete, fall back to UniversalState.from_Battle."""
        # 1) Prefer replay snapshots (fast path)
        try:
            obs = self._obs_from_replay_state(ctx)
            try:
                ctx.last_obs_src = "replay"
            except Exception:
                pass
            return obs
        except _IncompleteObs as e_in:
            # Incomplete replay snapshot; try battle-based fallback
            try:
                ustate = UniversalState.from_Battle(ctx.mm_battle)
                obs = self._obs_space(ustate)
                try:
                    ctx.last_obs_src = "battle"
                except Exception:
                    pass
                return obs
            except Exception as e_fb:
                try:
                    ctx.last_obs_src = None
                    et1, em1 = type(e_in).__name__, str(e_in)
                    et2, em2 = type(e_fb).__name__, str(e_fb)
                    print(f"[MetamonObsDBG] fallback failed: replay_incomplete({et1}: {em1}) battle_build({et2}: {em2})", flush=True)
                except Exception:
                    pass
                return None
        except Exception as e:
            # Unexpected (e.g., ZeroDivisionError on 0/0): log then try battle-based fallback
            self._dbg(f"replay_state builder failed: {e}")
            try:
                ustate = UniversalState.from_Battle(ctx.mm_battle)
                obs = self._obs_space(ustate)
                try:
                    ctx.last_obs_src = "battle"
                except Exception:
                    pass
                return obs
            except Exception:
                try:
                    ctx.last_obs_src = None
                except Exception:
                    pass
                return None

    def _obs_from_replay_state(self, ctx: _Ctx) -> Dict[str, np.ndarray]:
        """Construct obs via UniversalState.from_ReplayState if snapshot is complete enough."""
        # Lazy import to avoid import-time errors
        from metamon.backend.replay_parser.replay_state import ReplayState as _ReplayState

        mm = ctx.mm_battle
        replay = getattr(mm, "_mm_battle", None)
        if replay is None or not getattr(replay, "turnlist", None):
            # Nothing parsed yet (e.g., just connected / pre-preview)
            try:
                bid = getattr(mm, "battle_tag", "?")
                print(f"[MetamonObsDBG] incomplete: no replay turns yet battle={bid}", flush=True)
            except Exception:
                pass
            raise _IncompleteObs("no replay turns yet")

        turn = replay.turnlist[-1]

        # Determine player's POV (prefer ctx cache, then backend field)
        role = getattr(ctx, "player_role", None) or getattr(mm, "player_role", None)
        p1 = (role == "p1")

        # Active mons (may be empty at preview/very early stages)
        active_list = turn.active_pokemon_1 if p1 else turn.active_pokemon_2
        opp_active_list = turn.active_pokemon_2 if p1 else turn.active_pokemon_1
        active_mon = active_list[0] if active_list else None
        opp_active_mon = opp_active_list[0] if opp_active_list else None

        # If actives are unknown at this snapshot, defer and let fallback handle it
        def _hp_known(p):
            try:
                if p is None:
                    return False
                chp = getattr(p, "current_hp", None)
                mhp = getattr(p, "max_hp", None)
                # Metamon accepts 0/0 for fainted mons; treat as known
                if chp == 0 and mhp == 0:
                    return True
                return (
                    chp is not None
                    and mhp is not None
                    and int(mhp) > 0
                )
            except Exception:
                return False

        if not _hp_known(active_mon) or not _hp_known(opp_active_mon):
            # Emit a concise diagnostic so we can see exactly what's missing
            try:
                tnum = getattr(turn, "turn_number", None)
                fs = bool(getattr(turn, "is_force_switch", False))
                def _hp_tuple(p):
                    try:
                        return (getattr(p, "current_hp", None), getattr(p, "max_hp", None))
                    except Exception:
                        return (None, None)
                ahp = _hp_tuple(active_mon)
                ohp = _hp_tuple(opp_active_mon)
                a_present = active_mon is not None
                o_present = opp_active_mon is not None
                bid = getattr(mm, "battle_tag", "?")
                print(
                    f"[MetamonObsDBG] incomplete: hp_missing battle={bid} turn={tnum} role={role} "
                    f"force_switch={fs} active_present={a_present} opp_present={o_present} "
                    f"active_hp={ahp} opp_hp={ohp}",
                    flush=True,
                )
            except Exception:
                pass
            raise _IncompleteObs("active or opponent HP unknown")

        # Available switches (graceful defaults)
        try:
            switches = list(getattr(mm, "_available_switches", []))
        except Exception:
            switches = (turn.available_switches_1 if p1 else turn.available_switches_2) or []
        # Filter out entries with unknown/invalid HP info to avoid downstream divide-by-zero
        def _ok_switch(p):
            try:
                return (
                    p is not None
                    and getattr(p, "current_hp", None) is not None
                    and getattr(p, "max_hp", None) is not None
                    and int(getattr(p, "max_hp")) > 0
                )
            except Exception:
                return False
        switches = [p for p in switches if _ok_switch(p)]

        state = _ReplayState(
            format=replay.format or "gen9ou",
            force_switch=bool(getattr(turn, "is_force_switch", False)),
            active_pokemon=active_mon,
            opponent_active_pokemon=opp_active_mon,
            opponent_team=(turn.pokemon_2 if p1 else turn.pokemon_1),
            available_switches=switches,
            player_prev_move=getattr(active_mon, "last_used_move", None),
            opponent_prev_move=getattr(opp_active_mon, "last_used_move", None),
            player_conditions=(turn.conditions_1 if p1 else turn.conditions_2),
            opponent_conditions=(turn.conditions_2 if p1 else turn.conditions_1),
            weather=getattr(turn, "weather", None),
            battle_field=getattr(turn, "battle_field", None),
            battle_won=bool(getattr(mm, "won", False)),
            battle_lost=bool(getattr(mm, "lost", False)),
            can_tera=(turn.can_tera_1 if p1 else turn.can_tera_2),
            opponent_teampreview=(turn.teampreview_2 if p1 else turn.teampreview_1),
        )

        ustate = UniversalState.from_ReplayState(state)
        return self._obs_space(ustate)


# Global, lazily used bridge
METAMON_BRIDGE = MetamonBridge()


def enable(model_name: Optional[str] = None, *, strict: bool = False):
    """Enable the Metamon bridge and optionally validate readiness eagerly.

    Args:
        model_name: Optional pretrained model name to use (e.g., "Abra").
        strict: If True, eagerly load the model and validate readiness. Raises
            on failure instead of silently falling back.
    """
    if model_name:
        METAMON_BRIDGE._model_name = model_name
    METAMON_BRIDGE.install_hooks()
    if strict:
        # Validate required env and eager model load
        cache_dir = os.getenv("METAMON_CACHE_DIR")
        if not cache_dir or not os.path.isdir(cache_dir):
            raise RuntimeError("METAMON_CACHE_DIR must be set to a valid directory for Metamon models")
        METAMON_BRIDGE._ensure_model()
        ready, reason = METAMON_BRIDGE.is_ready()
        if not ready:
            raise RuntimeError(f"Metamon bridge not ready: {reason}")
