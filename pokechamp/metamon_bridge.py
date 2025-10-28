"""
MetamonBridge: Runtime bridge that mirrors Showdown messages/requests (both live messages as well as messages from simulated nodes?) into a
Metamon trajectory context, and provides a leaf scoring API for minimax.

Design goals
- Do not modify poke_env. Install lightweight monkey‑patches at runtime.
- Keep one context per Python Battle instance (keyed by id(battle)).
- Maintain a live root context from real server traffic; clone it for branches.
- Provide score(battle) -> int in [0, 100] using a pretrained Metamon model.

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

try:
    import torch
except Exception:  # pragma: no cover - optional at import time
    torch = None  # type: ignore

# metamon installed as an editable module from the pokechamp root folder
from metamon.env.metamon_battle import MetamonBackendBattle
from metamon.rl.pretrained import get_pretrained_model
from metamon.rl.win_probability_head import WinProbabilityHead
from metamon.interface import UniversalState
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
        self._head: Optional[WinProbabilityHead] = None
        self._head_loaded: bool = False

        # Cache hooks (per-target)
        self._patched: Dict[str, bool] = {}
        self._orig_parse_message: Dict[str, Any] = {}
        self._orig_parse_request: Dict[str, Any] = {}
        # Pre-log sequence counters per battle
        self._prelog_seq: Dict[int, int] = {}

        # Configuration
        self._model_name = os.getenv("POKECHAMP_METAMON_MODEL", "Abra")
        self._winprob_ckpt = os.getenv("METAMON_WINPROB_CKPT")
        self._debug = str(os.getenv("POKECHAMP_METAMON_DEBUG", "")).lower() in {"1", "true", "t", "yes", "y"}
        # LocalSim normalization layer
        self._normalizer = SimProtocolNormalizer()

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
        # Small MLP head for win probability
        self._head = WinProbabilityHead(state_dim=self._agent.traj_encoder.emb_dim)
        # Optionally load trained win-probability head
        self._head_loaded = False
        if self._winprob_ckpt:
            try:
                ckpt = torch.load(self._winprob_ckpt, map_location=self._device)
                state = ckpt.get("head_state_dict") if isinstance(ckpt, dict) else None
                if state:
                    missing, unexpected = self._head.load_state_dict(state, strict=False)
                    if missing or unexpected:
                        self._dbg(f"Loaded head with missing={missing} unexpected={unexpected}")
                    self._head_loaded = True
                else:
                    self._dbg("WinProb checkpoint missing 'head_state_dict'; using randomly initialized head")
            except Exception as e:
                self._dbg(f"Failed to load winprob head from {self._winprob_ckpt}: {e}")
        self._head.eval()
        try:
            self._head.to(self._device)
        except Exception:
            pass
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
                # Pre-log the raw message before any processing
                try:
                    # Print a clear separator at turn boundaries
                    if self._debug and isinstance(split_message, list) and len(split_message) > 1:
                        if split_message[1] == "turn":
                            turn_no = split_message[2] if len(split_message) > 2 else "?"
                            btag = getattr(this, "battle_tag", "?")
                            print(f"\n\n\n[MetamonBridge] ===== TURN {turn_no} ===== battle={btag}", flush=True)
                    if self._debug:
                        bid = id(this)
                        n = self._prelog_seq.get(bid, 0) + 1
                        self._prelog_seq[bid] = n
                        evt = split_message[1] if isinstance(split_message, list) and len(split_message) > 1 else "?"
                        tag_str = f"{tag}"
                        btag = getattr(this, "battle_tag", "?")
                        print(
                            f"[MetamonBridge] PRE[{tag_str}] msg#{n} id={bid} battle={btag} event={evt} raw={split_message}",
                            flush=True,
                        )
                except Exception:
                    pass
                # Run original first to keep engine behavior authoritative
                _orig(this, split_message)
                try:
                    self.on_message(this, split_message, source=tag)
                except Exception as e:
                    self._dbg(f"{tag}.on_message error: {e}")

            _AB.parse_message = _wrapped_parse_message  # type: ignore

            self._orig_parse_request[tag] = _B.parse_request

            def _wrapped_parse_request(this, request, _orig=self._orig_parse_request[tag]):  # type: ignore
                # Pre-log the raw request before any processing
                try:
                    if self._debug:
                        bid = id(this)
                        n = self._prelog_seq.get(bid, 0) + 1
                        self._prelog_seq[bid] = n
                        btag = getattr(this, "battle_tag", "?")
                        is_tp = False
                        try:
                            is_tp = bool(request.get("teamPreview", False))
                        except Exception:
                            is_tp = False
                        print(
                            f"[MetamonBridge] PRE[{tag}] req#{n} id={bid} battle={btag} teamPreview={is_tp} keys={list(request.keys())}",
                            flush=True,
                        )
                except Exception:
                    pass
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

        Checks model, device, and head readiness. If POKECHAMP_ALLOW_UNTRAINED_HEAD
        is set, absence of a head checkpoint is not considered fatal.
        """
        try:
            allow_untrained = str(os.getenv("POKECHAMP_ALLOW_UNTRAINED_HEAD", "0")).lower() in {"1", "true", "t", "yes", "y"}
            if self._experiment is None or self._agent is None:
                return False, "model_not_loaded"
            if self._obs_space is None or self._rl2_width is None:
                return False, "observation_space_not_ready"
            if self._head is None:
                return False, "head_missing"
            if not self._head_loaded and not allow_untrained:
                return False, "head_not_loaded_set_METAMON_WINPROB_CKPT_or_allow_untrained"
            return True, "ok"
        except Exception as e:
            return False, f"exception:{e}"

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
            except Exception as e:
                self._dbg(f"clone_context deep copy failed: {e}")

            hidden = self._clone_hidden(parent.hidden) if torch is not None else None
            ctx = _Ctx(mm_battle=child_mm, hidden=hidden, time_idx=parent.time_idx, ephemeral=True)
            ctx.player_role = getattr(parent, "player_role", None)
            self._contexts[ckey] = ctx
            # Clone normalizer state so LocalSim branches inherit roster info
            try:
                self._normalizer.clone_state(parent_battle, child_battle)
            except Exception:
                pass
            # Debug: log clone mapping and parent replay size
            try:
                if self._debug or str(os.getenv("POKECHAMP_MINIMAX_LOG", "0")).lower() in {"1","true","t","yes","y"}:
                    turns = 0
                    try:
                        preplay = getattr(parent.mm_battle, "_mm_battle", None)
                        if preplay is not None and getattr(preplay, "turnlist", None):
                            turns = len(preplay.turnlist)
                    except Exception:
                        pass
                    print(
                        f"[MetamonBridge] clone_context parent_id={id(parent_battle)} -> child_id={id(child_battle)} turns={turns}",
                        flush=True,
                    )
            except Exception:
                pass

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
                        if self._debug and msg != split_message:
                            print(f"[MetamonBridge] normalize local msg: {split_message} -> {msg}", flush=True)
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
                                        if self._debug:
                                            print(f"[MetamonBridge] seed switch: {seed}", flush=True)
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
            except Exception as e:
                self._dbg_exc("advance_hidden on_request error", e)

    # -------- Scoring --------
    def score(self, battle) -> Optional[int]:
        """Return score in [0, 100] for the given battle branch.

        Uses the branch context’s hidden state as live history; applies a single
        observation step for the current snapshot; does not mutate hidden state.
        """
        key = id(battle)
        ctx = self._contexts.get(key)
        if ctx is None:
            return None
        self._ensure_model()
        if torch is None:
            return None
        # Build current obs (prefer replay; fall back to battle)
        obs_np = self._build_obs_np(ctx)
        if obs_np is None:
            return None
        # Optional debug: summarize obs inputs
        try:
            if self._debug or str(os.getenv("POKECHAMP_MINIMAX_LOG", "0")).lower() in {"1","true","t","yes","y"}:
                nums = obs_np.get("numbers")
                toks = obs_np.get("text_tokens")
                n_sum = float(np.sum(nums)) if isinstance(nums, np.ndarray) else float('nan')
                n_mean = float(np.mean(nums)) if isinstance(nums, np.ndarray) else float('nan')
                t_len = int(np.count_nonzero(toks)) if isinstance(toks, np.ndarray) else 0
                # Try to extract active/opponent snapshot
                role = getattr(ctx, "player_role", None) or getattr(ctx.mm_battle, "player_role", None)
                turns = 0
                who = "?"
                act = opp = "?"
                try:
                    replay = getattr(ctx.mm_battle, "_mm_battle", None)
                    if replay is not None and getattr(replay, "turnlist", None):
                        turns = len(replay.turnlist)
                        p1 = (role == "p1")
                        last = replay.turnlist[-1]
                        a = (last.active_pokemon_1 if p1 else last.active_pokemon_2)
                        o = (last.active_pokemon_2 if p1 else last.active_pokemon_1)
                        if a:
                            act_hp = f"{getattr(a[0],'current_hp',None)}/{getattr(a[0],'max_hp',None)}"
                            act = f"{getattr(a[0], 'name', '?')}({act_hp})"
                        if o:
                            opp_hp = f"{getattr(o[0],'current_hp',None)}/{getattr(o[0],'max_hp',None)}"
                            opp = f"{getattr(o[0], 'name', '?')}({opp_hp})"
                        who = role or "None"
                except Exception:
                    pass
                print(f"[MetamonBridge] score dbg: ctx={id(battle)} ephemeral={ctx.ephemeral} t_idx={ctx.time_idx} role={who} turns={turns} active={act} opp={opp} nums(sum={n_sum:.2f}, mean={n_mean:.2f}) toks_nz={t_len}", flush=True)
        except Exception:
            pass
        obs = self._to_torch_batch(obs_np, device=self._device)
        rl2s = torch.zeros((1, 1, int(self._rl2_width)), dtype=torch.float32, device=self._device)
        # Transformer expects pos indices with shape [B, L, 1]; add last dim
        time_idxs = torch.tensor([[ctx.time_idx]], dtype=torch.long, device=self._device).unsqueeze(-1)

        with torch.no_grad():
            tstep_emb = self._agent.tstep_encoder(obs=obs, rl2s=rl2s)
            ret = self._agent.traj_encoder(seq=tstep_emb, time_idxs=time_idxs, hidden_state=ctx.hidden)
            s_rep = ret[0] if isinstance(ret, tuple) else ret
            logits = self._head(s_rep, mask=None)  # [1,1]
            prob = torch.sigmoid(logits).item()
            try:
                if self._debug or str(os.getenv("POKECHAMP_MINIMAX_LOG", "0")).lower() in {"1","true","t","yes","y"}:
                    print(f"[MetamonBridge] score dbg: ctx={id(battle)} prob={prob:.4f}", flush=True)
            except Exception:
                pass
        score = int(round(max(0.0, min(1.0, prob)) * 100.0))
        return score

    # -------- Observation builder (ReplayState) --------
    def _build_obs_np(self, ctx: _Ctx) -> Optional[Dict[str, np.ndarray]]:
        """Try replay-state obs; if incomplete, fall back to UniversalState.from_Battle."""
        # 1) Prefer replay snapshots (fast path)
        try:
            return self._obs_from_replay_state(ctx)
        except _IncompleteObs:
            # Incomplete early-phase info: skip advancing/score for now
            return None
        except Exception as e:
            # Truly unexpected; log once
            self._dbg(f"replay_state builder failed: {e}")
            return None

    def _obs_from_replay_state(self, ctx: _Ctx) -> Dict[str, np.ndarray]:
        """Construct obs via UniversalState.from_ReplayState if snapshot is complete enough."""
        # Lazy import to avoid import-time errors
        from metamon.backend.replay_parser.replay_state import ReplayState as _ReplayState

        mm = ctx.mm_battle
        replay = getattr(mm, "_mm_battle", None)
        if replay is None or not getattr(replay, "turnlist", None):
            # Nothing parsed yet (e.g., just connected / pre-preview)
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
                return (
                    p is not None
                    and getattr(p, "current_hp", None) is not None
                    and getattr(p, "max_hp", None) is not None
                    and int(getattr(p, "max_hp")) > 0
                )
            except Exception:
                return False

        if not _hp_known(active_mon) or not _hp_known(opp_active_mon):
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
