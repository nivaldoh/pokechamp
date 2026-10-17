from __future__ import annotations

from typing import Dict, List


def _canon(s: str) -> str:
    """Canonicalize a species/nickname string for fuzzy matching.

    Lowercase; remove non-alphanumeric characters.
    """
    return "".join(ch for ch in s.lower() if ch.isalnum())


class _NormState:
    def __init__(self) -> None:
        # Roster per side ('p1'/'p2') as details strings (e.g., 'Gliscor, F')
        self.roster: Dict[str, List[str]] = {"p1": [], "p2": []}


class SimProtocolNormalizer:
    """Normalize LocalSim Showdown-style messages for Metamon's parser.

    Fills missing fields (e.g., empty switch details) using roster info from earlier
    'poke' events. Designed to be used only for LocalSim messages. Live pip poke_env
    messages should be forwarded without normalization.
    """

    def __init__(self) -> None:
        self._states: Dict[int, _NormState] = {}

    def _state(self, battle) -> _NormState:
        bid = id(battle)
        st = self._states.get(bid)
        if st is None:
            st = _NormState()
            self._states[bid] = st
        return st

    def clone_state(self, parent_battle, child_battle) -> None:
        """Clone normalization state (e.g., roster) from parent to child battle id.

        Needed so ephemeral LocalSim branches inherit team preview rosters collected
        from earlier 'poke' events in the live context.
        """
        p = self._states.get(id(parent_battle))
        if p is None:
            return
        child = _NormState()
        # shallow copy of lists is fine here
        child.roster = {side: list(lst) for side, lst in p.roster.items()}
        self._states[id(child_battle)] = child

    def cleanup(self, battle) -> None:
        """Remove any stored state for a battle id (avoid leaks)."""
        self._states.pop(id(battle), None)

    def normalize(self, battle, split_message: List[str]) -> List[str]:
        # Defensive copy
        msg = list(split_message)
        if not msg or len(msg) < 2:
            return msg
        kind = msg[1]
        st = self._state(battle)

        if kind == "poke":
            # ['', 'poke', 'p1', 'Thundurus, M', '']
            if len(msg) >= 4:
                side = str(msg[2]).strip() if msg[2] else ""
                details = str(msg[3]).strip() if msg[3] else ""
                if side in ("p1", "p2") and details:
                    if details not in st.roster[side]:
                        st.roster[side].append(details)
            return msg

        if kind == "switch":
            # ['', 'switch', 'p2a: Gliscor', 'Gliscor, F', '100/100']
            # Some LocalSim messages provide empty details -> reconstruct
            if len(msg) >= 5:
                ident = str(msg[2]) if msg[2] else ""
                details = str(msg[3]) if msg[3] else ""
                if not details:
                    # Derive side and nickname/species from ident
                    side = ""
                    name = ""
                    try:
                        prefix, name = ident.split(":", 1)
                        name = name.strip()
                        side = prefix.strip()[:2]  # 'p1a' -> 'p1'
                    except Exception:
                        pass
                    # Try to map to a known roster species for this side
                    chosen = None
                    if side in ("p1", "p2") and st.roster[side]:
                        cn = _canon(name)
                        # match against species only (strip trailing gender)
                        def _species_only(s: str) -> str:
                            return s.split(",")[0].strip()
                        for cand in st.roster[side]:
                            cc = _canon(_species_only(cand))
                            if cn and (cn in cc or cc in cn):
                                chosen = cand
                                break
                    # If still none, attempt to query from the local battle object
                    if chosen is None and side in ("p1", "p2"):
                        try:
                            team = getattr(battle, "team") if side == "p1" else getattr(battle, "opponent_team")
                            # team is a dict of Pokemon
                            def _pick_gender(g):
                                try:
                                    # PokemonGender enum exposes name like 'MALE'/'FEMALE'/'NEUTRAL'
                                    if g and getattr(g, "name", "").startswith("MALE"):
                                        return "M"
                                    if g and getattr(g, "name", "").startswith("FEMALE"):
                                        return "F"
                                    return ""
                                except Exception:
                                    return ""
                            best_p = None
                            cn = _canon(name)
                            for p in getattr(team, "values")() if callable(getattr(team, "values", None)) else []:
                                sp = getattr(p, "species", None)
                                if not sp:
                                    continue
                                if cn and (cn in _canon(sp) or _canon(sp) in cn):
                                    best_p = p
                                    break
                            if best_p is not None:
                                g = _pick_gender(getattr(best_p, "gender", None))
                                chosen = f"{getattr(best_p, 'species')}" + (f", {g}" if g else "")
                        except Exception:
                            pass
                    # Fall back to name only
                    msg[3] = chosen or name
            return msg

        # no-op for other kinds
        if kind == "move":
            # Canonicalize common LocalSim move name variants to Showdown names
            # ['', 'move', 'pXa: Name', 'MoveName', 'pYa: Target']
            try:
                if len(msg) >= 4 and msg[3]:
                    nm = str(msg[3]).strip()
                    mapping = {
                        # common LocalSim variants
                        'Voltswitch': 'Volt Switch',
                        'Flipturn': 'Flip Turn',
                        'Makeitrain': 'Make It Rain',
                        'Shadowball': 'Shadow Ball',
                        'Ceaselessedge': 'Ceaseless Edge',
                    }
                    # If already spaced correctly keep it
                    if nm in mapping:
                        msg[3] = mapping[nm]

                    # LocalSim sometimes emits a pseudo-move 'Terastallize'.
                    # Metamon's replay parser expects a side effect line:
                    #   ['','-terastallize', IDENT, TYPE]
                    # Translate here by deriving the Tera type from the local battle state.
                    if nm.lower() == 'terastallize':
                        try:
                            ident = str(msg[2]) if len(msg) > 2 else ''
                            # Resolve the acting Pokemon from the local battle to read its (or guessed) Tera type
                            p = battle.get_pokemon(ident) if hasattr(battle, 'get_pokemon') else None
                            tera_type: str = ''
                            if p is not None:
                                # Prefer an already set terastallized type
                                t = getattr(p, '_terastallized_type', None)
                                if t is not None:
                                    # PokemonType has a .name attribute (e.g., 'Bug', 'Fairy')
                                    tera_type = getattr(t, 'name', '') or ''
                                if not tera_type and hasattr(p, 'guess_tera'):
                                    # Fall back to model-backed guess from local sets data
                                    try:
                                        tera_type = p.guess_tera() or ''
                                    except Exception:
                                        tera_type = ''
                            # Only translate if we have a plausible type; otherwise keep the original move line
                            if ident and tera_type:
                                return ['', '-terastallize', ident, tera_type]
                        except Exception:
                            # If anything goes wrong, fall through and keep the original message
                            pass
            except Exception:
                pass
        return msg

    def reconstruct_details(self, battle, side: str, name: str) -> str:
        """Reconstruct a details string like 'Species, M' for a side/name.

        Uses roster first, falls back to local battle team/opponent_team.
        """
        st = self._state(battle)
        # try roster species w/ gender
        if side in ("p1", "p2"):
            for cand in st.roster[side]:
                if _canon(name) in _canon(cand) or _canon(cand) in _canon(name):
                    return cand
        # fallback to local battle object
        try:
            team = getattr(battle, "team") if side == "p1" else getattr(battle, "opponent_team")
            for p in getattr(team, "values")():
                sp = getattr(p, 'species', None)
                if not sp:
                    continue
                if _canon(name) in _canon(sp) or _canon(sp) in _canon(name):
                    g = getattr(p, 'gender', None)
                    gch = ''
                    try:
                        if g and getattr(g, 'name', '').startswith('MALE'):
                            gch = ', M'
                        elif g and getattr(g, 'name', '').startswith('FEMALE'):
                            gch = ', F'
                    except Exception:
                        pass
                    return f"{sp}{gch}"
        except Exception:
            pass
        return name
