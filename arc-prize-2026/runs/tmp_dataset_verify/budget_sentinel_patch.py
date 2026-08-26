"""Budget sentinel — component (a) of the war-v3 conversion stack.

Design refs:
  - learnings/war_room/grinder_cracking_design.md §(a) "Budget sentinel"
    (ceiling +0.06/draw): every Qwen grinder death was a budget death it never
    saw coming (lp85 GAME_OVERs at 68 and 131-133 vs a 60-click budget; sb26
    move-limit GAME_OVER at 140; ft09 x2 exhaustion GAME_OVERs; tu93 burned 301
    acts on L1 vs base 19). GPT-5.6 read the budget on turn 2 and never died to
    it. The fix makes Qwen AWARE of remaining action budget so it stops
    open-ended exploration before the cap kills the level attempt in progress.
  - grinder_cracking_design.md §3 (pre-registered offline gate): the mechanism
    prong needs a trigger counter >=1/run on >=5 games under a compressed
    budget (the A10 canary). Every threshold crossing emits ONE countable event.
  - duck_eval/ewm_exec/EVENT_SCHEMA.md (event-shaped canary): countable trigger
    events on stdout, one greppable line per firing, so the canary can verify
    firing per game (totals-shaped counters are structurally insufficient).

THE MECHANISM (minimal intervention):
  On every executed action the harness knows both the per-game action budget
  (``solver.max_actions_per_game``) and the actions spent so far
  (``session.action_count``). When the FRACTION of the GAME's total budget
  consumed (cumulative actions / budget) crosses a registered threshold
  (default 50% / 75% / 90%) for the first time in the game, the sentinel:
    1. emits ONE ``SENTINEL`` event (greppable stdout line + per-game
       ``*_sentinel_events.jsonl`` sidecar) marking game / action_num /
       threshold -- the canary counts these;
    2. queues a single budget-state FACT for that game; the FACT is drained and
       appended to the NEXT user prompt exactly once, then discarded.
  The FACT is injected ONLY on threshold crossings, never every turn (always-on
  injection is a known failure mode -- war-v2's 1552-digest / 0-escalation
  constant context tax).

  v2 UNIT CHANGE (panel R16, unanimous across 5 reviewers): v1 keyed the budget
  per LEVEL ATTEMPT (fresh clock on level-up/GAME_OVER restart), but the token-
  implied envelope the budget models (~63k tokens = 147-156 actions) is per
  GAME -- attempt re-arming made the sentinel warn late or never in multi-
  attempt grinder games (runs/sentinel_attempt_unit_b150.json: 15/33 envelope-
  crossing (game,seed) units got no warning by 90% of the envelope; 13 cross-
  attempt-waste episodes). v2 counts CUMULATIVE game actions against the budget
  and each threshold fires at most once per game (max 3 events/game). Level-
  attempt boundaries are still tracked, but only as event metadata.

House pattern (mirrors duck_eval/ledger/ledger_patch.py + continuation_patch.py):
  VERSION marker, env kill switch (SENTINEL_DISABLE=1), blanket-guarded apply,
  runtime banner print, vanilla fallback on ANY failure (worst case: stock duck,
  unchanged). Per-game store keyed by the runtime-state filename stem (NOT the
  parent dir -- ledger v1's parent-dir keying silently shared one store across
  all concurrent games; panel R12 N6). NO threading.Lock is ever held while
  calling a function that re-acquires it (a prior deadlock scored 0.00): the
  lock guards dict ops ONLY, every call-out (FACT build, event emit, I/O)
  happens outside the lock.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger("sentinel")

VERSION = "v2"

# Thresholds = fraction of the per-game action budget CONSUMED. Firing at 0.90
# still leaves ~10% of the budget for the model to convert an in-progress level;
# 0.50/0.75 are early warnings so the model can wind down open-ended probing.
# Override with SENTINEL_THRESHOLDS="0.5,0.75,0.9" (comma-separated fractions).
_DEFAULT_THRESHOLDS = (0.50, 0.75, 0.90)

# Event grep anchor (EVENT_SCHEMA.md convention: one greppable stdout line per
# firing; parsers anchor on the first occurrence of this token, not column 0).
EVENT_ANCHOR = "SENTINEL "

_APPLIED = False
_STORES: dict[str, "_GameBudget"] = {}
_STORES_LOCK = threading.Lock()  # guards _STORES dict ops ONLY -- never held
                                 # across a call-out (deadlock lesson).


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
def _thresholds() -> tuple[float, ...]:
    raw = os.environ.get("SENTINEL_THRESHOLDS", "").strip()
    if not raw:
        return _DEFAULT_THRESHOLDS
    out: list[float] = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except ValueError:
            continue
        if 0.0 < v < 1.0:
            out.append(v)
    return tuple(sorted(set(out))) or _DEFAULT_THRESHOLDS


def _budget_override() -> int | None:
    """SENTINEL_BUDGET forces a per-game budget when the solver leaves
    max_actions_per_game unset (uncapped runs, or the canary/smoke rigs)."""
    raw = os.environ.get("SENTINEL_BUDGET", "").strip()
    if not raw:
        return None
    try:
        v = int(raw)
        return v if v > 0 else None
    except ValueError:
        return None


def _store_key(state_path: Path) -> str:
    """Per-game registry key from the runtime-state filename stem.

    Mirrors ledger_patch._ledger_key: the live harness puts every game's
    ``<run_stem>_runtime_state.json`` in ONE shared artifacts dir, so keying by
    the parent dir would share one store across all concurrent games (ledger v1
    bug). The stem ``<game_id>_p<pass>`` is unique per game/pass.
    """
    p = Path(state_path)
    name = p.name
    i = name.find("runtime_state")
    stem = name[:i].rstrip("_.") if i >= 0 else p.stem
    if not stem:
        stem = p.parent.name or "game"
    return stem


# --------------------------------------------------------------------------- #
# per-game budget state
# --------------------------------------------------------------------------- #
class _GameBudget:
    """Threshold-crossing tracker for one game. Not internally locked: each
    instance is only ever touched from the single solver worker driving that
    game (harness sessions are one-thread-per-game); the module lock only
    protects the SHARED registry dict."""

    __slots__ = ("game", "thresholds", "fired", "attempt", "pending_fact",
                 "last_level", "attempt_base")

    def __init__(self, game: str, thresholds: tuple[float, ...]):
        self.game = game
        self.thresholds = thresholds
        self.fired: set[float] = set()      # thresholds fired this GAME (v2:
                                            # never re-armed; max 3 events/game)
        self.attempt = 0                    # level-attempt ordinal (0-based),
                                            # event metadata only in v2
        self.pending_fact: str | None = None
        self.last_level = 1
        self.attempt_base = 0               # actions spent when this attempt began

    def reset_attempt(self) -> None:
        """v2: a fresh level attempt (level-up or GAME_OVER restart) only
        advances the attempt ordinal for event metadata. It does NOT re-arm
        thresholds and does NOT restart the budget clock -- the budget models
        the per-GAME token envelope (R16 unit repair)."""
        self.attempt += 1

    def crossings(self, consumed: int, budget: int) -> list[tuple[float, int]]:
        """Return the (threshold, remaining) pairs newly crossed at this
        action, in ascending threshold order; mark them fired. Fires at most
        once per threshold per GAME (v2)."""
        if budget <= 0:
            return []
        frac = consumed / budget
        out: list[tuple[float, int]] = []
        for th in self.thresholds:
            if th not in self.fired and frac >= th:
                self.fired.add(th)
                out.append((th, max(0, budget - consumed)))
        return out


def _get_store(state_path: Path, thresholds: tuple[float, ...]) -> _GameBudget:
    key = _store_key(state_path)
    with _STORES_LOCK:                       # dict ops only under the lock
        st = _STORES.get(key)
        if st is None:
            st = _STORES[key] = _GameBudget(key, thresholds)
    return st


# --------------------------------------------------------------------------- #
# FACT + event emission
# --------------------------------------------------------------------------- #
def _build_fact(game: str, action_num: int, threshold: float, budget: int,
                remaining: int) -> str:
    pct = int(round(threshold * 100))
    return (
        f"FACT: budget sentinel -- you have used {action_num} of ~{budget} "
        f"total actions for this game ({pct}% of the game's action budget); "
        f"about {remaining} actions remain for ALL remaining levels. "
        "Stop open-ended exploration: commit to your single best hypothesis and "
        "the shortest action sequence that could clear the current level now. "
        "Wasted actions are gone for the whole game, not just this attempt."
    )


def _sidecar_path(state_path: Path) -> Path:
    """Per-game sentinel events sidecar next to the runtime-state file
    (same artifacts dir the viewer sidecars live in)."""
    p = Path(state_path)
    stem = _store_key(p)
    return p.parent / f"{stem}_sentinel_events.jsonl"


def _emit_event(state_path: Path, game: str, action_num: int, threshold: float,
                budget: int, remaining: int, attempt: int) -> None:
    """Emit ONE countable trigger event: greppable stdout line (canary reads
    this) + best-effort per-game jsonl sidecar. Called OUTSIDE any lock."""
    pct = int(round(threshold * 100))
    # Greppable stdout line (EVENT_SCHEMA.md style: anchor + fixed key=value
    # tokens, ASCII, no spaces inside values, one event per line).
    print(
        f"SENTINEL v=2 kind=budget_threshold game={game} action_num={action_num} "
        f"threshold={threshold:.2f} pct={pct} budget={budget} "
        f"remaining={remaining} attempt={attempt}",
        flush=True,
    )
    try:
        rec = {
            "kind": "budget_threshold",
            "game": game,
            "action_num": action_num,
            "threshold": round(threshold, 4),
            "pct": pct,
            "budget": budget,
            "remaining": remaining,
            "attempt": attempt,
        }
        path = _sidecar_path(state_path)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception as exc:  # noqa: BLE001 - sidecar is best-effort
        log.debug("sentinel sidecar write failed: %s", exc)


# --------------------------------------------------------------------------- #
# patches
# --------------------------------------------------------------------------- #
def _apply_patches() -> None:
    global _APPLIED
    if _APPLIED:
        return

    import inference.framework.solver as solver_mod
    from inference.agent.tool_agent import ToolAgent

    thresholds = _thresholds()

    # (1) harness-side: detect threshold crossings on every executed action ----
    _orig_execute_action = solver_mod._HarnessGameSession._execute_action

    def sentinel_execute_action(self, action, *, batch_index, batch_size,
                                generated_tokens=None, flush_viewer_payload=True):
        payload = _orig_execute_action(
            self, action, batch_index=batch_index, batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload)
        try:
            budget = _budget_override()
            if budget is None:
                budget = getattr(self.solver, "max_actions_per_game", None)
            if not budget or budget <= 0:
                return payload  # uncapped run -> sentinel is a silent no-op
            state_path = Path(self.state_path)
            st = _get_store(state_path, thresholds)

            # v2: attempt boundaries are tracked ONLY to label events with the
            # attempt ordinal -- they do NOT re-arm thresholds or restart the
            # budget clock (the budget models the per-GAME token envelope; R16
            # unit repair). Boundary triggers unchanged from v1, each fired at
            # most once per boundary:
            #   * a GAME_OVER restart (same level, fresh attempt), or
            #   * the level number advancing.
            action_num = int(payload.get("action_num") or self.action_count)
            level = int(payload.get("level") or st.last_level)
            level_up = level != st.last_level
            new_attempt = bool(payload.get("game_over")) or level_up
            if new_attempt and action_num > st.attempt_base:
                st.reset_attempt()
                st.last_level = level
                st.attempt_base = (action_num - 1) if level_up else action_num

            # Budget fraction = CUMULATIVE game actions / per-game budget.
            consumed = max(0, action_num)
            crossings = st.crossings(consumed, budget)
            if crossings:
                # Highest threshold crossed carries the FACT (most urgent);
                # every crossing still emits its own countable event.
                for th, remaining in crossings:
                    _emit_event(state_path, st.game, action_num, th, budget,
                                remaining, st.attempt)
                top_th, top_rem = crossings[-1]
                st.pending_fact = _build_fact(
                    st.game, action_num, top_th, budget, top_rem)
        except Exception as exc:  # noqa: BLE001 - never break the action path
            log.debug("sentinel execute_action hook failed: %s", exc)
        return payload

    solver_mod._HarnessGameSession._execute_action = sentinel_execute_action

    # (2) analyze: bind the per-game store so the prompt hook can find it ------
    _orig_analyze = ToolAgent.analyze

    def sentinel_analyze(self, state_path, action_num, valid_actions=None,
                         step_env=None, **kwargs):
        try:
            self._sentinel_state_path = Path(state_path)
        except Exception as exc:  # noqa: BLE001 - never break the turn
            log.debug("sentinel pre-analyze hook failed: %s", exc)
        return _orig_analyze(self, state_path, action_num,
                             valid_actions=valid_actions, step_env=step_env,
                             **kwargs)

    ToolAgent.analyze = sentinel_analyze

    # (3) user prompt: drain the pending budget FACT (ONLY on crossing turns) --
    _orig_build_prompt = ToolAgent._build_user_prompt

    def sentinel_build_user_prompt(self, action_num, **kwargs):
        prompt = _orig_build_prompt(self, action_num, **kwargs)
        try:
            sp = getattr(self, "_sentinel_state_path", None)
            if sp is None:
                return prompt
            key = _store_key(sp)
            with _STORES_LOCK:               # dict lookup only under the lock
                st = _STORES.get(key)
            if st is None or st.pending_fact is None:
                return prompt                # no crossing -> zero token cost
            fact = st.pending_fact
            st.pending_fact = None           # one-shot: injected exactly once
            return prompt + "\n" + fact
        except Exception as exc:  # noqa: BLE001 - never break the turn
            log.debug("sentinel prompt injection failed: %s", exc)
            return prompt

    ToolAgent._build_user_prompt = sentinel_build_user_prompt

    _APPLIED = True


# --------------------------------------------------------------------------- #
# entry points
# --------------------------------------------------------------------------- #
def apply(bm: Any = None) -> bool:
    """Install the budget sentinel. Returns True on success (or already
    applied), False on kill switch / any failure -- in which case NOTHING is
    changed (vanilla fallback: stock duck harness).

    Kill switch: SENTINEL_DISABLE=1 -> no-op, returns False.
    """
    if os.environ.get("SENTINEL_DISABLE") == "1":
        log.info("sentinel %s: SENTINEL_DISABLE=1 -> no-op", VERSION)
        return False
    try:
        _apply_patches()
        thresholds = _thresholds()
        if bm is not None and hasattr(bm, "label"):
            bm.label = f"{bm.label}-sentinel-{VERSION}"
        # Runtime banner on stdout: the build log is the only proof of which
        # dataset version actually ran (feedback_kaggle_dataset_code_sync).
        th_str = "/".join(f"{int(round(t * 100))}%" for t in thresholds)
        print(
            f"sentinel {VERSION}: budget sentinel ACTIVE "
            f"(unit=game-envelope; thresholds={th_str}; "
            f"FACT injected on crossing only)",
            flush=True,
        )
        log.info("sentinel %s installed (thresholds=%s)", VERSION, thresholds)
        return True
    except Exception:  # noqa: BLE001 - any failure -> vanilla fallback
        log.exception("[sentinel] apply failed -> stock duck harness (vanilla)")
        return False
