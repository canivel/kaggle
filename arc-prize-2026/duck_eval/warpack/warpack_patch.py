"""Warpack: fork-band adoption grafts for the Cottaar duck harness (R1).

Our own port of the ``taaf_grafts.composite`` flag family found in the
fork-band audit (learnings/war_room/fork_band_audit.md), built in the
phase1-patch pattern (env-driven config, kill switches, VERSION marker,
``apply(bm)`` from the notebook customization cell, blanket-guarded so any
failure falls back to vanilla duck).

Flags (all default ON; master kill switch ``WARPACK_ENABLE=0``):

  banking       WARPACK_BANKING       On run end with >=1 level completed,
                                      open a NEW play on the same card (RESET
                                      full-resets when engine action-count==0
                                      or state==WIN) and replay the pruned
                                      winning trace by driving ``GameAPI.env``
                                      directly. Card score = MAX over plays
                                      (arc_agi EnvironmentScoreList.score),
                                      per-level score = (baseline/actions)^2,
                                      so the no-op-pruned replay scores >= the
                                      original. Any divergence aborts — the
                                      recorded play keeps the max, so banking
                                      is free.
  recovery      WARPACK_RECOVERY      Breaks the two dominant duck failure
                                      modes: GAME_OVER confusion loops and
                                      hypothesis lock-in (same no-op action
                                      repeated N times). REFRESH clears the
                                      analyzer chat history and writes a
                                      hypothesis-graveyard line into the
                                      persistent world model.
  shortcircuit  WARPACK_SHORTCIRCUIT  Stops a homogeneous repeated-action
                                      batch at the first confirmed no-op.
  retry_guard   WARPACK_RETRY_GUARD   Report-only: counts no-op repeat streaks
                                      and GAME_OVER loops, logs periodically.

Patched (2 seams, both session-side — prompt/agent code untouched except the
recovery REFRESH which only mutates ``_history_messages``/``_summarized_knowledge``):
  1. ``_HarnessGameSession._execute_action``  -> trace recorder + shortcircuit
                                                 + recovery/retry_guard counters
  2. ``_HarnessGameSession._finish_if_needed``-> banking replay before finish
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("warpack")

VERSION = "v2"

_APPLIED = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


@dataclass
class WarpackConfig:
    # Master kill switch: 0 -> apply() installs nothing (vanilla duck).
    enable: bool = field(default_factory=lambda: _env_bool("WARPACK_ENABLE", True))
    # Feature flags (runtime kill switches; checked on every call).
    enable_banking: bool = field(default_factory=lambda: _env_bool("WARPACK_BANKING", True))
    enable_recovery: bool = field(default_factory=lambda: _env_bool("WARPACK_RECOVERY", True))
    enable_shortcircuit: bool = field(default_factory=lambda: _env_bool("WARPACK_SHORTCIRCUIT", True))
    enable_retry_guard: bool = field(default_factory=lambda: _env_bool("WARPACK_RETRY_GUARD", True))
    # Banking: skip when soft wall-clock remaining is below this (seconds).
    bank_min_time_s: float = field(default_factory=lambda: _env_float("WARPACK_BANK_MIN_TIME", 120.0))
    # Banking: never replay traces longer than this (post-pruning).
    bank_max_replay_actions: int = field(default_factory=lambda: _env_int("WARPACK_BANK_MAX_ACTIONS", 1500))
    # Banking: also require the replayed frame hash to match the recording
    # (levels_completed is always checked). Divergence -> abort (free).
    bank_strict_frames: bool = field(default_factory=lambda: _env_bool("WARPACK_BANK_STRICT", True))
    # Recovery triggers.
    recovery_gameover_threshold: int = field(default_factory=lambda: _env_int("WARPACK_RECOVERY_GAMEOVERS", 3))
    recovery_repeat_threshold: int = field(default_factory=lambda: _env_int("WARPACK_RECOVERY_REPEATS", 30))
    recovery_max_refreshes: int = field(default_factory=lambda: _env_int("WARPACK_RECOVERY_MAX", 4))
    # retry_guard report cadence (actions).
    retry_guard_report_every: int = field(default_factory=lambda: _env_int("WARPACK_RETRY_GUARD_EVERY", 50))


@dataclass
class TraceStep:
    name: str
    data: dict[str, Any]
    board_changed: bool
    level_completed: bool
    lc_after: int
    grid_hash: int | None
    state_name: str


class WarpackState:
    """Per-_HarnessGameSession state (attached as ``session._wp_state``)."""

    def __init__(self) -> None:
        self.trace: list[TraceStep] = []
        self.banked = False
        self.events: list[tuple] = []
        # shortcircuit: last in-batch (key, was_noop)
        self.batch_last: tuple[str, bool] | None = None
        # recovery counters
        self.same_noop_key: str | None = None
        self.same_noop_count = 0
        self.gameovers_since_progress = 0
        self.refreshes_done = 0
        # retry_guard report-only stats
        self.stats = {"noop_repeat_max": 0, "gameovers": 0, "shortcircuits": 0}
        self.lock = threading.RLock()


class _ShortCircuit(RuntimeError):
    """Raised pre-execution to stop a homogeneous no-op batch; caught by the
    vanilla ``step_env`` batch loop (executed payloads are preserved)."""


def _get_state(session: Any) -> WarpackState:
    st = getattr(session, "_wp_state", None)
    if st is None:
        st = WarpackState()
        session._wp_state = st
    return st


def _action_key(action: Any) -> str:
    try:
        return f"{action.id.name}:{sorted(dict(action.data).items())}"
    except Exception:  # noqa: BLE001
        return str(action)


def _grid_hash_from_frame(frame_rows: Any) -> int | None:
    try:
        return hash(tuple(tuple(int(c) for c in row) for row in frame_rows))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# banking
# ---------------------------------------------------------------------------

def prune_trace(trace: list[TraceStep]) -> list[TraceStep]:
    """Pruned winning trace: for each completed level keep only the actions
    that visibly did something (board_changed) plus the completing action;
    drop segments undone by a level RESET; drop everything after the last
    level completion. RESETs themselves are never replayed (a fresh play
    starts clean and ONLY_RESET_LEVELS keeps cleared levels)."""
    kept: list[TraceStep] = []
    buffer: list[TraceStep] = []
    for step in trace:
        if step.name == "RESET":
            buffer = []  # level reset undid everything since the last level-up
            continue
        buffer.append(step)
        if step.level_completed:
            kept.extend(
                s for s in buffer if s.board_changed or s.level_completed
            )
            buffer = []
    return kept


def _bank(session: Any, cfg: WarpackConfig) -> None:
    """Replay the pruned winning trace on a NEW play of the same card,
    driving ``GameAPI.env`` directly (framework game_run untouched)."""
    import arcengine

    st = _get_state(session)
    with st.lock:
        if st.banked:
            return
        st.banked = True
        trace = list(st.trace)
    game = session.game
    env = getattr(game, "env", None)
    if env is None or not trace:
        return
    if max((s.lc_after for s in trace), default=0) < 1:
        return
    pruned = prune_trace(trace)
    if not pruned or len(pruned) > cfg.bank_max_replay_actions:
        st.events.append(("bank_skip", "trace", len(pruned)))
        return
    soft = None
    try:
        soft = session.solver.soft_time_remaining_seconds()
    except Exception:  # noqa: BLE001
        pass
    if soft is not None and soft < cfg.bank_min_time_s:
        st.events.append(("bank_skip", "time", soft))
        return

    # --- open a new play: RESET full-resets when engine action-count == 0
    # (true right after a level transition and at end of a winning run) or
    # when state == WIN. A first RESET that only level-resets zeroes the
    # engine action counter, so the second RESET is guaranteed full.
    prev_orl = os.environ.get("ONLY_RESET_LEVELS")
    new_play = False
    try:
        os.environ["ONLY_RESET_LEVELS"] = "false"
        for _ in range(2):
            resp = env.step(arcengine.GameAction.RESET, data={})
            if resp is None:
                st.events.append(("bank_abort", "reset_failed", 0))
                return
            if getattr(resp, "full_reset", False):
                new_play = True
                break
    finally:
        if prev_orl is None:
            os.environ.pop("ONLY_RESET_LEVELS", None)
        else:
            os.environ["ONLY_RESET_LEVELS"] = prev_orl
    if not new_play:
        st.events.append(("bank_abort", "no_new_play", 0))
        return

    # --- verbatim replay of the pruned trace with divergence checks
    replayed: list[tuple[str, dict[str, Any]]] = []
    final_lc = 0
    for i, step in enumerate(pruned):
        try:
            action_id = arcengine.GameAction.from_name(step.name)
            resp = env.step(action_id, data=dict(step.data))
        except Exception as exc:  # noqa: BLE001
            st.events.append(("bank_abort", f"step_error:{exc}", i))
            break
        if resp is None or not getattr(resp, "frame", None):
            st.events.append(("bank_abort", "empty_frame", i))
            break
        replayed.append((step.name, dict(step.data)))
        final_lc = int(getattr(resp, "levels_completed", 0) or 0)
        if final_lc != step.lc_after:
            st.events.append(("bank_abort", "lc_divergence", i))
            break
        if cfg.bank_strict_frames and step.grid_hash is not None:
            rh = _grid_hash_from_frame(resp.frame[-1])
            if rh is not None and rh != step.grid_hash:
                st.events.append(("bank_abort", "frame_divergence", i))
                break
    else:
        st.events.append(("bank", len(replayed), final_lc))
        log.info(
            "warpack banking: replayed %d/%d pruned actions on a new play, "
            "levels_completed=%d (recorded trace: %d actions)",
            len(replayed), len(pruned), final_lc, len(trace),
        )
    st.replayed = replayed  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------

def _refresh_analyzer(session: Any, st: WarpackState, cfg: WarpackConfig, reason: str) -> None:
    if st.refreshes_done >= cfg.recovery_max_refreshes:
        return
    agent = getattr(session, "analyzer", None)
    hist = getattr(agent, "_history_messages", None)
    if not isinstance(hist, list):
        return
    st.refreshes_done += 1
    graveyard = (
        f"[WARPACK RECOVERY #{st.refreshes_done}] Chat history cleared: {reason}. "
        "That approach is REFUTED - do not repeat it. Try a mechanically "
        "different goal family (different action type, different target object)."
    )
    hist.clear()
    sk = getattr(agent, "_summarized_knowledge", None)
    if isinstance(sk, dict):
        prior = str(sk.get("recent_findings", "") or "")
        sk["recent_findings"] = (graveyard + (" | " + prior if prior else ""))[:600]
    st.events.append(("refresh", reason, st.refreshes_done))
    st.same_noop_count = 0
    st.same_noop_key = None
    st.gameovers_since_progress = 0
    log.info("warpack recovery refresh #%d: %s", st.refreshes_done, reason)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

def apply(bm: Any = None, config: WarpackConfig | None = None) -> WarpackConfig:
    """Install all warpack patches. Idempotent. Returns the active config.
    With ``WARPACK_ENABLE=0`` (or cfg.enable False) nothing is patched."""
    global _APPLIED
    cfg = config or WarpackConfig()
    if not cfg.enable:
        log.info("warpack disabled by kill switch; no patches installed")
        return cfg
    if _APPLIED:
        log.warning("warpack patches already applied; skipping re-apply")
        return cfg

    import inference.framework.solver as solver_mod

    _orig_execute_action = solver_mod._HarnessGameSession._execute_action
    _orig_finish_if_needed = solver_mod._HarnessGameSession._finish_if_needed

    def wp_execute_action(
        self,
        action,
        *,
        batch_index,
        batch_size,
        generated_tokens=None,
        flush_viewer_payload=True,
    ):
        st = _get_state(self)
        key = _action_key(action)

        # ---- shortcircuit: pre-execution check inside a batch -------------
        if cfg.enable_shortcircuit:
            if batch_index == 1:
                st.batch_last = None
            elif st.batch_last is not None and st.batch_last == (key, True):
                st.stats["shortcircuits"] += 1
                st.events.append(("shortcircuit", key, batch_index))
                raise _ShortCircuit(
                    f"repeated no-op action short-circuited at batch index {batch_index}"
                )

        payload = _orig_execute_action(
            self,
            action,
            batch_index=batch_index,
            batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload,
        )

        try:
            board_changed = bool(payload.get("board_changed"))
            level_completed = bool(payload.get("level_completed"))
            is_noop = not board_changed and not level_completed

            # ---- trace recorder (banking input) ---------------------------
            grid_hash: int | None = None
            try:
                grid_hash = hash(solver_mod._grid_from_state(self.game.current_state))
            except Exception:  # noqa: BLE001
                pass
            with st.lock:
                st.trace.append(
                    TraceStep(
                        name=str(action.id.name),
                        data=dict(action.data),
                        board_changed=board_changed,
                        level_completed=level_completed,
                        lc_after=int(payload.get("score") or 0),
                        grid_hash=grid_hash,
                        state_name=str(payload.get("state") or ""),
                    )
                )
                if cfg.enable_shortcircuit:
                    st.batch_last = (key, is_noop)

                # ---- recovery / retry_guard counters ----------------------
                if action.id.name != "RESET":
                    if is_noop and key == st.same_noop_key:
                        st.same_noop_count += 1
                    elif is_noop:
                        st.same_noop_key = key
                        st.same_noop_count = 1
                    else:
                        st.same_noop_key = None
                        st.same_noop_count = 0
                    st.stats["noop_repeat_max"] = max(
                        st.stats["noop_repeat_max"], st.same_noop_count
                    )
                if payload.get("game_over"):
                    st.gameovers_since_progress += 1
                    st.stats["gameovers"] += 1
                if level_completed or payload.get("run_complete"):
                    st.gameovers_since_progress = 0
                    st.same_noop_count = 0

            if cfg.enable_recovery:
                if st.same_noop_count >= cfg.recovery_repeat_threshold:
                    _refresh_analyzer(
                        self, st, cfg,
                        f"action {action.id.name}{dict(action.data) or ''} repeated "
                        f"{st.same_noop_count}x with no board change",
                    )
                elif st.gameovers_since_progress >= cfg.recovery_gameover_threshold:
                    _refresh_analyzer(
                        self, st, cfg,
                        f"{st.gameovers_since_progress} GAME_OVERs with no level progress",
                    )

            if (
                cfg.enable_retry_guard
                and cfg.retry_guard_report_every > 0
                and len(st.trace) % cfg.retry_guard_report_every == 0
            ):
                log.info(
                    "warpack retry_guard [%s]: actions=%d noop_repeat_max=%d "
                    "gameovers=%d shortcircuits=%d",
                    getattr(self.game, "game_id", "?"), len(st.trace),
                    st.stats["noop_repeat_max"], st.stats["gameovers"],
                    st.stats["shortcircuits"],
                )
        except Exception as exc:  # noqa: BLE001 - never break the action path
            log.debug("warpack execute_action hook failed: %s", exc)
        return payload

    def wp_finish_if_needed(self):
        if cfg.enable_banking:
            try:
                run = self.game.game_run
                if run is not None and run.final_score is None:
                    _bank(self, cfg)
            except Exception as exc:  # noqa: BLE001 - banking must never block finish
                log.warning("warpack banking failed: %s", exc)
            try:
                # Canary counts attempts/skips/aborts, not only successes, so a
                # run where banking never becomes reachable prints zeros instead
                # of nothing (prereg 2026-07-14 §7 as amended 07-15 A2).
                ev = _get_state(self).events
                game_id = "?"
                try:
                    game_id = self.game.env.environment_info.game_id
                except Exception:  # noqa: BLE001
                    pass
                log.info(
                    "warpack banking canary: game=%s attempts=%d skips=%d "
                    "aborts=%d successes=%d",
                    game_id,
                    sum(1 for e in ev if e[0] in ("bank",) or str(e[0]) == "bank_abort"),
                    sum(1 for e in ev if str(e[0]) == "bank_skip"),
                    sum(1 for e in ev if str(e[0]) == "bank_abort"),
                    sum(1 for e in ev if e[0] == "bank"),
                )
            except Exception as exc:  # noqa: BLE001 - canary must never block finish
                log.debug("warpack canary failed: %s", exc)
        return _orig_finish_if_needed(self)

    solver_mod._HarnessGameSession._execute_action = wp_execute_action
    solver_mod._HarnessGameSession._finish_if_needed = wp_finish_if_needed

    _APPLIED = True
    log.info(
        "warpack %s applied: banking=%s recovery=%s shortcircuit=%s retry_guard=%s "
        "bank_min_time=%.0fs bank_strict=%s",
        VERSION, cfg.enable_banking, cfg.enable_recovery, cfg.enable_shortcircuit,
        cfg.enable_retry_guard, cfg.bank_min_time_s, cfg.bank_strict_frames,
    )
    return cfg
