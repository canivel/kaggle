"""R1 attempt scheduler: graft restart/park scheduling onto the Cottaar duck harness.

Implements learnings/path_forward_v2_2026-07-13.md SSR1 (survived panel rounds
5+6). Semantics, verbatim from the plan:

  - Trigger: within a game, if levels_completed == 0 at SCHED_RESTART_AT (=90,
    null10 p90 of time-to-first-level) actions since episode start (per-attempt
    counter, resets on scheduler restart) -> RESET to a fresh episode.
  - Cap: maximum SCHED_MAX_RESTARTS (=2) restarts per game (3 attempts),
    counted by a cumulative attempt counter that never resets.
  - Park: after the cap, if still lc == 0 at the trigger point, the game stops
    consuming analyzer turns (session.should_stop() -> True). Precedence: park
    dominates restart. Dead games are bounded at ~272 actions
    (90 + RESET + 90 + RESET + 90) instead of grinding to the wall.
  - NO context injection whatsoever: the prompt/context path is untouched.
    The injected RESET lands in the ordinary action history exactly like the
    harness's existing auto-reset after GAME_OVER does today.

Applied by calling ``apply()`` from the duckfork notebook's customization-hook
cell (see hook_cell.py), after ``bm`` is unpickled and the bundled sources are
on ``sys.path``. Both patches are class-level monkeypatches on
``inference.framework.solver._HarnessGameSession``, so they affect the
already-pickled ``bm.solver`` (methods resolve at call time).

What gets patched (see PATCH_NOTES.md for the full map):
  1. _HarnessGameSession._execute_action  -> post-action per-attempt counter;
                                             fires the RESET / park decision
  2. _HarnessGameSession.should_stop      -> parked sessions stop immediately

NOT patched: ToolAgent (analyze / _build_user_prompt / message trimming),
python sandbox, tool descriptions -- nothing the model reads is changed.

Kill switch: SCHED_ENABLE=0 makes ``apply()`` a no-op (no patches installed).
Thresholds are game-agnostic null10 percentiles; no game-ID-keyed logic.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("sched")

# Patch-layer version. v1 = R1 attempt scheduler (restart @ 90 / cap 2 / park).
VERSION = "v1"

_APPLIED = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
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
class SchedulerConfig:
    # Kill switch: SCHED_ENABLE=0 -> apply() installs nothing (vanilla duck).
    enable: bool = field(default_factory=lambda: _env_bool("SCHED_ENABLE", True))
    # Per-attempt no-level trigger: restart when levels_completed == 0 after
    # this many actions since episode start (null10 p90 time-to-first-level).
    restart_at: int = field(default_factory=lambda: _env_int("SCHED_RESTART_AT", 90))
    # Cumulative restart cap per game (never resets). After the cap the game
    # is parked at the next trigger point instead of restarted.
    max_restarts: int = field(default_factory=lambda: _env_int("SCHED_MAX_RESTARTS", 2))

    def __post_init__(self) -> None:
        self.restart_at = max(1, int(self.restart_at))
        self.max_restarts = max(0, int(self.max_restarts))


class SchedulerState:
    """Per-game (per _HarnessGameSession) scheduler state."""

    def __init__(self) -> None:
        # action_count at the start of the current attempt (per-attempt
        # counter base; moves forward only on a scheduler restart).
        self.attempt_start = 0
        # Cumulative restarts fired for this game. NEVER resets.
        self.restarts_done = 0
        self.parked = False
        # Re-entrancy guard while the scheduler's own RESET executes.
        self.in_scheduler_reset = False
        # Forensics: ("restart"|"park", total_action_count, per_attempt_actions)
        self.events: list[tuple[str, int, int]] = []


def _get_state(session: Any) -> SchedulerState:
    state = getattr(session, "_sched_state", None)
    if state is None:
        state = SchedulerState()
        session._sched_state = state
    return state


def apply(bm: Any = None, config: SchedulerConfig | None = None) -> SchedulerConfig:
    """Apply the attempt-scheduler monkeypatches. Idempotent.

    Returns the active config. With the kill switch (SCHED_ENABLE=0 or
    ``config.enable = False``) nothing is patched and the harness runs vanilla.
    """
    global _APPLIED
    cfg = config or SchedulerConfig()
    if not cfg.enable:
        log.info("sched %s: SCHED_ENABLE=0 kill switch -- no patches installed", VERSION)
        print(f"[sched] {VERSION}: kill switch active (SCHED_ENABLE=0), vanilla scheduler")
        return cfg
    if _APPLIED:
        log.warning("sched patches already applied; skipping re-apply")
        return cfg

    import arcengine
    import inference.framework.solver as solver_mod

    _orig_execute_action = solver_mod._HarnessGameSession._execute_action
    _orig_should_stop = solver_mod._HarnessGameSession.should_stop

    # ------------------------------------------------------------------
    # (2) should_stop: a parked game stops consuming analyzer turns
    # ------------------------------------------------------------------
    def sched_should_stop(self):
        st = getattr(self, "_sched_state", None)
        if st is not None and st.parked:
            return True
        return _orig_should_stop(self)

    solver_mod._HarnessGameSession.should_stop = sched_should_stop

    # ------------------------------------------------------------------
    # (1) _execute_action: per-attempt counting + restart / park decision
    # ------------------------------------------------------------------
    def _maybe_restart_or_park(session, action, flush_viewer_payload) -> None:
        st = _get_state(session)
        if st.parked or st.in_scheduler_reset:
            return
        if action.id == arcengine.GameAction.RESET:
            # A RESET we didn't fire (harness auto-reset after GAME_OVER).
            # Same attempt continues: the per-attempt counter does NOT reset,
            # so a game-over-thrashing attempt still reaches the trigger.
            return
        run = session.game.game_run
        levels = int(
            run.levels_completed
            if run is not None
            else session.game.current_state.levels_completed
        )
        if levels > 0:
            # Progressing game: scheduler never fires (trigger needs lc == 0).
            return
        per_attempt = session.action_count - st.attempt_start
        if per_attempt < cfg.restart_at:
            return
        game_id = run.game_id if run is not None else str(session.game_index)
        # Precedence: park dominates restart.
        if st.restarts_done >= cfg.max_restarts:
            st.parked = True
            st.events.append(("park", session.action_count, per_attempt))
            log.info(
                "sched PARK %s: lc=0 on attempt %d after %d actions "
                "(cap %d restarts reached; total actions %d)",
                game_id, st.restarts_done + 1, per_attempt,
                cfg.max_restarts, session.action_count,
            )
            print(
                f"[sched] PARK {game_id}: lc=0 on attempt {st.restarts_done + 1} "
                f"after {per_attempt} actions (cap {cfg.max_restarts} restarts; "
                f"total actions {session.action_count})"
            )
            return
        # Don't fire into a terminal / stopping session (wall, cancel, WIN).
        if run is not None and run.state != "playing":
            return
        if _orig_should_stop(session):
            return
        st.in_scheduler_reset = True
        try:
            reset = arcengine.ActionInput(id=arcengine.GameAction.RESET, data={})
            _orig_execute_action(
                session,
                reset,
                batch_index=1,
                batch_size=1,
                generated_tokens=0,
                flush_viewer_payload=flush_viewer_payload,
            )
        finally:
            st.in_scheduler_reset = False
        st.restarts_done += 1
        st.attempt_start = session.action_count
        st.events.append(("restart", session.action_count, per_attempt))
        log.info(
            "sched RESTART %s: lc=0 at %d actions since episode start -> fresh "
            "episode (restart %d/%d; total actions %d)",
            game_id, per_attempt, st.restarts_done, cfg.max_restarts,
            session.action_count,
        )
        print(
            f"[sched] RESTART {game_id}: lc=0 at {per_attempt} actions since "
            f"episode start -> fresh episode (restart {st.restarts_done}/"
            f"{cfg.max_restarts}; total actions {session.action_count})"
        )

    def sched_execute_action(
        self,
        action,
        *,
        batch_index,
        batch_size,
        generated_tokens=None,
        flush_viewer_payload=True,
    ):
        payload = _orig_execute_action(
            self,
            action,
            batch_index=batch_index,
            batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload,
        )
        try:
            _maybe_restart_or_park(self, action, flush_viewer_payload)
        except Exception as exc:  # noqa: BLE001 - never break the action path
            log.warning("sched post-action hook failed: %s", exc)
        return payload

    solver_mod._HarnessGameSession._execute_action = sched_execute_action

    _APPLIED = True
    log.info(
        "sched %s patches applied: restart_at=%d max_restarts=%d",
        VERSION, cfg.restart_at, cfg.max_restarts,
    )
    print(
        f"[sched] {VERSION} patches applied: restart_at={cfg.restart_at} "
        f"max_restarts={cfg.max_restarts} (park after cap)"
    )
    return cfg
