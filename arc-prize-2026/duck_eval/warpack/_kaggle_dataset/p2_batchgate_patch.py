"""P2 - VERIFIED-PLAN BATCH GATING (a CONSTRAIN mechanism, not an INFORM one).

Spec: ``learnings/war_room/harness_diff_2026-08-13.md`` sec4 + the 08-13 EFFNOTE
result (``duck_eval/warpack/effnote_push_report_2026-08-13.md`` sec6.7).
Prereg: ``learnings/war_room/p2_prereg_2026-08-13.md``.

*** STATUS 2026-08-13: BUILT AND ENGINE-TESTED, **NEVER PUSHED**. ***
The pre-push replay (``duck_eval/warpack/p2_replay.py``, 30 rules x 4 recorded
runs) REFUSED the mechanism as briefed: a position cap of 1-2 on stall-gated
batches deletes 5 LEVEL-COMPLETING actions across ar25/sp80/tu93, and once those
losses are scored honestly its multiplier is x0.9457, not x1.0574. The safe
redesign that ships here (the stale-state run cap) is worth x1.003-x1.012 --
inside the noise, and a strict subset of P1 mechanism B, which already ran live
on 08-12 and was NO-PROMOTE. This module is therefore **flag-gated OFF and
inert**; arming it for a scored run requires a NEW prereg that answers
p2_prereg_2026-08-13.md sec3 and sec6 head-on.

WHY THIS EXISTS - the two dead INFORM arms.
Two mechanisms have now failed identically on this rail:

  P1 mechanism C (08-12)  hand the agent the ground truth it already holds
                          -> delivered on 96.3% of turns
                          -> dead-reissue rate INSIDE the control spread.
  EFFNOTE (08-13)         hand it the scoring rule + its own live action count
                          + a firing stall alarm
                          -> delivered on 93.8% of turns, 284 chars
                          -> post-stall revisit 0.4971, INSIDE the control
                             spread 0.3986-0.5487.

**The agent does not act on runner-supplied context.** That closes the whole
runner-side PUSH lane on measured evidence.

But EFFNOTE measured one thing that is actionable without the agent's consent:

    B4  mean actions per STALL turn   ARM 11.11   control max 7.28
    B4c mean actions per non-stall    ARM  3.92   control range 3.46-3.94

On exactly the turns whose note said "commit to the shortest sequence that
tests it", the agent **batched HARDER** - ~11 actions fired blind into a board
it had just been told was cycling - while its non-stall batch size was
untouched. Those long blind batches are also the 17.6% blind-batch tail of the
08-12 diagnosis. P2 does not ask. P2 **caps**.

THE MECHANISM.
When the runner's own stall detectors are active at the moment a batch is
issued (the SAME pure functions EFFNOTE shipped - imported, never
re-implemented), the batch is executed under a cap. When the cap fires, the
remainder of the batch is aborted and control returns to the model with the
board it can actually see. Outside stall conditions batches pass through
completely untouched. Game-agnostic: no game id is read, no per-game table
exists, no baseline of any kind is consulted.

THE SAFETY CONSTRAINT THAT SHAPES THE RULE.
The level-completing batches in our recorded traces **OPEN WITH RE-TRAVERSAL**:

    sp80 L1   RIGHT(revisit) x3   SPACE  <- COMPLETES   (4 actions)
    tu93 L1   DOWN(revisit) ...   DOWN   <- COMPLETES
    ar25 L1   LEFT(revisit) x5  DOWN x9  DOWN <- COMPLETES  (15 actions)

A naive POSITION cap deletes level-completing actions, and an abort-on-first-
revisit deletes them too - that is exactly the wall that forced
``P1_ABORT_REVISIT=0`` on 08-12. So the shipped rule is neither.

THE SHIPPED RULE - a **k-tolerant STALE-STATE RUN cap**. On a gated batch,
abort the remainder once ``P2_STALE_RUN`` CONSECUTIVE executed actions have
failed to reach a board state that is NEW for this level. Re-traversal is
therefore ALLOWED, up to a tolerance the offline replay picked so that **zero**
level-completing actions are cut on any recorded run; what is not allowed is
wandering indefinitely through states the runner has already seen.

Note the containment: a byte-identical no-op necessarily lands on an
already-visited state, so the no-op run is a special case of the stale run.
``P2_NOOP_RUN`` is kept as a separate, tighter knob for the pure-no-op case.

``P2_CAP`` (the raw position cap the first draft proposed) survives as an
**ablation handle only**, defaulted OFF; the replay shows it destroys level
completions at every value that saves anything.

HOUSE PATTERN (mirrors p1_suppressor_patch / effnote_patch / animation_patch):
VERSION marker, arm flag + kill switch, blanket-guarded ``apply()``, runtime
banner, ``bm.label`` stamp, greppable events, canary counters, per-session state
(no globals, no locks, no threads), **vanilla fallback on ANY failure**.

Seams (4):
  1. ``solver._HarnessGameSession.play``               -> per-game state+canary
  2. ``solver._HarnessGameSession.step_env``           -> arm the gate
  3. ``solver._HarnessGameSession._execute_action``    -> enforce the cap
  4. ``solver._HarnessGameSession._normalize_actions`` -> count REQUESTED
     actions (read-only; the denominator of the delivery rate)

The abort is raised from ``_execute_action`` BEFORE the engine is touched and is
caught by the STOCK ``step_env`` loop (``except Exception ... break``), so
aggregation, viewer payloads and the run bookkeeping are all the vanilla code
paths - exactly the pattern P1 mechanism B ran live on 08-12 (75 abort events,
``errors=0``).

LEGALITY: no game id, no baseline read, no ``metadata.json`` rglob, no duplicate
-game replay gate. Zero LLM calls, zero GPU, zero new tools.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Sequence

log = logging.getLogger("p2gate")

VERSION = "v1"
_EVENT_V = "1"

_APPLIED = False

# The detectors are IMPORTED from the shipped EFFNOTE module, never
# re-implemented -- arm and replay and control all drive the identical code.
try:  # pragma: no cover - exercised by smoke S3
    import effnote_patch as _EN
except Exception:  # noqa: BLE001 - pragma: no cover
    _EN = None


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
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


def _flag_on() -> bool:
    return os.environ.get("P2_BATCHGATE", "").strip() == "1"


def _kill_switch() -> bool:
    return os.environ.get("P2_BATCHGATE_DISABLE", "").strip() == "1"


class Config:
    """Read live on every use so every knob is a runtime kill switch."""

    # -- the cap ----------------------------------------------------------
    @property
    def stale_run(self) -> int:
        """THE SHIPPED RULE. Abort the rest of a GATED batch once this many
        CONSECUTIVE executed actions have failed to reach a board state that is
        NEW for the current level. 0 disables.

        k-tolerant on purpose: the recorded level-completing batches open with
        re-traversal (ar25 L1 = 5 stale actions before the productive run), so
        the value is chosen by ``p2_replay.py`` as the smallest k that cuts
        ZERO level-completing actions on every recorded run."""
        return max(0, _env_int("P2_STALE_RUN", 6))

    @property
    def noop_run(self) -> int:
        """Tighter companion knob: abort once this many CONSECUTIVE actions
        have left the board BYTE-IDENTICAL. A no-op is always stale, so this is
        a strict special case of ``stale_run``. 0 disables."""
        return max(0, _env_int("P2_NOOP_RUN", 0))

    @property
    def cap(self) -> int:
        """Raw position cap on a GATED batch (ABLATION HANDLE, default OFF).

        The offline replay proves this deletes level-completing actions at
        every value that saves anything - the recorded winning batches open
        with re-traversal and run 4-15 actions long. Shipped OFF; turning it on
        for a scored run requires a new prereg."""
        return max(0, _env_int("P2_CAP", 0))

    @property
    def persist(self) -> bool:
        """Carry the stale / no-op run ACROSS batches within a level instead of
        restarting it at every ``step_env`` call.

        This matters because of what the 08-13 replay found: the agent does not
        issue one long batch, it issues 2-4 SHORT batches per analysis turn. A
        per-batch run counter is therefore reset by the agent's own tool-call
        cadence and almost never reaches k."""
        return _env_bool("P2_PERSIST", True)

    @property
    def min_batch(self) -> int:
        """Never gate a batch smaller than this (a 1-action batch has no
        remainder to abort, so this is a floor of 2 by construction)."""
        return max(2, _env_int("P2_MIN_BATCH", 2))

    # -- which detectors arm the gate -------------------------------------
    @property
    def gate_net_zero(self) -> bool:
        return _env_bool("P2_GATE_NET_ZERO", True)

    @property
    def gate_stagnation(self) -> bool:
        return _env_bool("P2_GATE_STAGNATION", True)

    @property
    def gate_revisit(self) -> bool:
        return _env_bool("P2_GATE_REVISIT", True)

    @property
    def gate_always(self) -> bool:
        """ABLATION HANDLE: gate every batch, stall or not. Default OFF -- the
        whole thesis is that the cap applies only where the runner can already
        see waste."""
        return _env_bool("P2_GATE_ALWAYS", False)

    # -- detector thresholds ----------------------------------------------
    # Stated EXPLICITLY rather than inherited from EFFNOTE's Config, so that a
    # P2 run is not silently re-tuned by an EFFNOTE_* variable set for another
    # arm. The values ARE the EFFNOTE v1 shipped defaults.
    @property
    def min_roundtrip(self) -> int:
        return max(2, _env_int("P2_MIN_ROUNDTRIP", 6))

    @property
    def stagnation_min(self) -> int:
        return max(2, _env_int("P2_STAGNATION_MIN", 8))

    @property
    def revisit_min(self) -> int:
        return max(2, _env_int("P2_REVISIT_MIN", 4))

    @property
    def window(self) -> int:
        return max(8, _env_int("P2_WINDOW", 240))

    @property
    def revisit_window(self) -> int:
        return max(8, _env_int("P2_REVISIT_WINDOW", 120))

    # -- log volume --------------------------------------------------------
    @property
    def max_events(self) -> int:
        return max(0, _env_int("P2_MAX_EVENTS", 400))


CFG = Config()


class P2BatchAbort(Exception):
    """Raised from ``_execute_action`` to end a gated batch early.

    The STOCK ``step_env`` loop catches it (``except Exception as exc``) and,
    because at least one action has already executed, breaks with
    ``stop_reason='action_error'`` and aggregates normally. The step_env seam
    then relabels it. Never escapes to the play loop."""


# --------------------------------------------------------------------------- #
# pure gate logic (no I/O, no LLM, no game id - unit-testable standalone)
# --------------------------------------------------------------------------- #
def detectors_for(
    current_frame: Any,
    history_frames: Sequence[Any],
) -> dict[str, int]:
    """Run the three SHIPPED EFFNOTE detectors at P2's thresholds.

    Returns a dict of the detectors that fired, name -> magnitude. Empty means
    no stall. Pure; raises nothing (a missing EFFNOTE module yields ``{}``,
    which disarms the gate - fail-open, never fail-closed)."""
    if _EN is None or current_frame is None or not history_frames:
        return {}
    fired: dict[str, int] = {}
    if CFG.gate_net_zero:
        nz = _EN.detect_net_zero_cycle(
            current_frame, history_frames,
            min_roundtrip_actions=CFG.min_roundtrip, window=CFG.window)
        if nz:
            fired["nz"] = int(nz)
    if CFG.gate_stagnation:
        stag = _EN.detect_stagnation(
            current_frame, history_frames,
            min_run=CFG.stagnation_min, window=CFG.window)
        if stag:
            fired["stag"] = int(stag)
    if CFG.gate_revisit:
        rev = _EN.count_recent_revisits(
            current_frame, history_frames, window=CFG.revisit_window)
        if rev and int(rev) >= CFG.revisit_min:
            fired["rev"] = int(rev)
    return fired


def should_abort_remainder(
    *,
    executed: int,
    consecutive_noops: int,
    consecutive_stale: int,
    cap: int | None = None,
    noop_run: int | None = None,
    stale_run: int | None = None,
) -> str | None:
    """THE CAP, as a pure predicate. Called BEFORE executing the next action of
    a gated batch that has already executed ``executed`` >= 1 actions.

    Returns the reason string, or ``None`` to let the action through.

      ``stale_run`` the SHIPPED rule: abort once the batch has produced this
                    many CONSECUTIVE actions that reached NO board state new to
                    this level. Any action that discovers a new state resets the
                    run, so a batch that is making progress is never truncated.
      ``noop_run``  tighter companion: consecutive BYTE-IDENTICAL outcomes.
      ``cap``       raw position cap. ABLATION ONLY, defaults 0 (off): the
                    replay proves it cuts level-completing actions.

    Pure. No I/O."""
    lim_cap = CFG.cap if cap is None else cap
    lim_noop = CFG.noop_run if noop_run is None else noop_run
    lim_stale = CFG.stale_run if stale_run is None else stale_run
    if executed < 1:
        return None
    if lim_noop > 0 and consecutive_noops >= lim_noop:
        return f"noop_run>={lim_noop}"
    if lim_stale > 0 and consecutive_stale >= lim_stale:
        return f"stale_run>={lim_stale}"
    if lim_cap > 0 and executed >= lim_cap:
        return f"cap>={lim_cap}"
    return None


def gate_reason(fired: dict[str, int]) -> str:
    return ",".join(f"{k}={v}" for k, v in sorted(fired.items())) or "none"


def board_fingerprint(grid: Any) -> str:
    """Cheap, order-stable digest of a board. Used ONLY to answer "is this
    state new for this level?" -- never compared across games, never persisted.
    Pure."""
    try:
        return hashlib.blake2b(
            repr(grid).encode("utf-8", "replace"), digest_size=16
        ).hexdigest()
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# per-session state + canary
# --------------------------------------------------------------------------- #
class P2State:
    """Per-game counters. One instance per ``_HarnessGameSession``; no globals,
    no locks, no threads (each game runs in its own thread)."""

    __slots__ = (
        "game", "batches", "gated_batches", "gated_multi", "requested",
        "executed", "saved", "aborts", "fire_nz", "fire_stag", "fire_rev",
        "errors", "events", "gated_actions", "ungated_actions",
        "batch_gated", "batch_fired", "batch_executed", "batch_noop_run",
        "batch_stale_run", "batch_size", "batch_saved", "batch_reason",
        "in_batch", "max_saved_one_batch", "visited", "level",
        "gated_requested", "ungated_requested", "gated_batch_sizes",
    )

    def __init__(self, game: str) -> None:
        self.game = game
        self.batches = 0
        self.gated_batches = 0
        self.gated_multi = 0
        self.requested = 0
        self.executed = 0
        self.saved = 0
        self.aborts = 0
        self.fire_nz = 0
        self.fire_stag = 0
        self.fire_rev = 0
        self.errors = 0
        self.events = 0
        self.gated_actions = 0
        self.ungated_actions = 0
        self.gated_requested = 0
        self.ungated_requested = 0
        self.gated_batch_sizes = 0
        self.max_saved_one_batch = 0
        # per-LEVEL visited board states (the "is this state new?" memory).
        self.visited: set[str] = set()
        self.level = None
        self._reset_batch()

    def _reset_batch(self) -> None:
        self.batch_gated = False
        self.batch_fired = {}
        self.batch_executed = 0
        if not CFG.persist:
            # per-batch counters: the agent's tool-call cadence resets them
            self.batch_noop_run = 0
            self.batch_stale_run = 0
        else:
            self.batch_noop_run = getattr(self, "batch_noop_run", 0) or 0
            self.batch_stale_run = getattr(self, "batch_stale_run", 0) or 0
        self.batch_size = 0
        self.batch_saved = 0
        self.batch_reason = ""
        self.in_batch = False

    def sync_level(self, level: Any) -> None:
        """A new level is a new state space: the visited set MUST reset, or a
        stale run would be measured against another level's states."""
        if level != self.level:
            self.level = level
            self.visited = set()
            # a new level is a new state space: a run carried across batches
            # may not survive the boundary
            self.batch_stale_run = 0
            self.batch_noop_run = 0

    def observe(self, fingerprint: str, board_changed: bool) -> None:
        """Fold one executed action into the batch's run counters."""
        self.batch_noop_run = 0 if board_changed else self.batch_noop_run + 1
        if fingerprint and fingerprint not in self.visited:
            self.visited.add(fingerprint)
            self.batch_stale_run = 0
        else:
            self.batch_stale_run += 1


def _emit(kind: str, st: P2State, detail: str) -> None:
    if st.events >= CFG.max_events and kind != "game_end":
        return
    st.events += 1
    print(
        f"P2 v={_EVENT_V} kind={kind} game={st.game} "
        f"batches={st.batches} gated={st.gated_batches} aborts={st.aborts} "
        f"saved={st.saved} executed={st.executed} errors={st.errors} {detail}",
        flush=True,
    )


CANARY: dict[str, dict[str, Any]] = {}


def canary_report() -> dict[str, Any]:
    """End-of-run canary. Prints one greppable line and returns the numbers.

    K-G0  gate armed at all (``gated``/``batches``, and on how many games)
    K-G1  detector sanity (per-detector fire counts + distinct games)
    K-G2  MECHANISM DELIVERY: ``saved``/``requested`` -- the primary endpoint
    K-G3  the gate is not a nag (``gated``/``batches`` <= 40%)
    K-G4  ``errors``
    """
    games = sorted(CANARY)
    keys = ("batches", "gated_batches", "gated_multi", "requested", "executed",
            "saved", "aborts", "fire_nz", "fire_stag", "fire_rev", "errors",
            "gated_actions", "ungated_actions", "gated_requested",
            "ungated_requested")
    tot = {k: sum(int(CANARY[g].get(k, 0)) for g in games) for k in keys}
    ngames = {
        k: sum(1 for g in games if int(CANARY[g].get(k, 0)) > 0)
        for k in ("fire_nz", "fire_stag", "fire_rev", "aborts")
    }
    batches = tot["batches"] or 1
    requested = tot["requested"] or 1
    ungated_batches = max(0, tot["batches"] - tot["gated_batches"])
    report = {
        "version": VERSION,
        "games": len(games),
        "gate_rate": tot["gated_batches"] / batches,
        "saved_rate": tot["saved"] / requested,
        # G1 -- the BEHAVIOURAL endpoint (mean actions REQUESTED per gated
        # batch) and its within-run ungated counterpart.
        "G1_gated_batch_size": (tot["gated_requested"] / tot["gated_batches"]
                                if tot["gated_batches"] else 0.0),
        "G1c_ungated_batch_size": (tot["ungated_requested"] / ungated_batches
                                   if ungated_batches else 0.0),
        "games_net_zero": ngames["fire_nz"],
        "games_stagnation": ngames["fire_stag"],
        "games_revisit": ngames["fire_rev"],
        "games_aborted": ngames["aborts"],
        "stale_run": CFG.stale_run,
        "noop_run": CFG.noop_run,
        "cap": CFG.cap,
        **tot,
    }
    print(
        f"P2 CANARY v={_EVENT_V} version={VERSION} games={len(games)} "
        f"batches={tot['batches']} gated={tot['gated_batches']} "
        f"gate_rate={report['gate_rate']:.4f} "
        f"gated_multi={tot['gated_multi']} aborts={tot['aborts']} "
        f"abort_games={ngames['aborts']} "
        f"requested={tot['requested']} executed={tot['executed']} "
        f"saved={tot['saved']} saved_rate={report['saved_rate']:.4f} "
        f"G1={report['G1_gated_batch_size']:.4f} "
        f"G1c={report['G1c_ungated_batch_size']:.4f} "
        f"nz={tot['fire_nz']}/{ngames['fire_nz']}g "
        f"stag={tot['fire_stag']}/{ngames['fire_stag']}g "
        f"rev={tot['fire_rev']}/{ngames['fire_rev']}g "
        f"gated_actions={tot['gated_actions']} "
        f"ungated_actions={tot['ungated_actions']} "
        f"stale_run={CFG.stale_run} noop_run={CFG.noop_run} cap={CFG.cap} "
        f"errors={tot['errors']}",
        flush=True,
    )
    return report


# --------------------------------------------------------------------------- #
# live-object helpers
# --------------------------------------------------------------------------- #
def _game_label(session: Any) -> str:
    for getter in (
        lambda: session.game.game_run.game_id,
        lambda: session.game.env.environment_info.game_id,
        lambda: session.game.game_id,
    ):
        try:
            value = getter()
            if value:
                return str(value)
        except Exception:  # noqa: BLE001
            continue
    return "?"


def frames_of(session: Any) -> list[Any]:
    """The per-game frame history the agent itself sees, oldest first."""
    entries = getattr(session, "history_entries", None) or []
    return [getattr(e, "frame", None) for e in entries]


def _batch_len(arguments: Any) -> int:
    try:
        actions = arguments.get("actions")
    except Exception:  # noqa: BLE001
        return 1
    if isinstance(actions, list):
        return len(actions)
    return 1


# --------------------------------------------------------------------------- #
# patches
# --------------------------------------------------------------------------- #
def _apply_patches() -> int:
    global _APPLIED
    if _APPLIED:
        return 0

    import inference.framework.solver as solver_mod

    patched = 0

    def _state(session: Any) -> P2State:
        st = getattr(session, "_p2_state", None)
        if st is None:
            st = P2State(_game_label(session))
            session._p2_state = st
        return st

    # --- seam 1: per-game state + canary ----------------------------------
    _orig_play = solver_mod._HarnessGameSession.play

    def p2_play(self):
        st = _state(self)
        try:
            return _orig_play(self)
        finally:
            try:
                CANARY[st.game] = {
                    "batches": st.batches,
                    "gated_batches": st.gated_batches,
                    "gated_multi": st.gated_multi,
                    "requested": st.requested,
                    "executed": st.executed,
                    "saved": st.saved,
                    "aborts": st.aborts,
                    "fire_nz": st.fire_nz,
                    "fire_stag": st.fire_stag,
                    "fire_rev": st.fire_rev,
                    "errors": st.errors,
                    "gated_actions": st.gated_actions,
                    "ungated_actions": st.ungated_actions,
                    "gated_requested": st.gated_requested,
                    "ungated_requested": st.ungated_requested,
                }
                _emit("game_end", st,
                      f"max_saved_one_batch={st.max_saved_one_batch}")
            except Exception:  # noqa: BLE001
                pass

    solver_mod._HarnessGameSession.play = p2_play
    patched += 1

    # --- seam 2: arm the gate for this batch ------------------------------
    _orig_step_env = solver_mod._HarnessGameSession.step_env

    def p2_step_env(self, arguments):
        st = _state(self)
        st._reset_batch()
        st.in_batch = True
        size = 1
        try:
            size = _batch_len(arguments)
            st.batch_size = size
            st.batches += 1
            frames = frames_of(self)
            current = frames[-1] if frames else None
            if CFG.gate_always:
                fired = {"always": 1}
            else:
                fired = detectors_for(current, frames)
            if fired:
                st.fire_nz += 1 if "nz" in fired else 0
                st.fire_stag += 1 if "stag" in fired else 0
                st.fire_rev += 1 if "rev" in fired else 0
                st.gated_batches += 1
                # G1, the BEHAVIOURAL endpoint: how many actions the model
                # REQUESTS on a stall-gated batch. The cap truncates execution;
                # it does not shrink the request, so this stays a free measure
                # of what the agent chose to do.
                st.gated_requested += size
            else:
                st.ungated_requested += size
            # A batch below the floor has no remainder worth aborting.
            st.batch_gated = bool(fired) and size >= CFG.min_batch
            st.batch_fired = fired
            if st.batch_gated:
                st.gated_multi += 1
            # Seed the level's visited set with the state the batch starts
            # from, so the very first action can already be "stale".
            # Level identity is ALWAYS ``levels_completed`` -- the same key the
            # post-action hook uses, so the set never resets spuriously.
            st.sync_level(int(self.game.current_state.levels_completed))
            if current is not None:
                st.visited.add(board_fingerprint(getattr(current, "grid", None)))
        except Exception as exc:  # noqa: BLE001 - the action path may NEVER break
            st.errors += 1
            st.batch_gated = False
            log.debug("p2: gate arming failed: %s", exc)

        try:
            payload = _orig_step_env(self, arguments)
        finally:
            st.in_batch = False

        try:
            if st.batch_gated:
                st.gated_actions += st.batch_executed
            else:
                st.ungated_actions += st.batch_executed
            if isinstance(payload, dict) and st.batch_saved:
                payload["p2_saved"] = st.batch_saved
                payload["p2_gate"] = gate_reason(st.batch_fired)
                payload["stop_reason"] = "p2_batch_gate"
                payload["stopped_early"] = True
                payload.pop("error", None)
                payload["p2_note"] = (
                    f"P2: the runner stopped this batch after "
                    f"{st.batch_executed} of {st.batch_size} actions "
                    f"({st.batch_reason}); the board history showed a stall "
                    f"({gate_reason(st.batch_fired)}) when the batch was "
                    f"issued. The remaining {st.batch_saved} action(s) were "
                    f"NOT executed and NOT charged. Re-plan from the board you "
                    f"can now see."
                )
        except Exception as exc:  # noqa: BLE001
            st.errors += 1
            log.debug("p2: step_env bookkeeping failed: %s", exc)
        return payload

    solver_mod._HarnessGameSession.step_env = p2_step_env
    patched += 1

    # --- seam 3: enforce the cap ------------------------------------------
    _orig_execute_action = solver_mod._HarnessGameSession._execute_action

    def p2_execute_action(self, action, *, batch_index, batch_size,
                          generated_tokens=None, flush_viewer_payload=True):
        st = _state(self)

        # ``_execute_auto_reset`` also lands here; it is not part of a batch.
        gated = bool(st.in_batch and st.batch_gated)

        if gated:
            try:
                reason = should_abort_remainder(
                    executed=st.batch_executed,
                    consecutive_noops=st.batch_noop_run,
                    consecutive_stale=st.batch_stale_run,
                )
            except Exception as exc:  # noqa: BLE001
                st.errors += 1
                log.debug("p2: predicate failed: %s", exc)
                reason = None
            if reason:
                saved = max(1, batch_size - batch_index + 1)
                st.saved += saved
                st.batch_saved = saved
                st.batch_reason = reason
                st.aborts += 1
                st.max_saved_one_batch = max(st.max_saved_one_batch, saved)
                _emit("batch_gate", st,
                      f"bi={batch_index}/{batch_size} saved={saved} "
                      f"reason={reason} gate={gate_reason(st.batch_fired)}")
                raise P2BatchAbort(
                    "P2: the rest of this batch was not executed. The board "
                    "history showed a stall when you issued it and the batch "
                    "then produced no board change, so every later action was "
                    "fired blind. Re-plan from the board you can now see."
                )

        payload = _orig_execute_action(
            self, action,
            batch_index=batch_index, batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload,
        )

        try:
            if st.in_batch:
                st.executed += 1
                st.batch_executed += 1
            state = self.game.current_state
            st.sync_level(int(state.levels_completed))
            fp = board_fingerprint(solver_mod._grid_from_state(state))
            if st.in_batch:
                st.observe(fp, bool(payload.get("board_changed")))
            else:
                # an auto-RESET is not part of a batch, but its state still
                # counts as visited for the level it lands on
                st.visited.add(fp)
        except Exception as exc:  # noqa: BLE001
            st.errors += 1
            log.debug("p2: post-action bookkeeping failed: %s", exc)
        return payload

    solver_mod._HarnessGameSession._execute_action = p2_execute_action
    patched += 1

    # --- requested-action accounting (denominator for the delivery rate) ---
    _orig_normalize = solver_mod._HarnessGameSession._normalize_actions

    def p2_normalize_actions(self, arguments):
        actions, error = _orig_normalize(self, arguments)
        try:
            st = _state(self)
            if error is None and actions is not None:
                st.requested += len(actions)
        except Exception:  # noqa: BLE001
            pass
        return actions, error

    solver_mod._HarnessGameSession._normalize_actions = p2_normalize_actions
    patched += 1

    _APPLIED = True
    return patched


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def apply(bm: Any = None) -> bool:
    """Install P2 v1. True on success (or already applied); False on flag-off /
    kill switch / any failure - in which case NOTHING is changed (vanilla duck).

    Arm flag:    P2_BATCHGATE=1
    Kill switch: P2_BATCHGATE_DISABLE=1
    Sub-flags:   P2_STALE_RUN    (the SHIPPED rule; value picked by
                                  p2_replay.py as the smallest k cutting ZERO
                                  level-completing actions on every run)
                 P2_NOOP_RUN     (default 0 = off; tighter special case)
                 P2_CAP          (default 0  - ABLATION ONLY, unsafe: the
                                  offline replay proves a position cap deletes
                                  level-completing actions)
                 P2_MIN_BATCH    (default 2)
                 P2_GATE_NET_ZERO / P2_GATE_STAGNATION / P2_GATE_REVISIT
                 P2_GATE_ALWAYS  (ablation, default off)
                 P2_MIN_ROUNDTRIP / P2_STAGNATION_MIN / P2_REVISIT_MIN /
                 P2_WINDOW / P2_REVISIT_WINDOW  (EFFNOTE v1 shipped values)
                 P2_MAX_EVENTS   (log-volume bound)
    """
    if _kill_switch():
        log.info("p2 %s: P2_BATCHGATE_DISABLE=1 -> no-op", VERSION)
        return False
    if not _flag_on():
        log.info("p2 %s: P2_BATCHGATE!=1 -> no-op (flag-gated arm)", VERSION)
        return False
    if _EN is None:
        log.error("[p2] effnote_patch (detector source) not importable "
                  "-> stock duck harness (vanilla)")
        return False
    try:
        patched = _apply_patches()
        if bm is not None and hasattr(bm, "label"):
            bm.label = f"{bm.label}-p2-{VERSION}"
        print(
            f"p2 {VERSION}: ACTIVE ({patched} seams patched) - VERIFIED-PLAN "
            f"BATCH GATING. This is a CONSTRAIN arm, not an INFORM arm: the "
            f"runner CAPS the batch, it does not advise the model. "
            f"Gate = the SHIPPED EFFNOTE stall detectors (imported, not "
            f"re-implemented) evaluated on the frame history at the moment the "
            f"batch is issued: net-zero>={CFG.min_roundtrip}, "
            f"stagnation>={CFG.stagnation_min}, revisit>={CFG.revisit_min}. "
            f"Rule = STALE-STATE RUN cap: on a GATED batch of "
            f">={CFG.min_batch} actions, abort the remainder once "
            f"{CFG.stale_run} consecutive action(s) have reached NO board "
            f"state new to this level (k-tolerant, so re-traversal prefixes "
            f"survive); companion no-op run cap P2_NOOP_RUN={CFG.noop_run} "
            f"(0 = OFF). "
            f"Position cap P2_CAP={CFG.cap} (0 = OFF; ablation handle only - "
            f"the offline replay proves a position cap deletes "
            f"level-completing actions). "
            f"A productive batch is NEVER truncated. Ungated batches pass "
            f"through untouched. No game id, no baseline, no metadata read. "
            f"Zero LLM calls, vanilla fallback.",
            flush=True,
        )
        log.info("p2 %s installed (%d seams)", VERSION, patched)
        return True
    except Exception:  # noqa: BLE001 - any failure -> vanilla fallback
        log.exception("[p2] apply failed -> stock duck harness (vanilla)")
        return False
