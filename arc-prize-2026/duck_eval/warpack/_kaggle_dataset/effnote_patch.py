"""EFFNOTE - quantified per-turn action-efficiency note (report-only).

Spec: ``learnings/war_room/harness_diff_2026-08-13.md`` sec4 item #1.
Prereg: ``learnings/war_room/effnote_prereg_2026-08-13.md`` (SEALED before this
file was pushed to an eval kernel).

THE DEFECT. Our whole efficiency treatment is ONE unquantified sentence in the
stock system prompt - ``prompts.py:17``::

    "- Optimize for as few in-game actions as possible while still being reliable."

The model is NEVER shown the scoring rule and NEVER sees its own action count.
Meanwhile the per-level score is ``min(115, (baseline/actions)^2 * 100)``,
further capped at the human baseline contribution - i.e. QUADRATIC in waste,
with no credit for beating the human count. The 08-12 diagnosis measured our
17 cleared levels at 2.11x the human action count; the SAME 17 levels re-scored
at the human baseline give 2.549 local ~ 1.48-1.58 LB. Every other lever chases
more levels; this one makes the levels we already clear worth more, i.e. it
moves the MEAN.

THE MECHANISM (this module). Append to the USER turn - only when there is
something to say - a bounded note carrying:

  (a) the scoring rule, stated verbatim AND quantitatively (baseline=100,
      2x=25, 3x=11, 5x=4);
  (b) the live action count on THIS level plus the clamped, game-agnostic
      ``heuristic_action_target()`` proxy;
  (c) the over-target ratio;
  (d) three PURE stall detectors over the frame history the agent already has -
      ``detect_net_zero_cycle`` (shortest >=6-action round-trip back to an exact
      prior same-level grid, with a divergence requirement so a static board is
      not flagged), ``detect_stagnation`` (>=8 consecutive same-level actions
      leaving the grid byte-identical), ``count_recent_revisits`` (>=4 exact
      recurrences of the current grid);
  (e) a commit-don't-scan reminder.

Zero LLM calls, zero GPU, zero new tools. The hot ``step_env`` /
``_execute_action`` path is NEVER touched - this module patches exactly two
seams and one of them only reads.

Reference: ``EfficiencyToolAgent`` in ``thtennant/taaf-kaggle-source-share-fork``
(``taaf_grafts/agent_ext.py``), shipped in ``caoyupeng/arc3-duck-v12-1d7d88``
(Tara Labs #37 @ 1.46) behind ``install(bm, flags={"efficiency": True, ...})``.
This is OUR implementation in the house patch pattern, not a port. Four
DELIBERATE DIVERGENCES from the reference, each one a constraint we learned the
expensive way:

  1. PROXY-ONLY BASELINES. The reference prefers a REAL per-level baseline
     (``game.base_actions_per_level``, else an rglob over the shipped
     ``metadata.json`` files) and only falls back to the heuristic. WE NEVER
     READ A BASELINE. Reasons: (i) a per-game baseline table is game-specific
     and, per the 08-12 P1 finding, factually wrong on a rerun (the latent-state
     game set is run-dependent); (ii) the real baselines exist offline and are
     stripped on the hidden set, so preferring them means MEASURING one
     mechanism and SHIPPING another - the exact class of error that killed the
     animation arm. The clamped game-agnostic proxy is the only target that
     ships, so the offline eval and the hidden set see the identical mechanism.
  2. COST BOUNDED IN CHARACTERS, NOT TOKEN FRACTION. The note is an INPUT-token
     cost and the rail reports GENERATED tokens only. That denominator mismatch
     fired K-A3 and killed the animation arm, and forced P1 prereg addendum A1.
     ``EFFNOTE_MAX_CHARS`` (default 700) is a hard static clamp, enforced by
     dropping whole trailing lines - never a mid-sentence cut.
  3. MONKEYPATCH, NOT A ToolAgent SUBCLASS. The reference installs an
     ``analyzer_factory`` building an ``EfficiencyToolAgent``. Our house pattern
     patches ``ToolAgent._build_user_prompt`` directly (as P1 mechanism C does),
     so the arm composes with the (f) continuation default and cannot fight
     another graft over the factory slot.
  4. NO REPLAY / DUPLICATE-GAME GATE. caoyupeng's ``external_game_id=f"{env}-dup"``
     scouting gate is NOT ported in any form.

HOUSE PATTERN (mirrors p1_suppressor_patch / animation_patch / compaction_patch):
VERSION marker, arm flag + kill switch, blanket-guarded ``apply()``, runtime
banner, ``bm.label`` stamp, greppable events, canary counters, per-session state
(no globals, no locks, no threads), **vanilla fallback on ANY failure**.

Seams (2; zero LLM calls, zero prompt-FILE edits, hot action path untouched):
  1. ``solver._HarnessGameSession.play``          -> bind state to the analyzer
  2. ``tool_agent.ToolAgent._build_user_prompt``  -> append the note

LEGALITY: game-agnostic by construction. No game id is ever read, no per-game
table exists, the detectors read only the agent's own frame history.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Sequence

log = logging.getLogger("effnote")

VERSION = "v1"
_EVENT_V = "1"

_APPLIED = False


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
    return os.environ.get("EFFNOTE", "").strip() == "1"


def _kill_switch() -> bool:
    return os.environ.get("EFFNOTE_DISABLE", "").strip() == "1"


class Config:
    """Read live on every use so every knob is a runtime kill switch."""

    # -- cost bound (CHARACTERS - never a token fraction; see divergence 2) --
    @property
    def max_chars(self) -> int:
        return max(120, _env_int("EFFNOTE_MAX_CHARS", 700))

    # -- detectors ---------------------------------------------------------
    @property
    def min_roundtrip(self) -> int:
        # A same-state match below this span is a trivial no-op probe, not a
        # waste burst. Floor of 2 keeps the detector meaningful.
        return max(2, _env_int("EFFNOTE_MIN_ROUNDTRIP", 6))

    @property
    def stagnation_min(self) -> int:
        return max(2, _env_int("EFFNOTE_STAGNATION_MIN", 8))

    @property
    def revisit_min(self) -> int:
        return max(2, _env_int("EFFNOTE_REVISIT_MIN", 4))

    @property
    def window(self) -> int:
        # Bounds the backward scan so long games stay O(window) per turn.
        return max(8, _env_int("EFFNOTE_WINDOW", 240))

    @property
    def revisit_window(self) -> int:
        return max(8, _env_int("EFFNOTE_REVISIT_WINDOW", 120))

    # -- the clamped, GAME-AGNOSTIC target proxy ---------------------------
    @property
    def target_base(self) -> int:
        return _env_int("EFFNOTE_TARGET_BASE", 50)

    @property
    def target_per_action(self) -> int:
        return _env_int("EFFNOTE_TARGET_PER_ACTION", 5)

    @property
    def target_board_cap(self) -> int:
        return _env_int("EFFNOTE_TARGET_BOARD_CAP", 30)

    @property
    def target_min(self) -> int:
        return _env_int("EFFNOTE_TARGET_MIN", 40)

    @property
    def target_max(self) -> int:
        return _env_int("EFFNOTE_TARGET_MAX", 100)

    # -- switches ----------------------------------------------------------
    @property
    def detectors(self) -> bool:
        return _env_bool("EFFNOTE_DETECTORS", True)

    @property
    def max_events(self) -> int:
        # Bound the greppable per-game event stream so a 5k-action game cannot
        # flood the kernel log. The game_end summary always prints.
        return max(0, _env_int("EFFNOTE_MAX_EVENTS", 400))


CFG = Config()


# --------------------------------------------------------------------------- #
# pure detection (no I/O, no LLM, no game id - unit-testable standalone)
# --------------------------------------------------------------------------- #
def _grid_of(frame: Any) -> Any:
    return getattr(frame, "grid", None)


def _level_of(frame: Any) -> Any:
    return getattr(frame, "level", None)


def detect_net_zero_cycle(
    current_frame: Any,
    history_frames: Sequence[Any],
    *,
    min_roundtrip_actions: int | None = None,
    window: int | None = None,
) -> int | None:
    """Length (in actions) of the shortest recent net-zero round-trip that
    returned the board to ``current_frame``'s EXACT grid, else ``None``.

    Reported only when (a) the current grid re-appears at a prior same-level
    frame, (b) at least one intervening frame DIVERGED from it - so a genuinely
    static / no-op board is left to :func:`detect_stagnation` instead of being
    double-reported as a cycle - and (c) the span is >= ``min_roundtrip_actions``.

    ``history_frames`` is the per-game frame history, oldest first; its last
    element is the current post-action frame. Walks backwards, stops at the
    first frame from a different level, looks at most ``window`` frames back.
    Pure.
    """
    if current_frame is None or not history_frames:
        return None
    min_rt = CFG.min_roundtrip if min_roundtrip_actions is None else min_roundtrip_actions
    win = CFG.window if window is None else window
    cur_grid = _grid_of(current_frame)
    cur_level = _level_of(current_frame)
    recent = list(history_frames[-max(1, win):])
    n = len(recent)
    saw_divergence = False
    for k in range(1, n + 1):
        frame = recent[n - k]
        if frame is None or _level_of(frame) != cur_level:
            break
        if _grid_of(frame) != cur_grid:
            saw_divergence = True
            continue
        cycle_actions = k - 1
        if saw_divergence and cycle_actions >= min_rt:
            return cycle_actions
    return None


def detect_stagnation(
    current_frame: Any,
    history_frames: Sequence[Any],
    *,
    min_run: int | None = None,
    window: int | None = None,
) -> int | None:
    """How many consecutive most-recent same-level actions left the grid EXACTLY
    equal to ``current_frame``'s grid, if that run is >= ``min_run``, else
    ``None``.

    Baseline-free stall signal: a long no-change run means the agent is issuing
    actions with zero board effect (stuck on a screen / hammering an inert
    control). Pure.
    """
    if current_frame is None or not history_frames:
        return None
    floor = CFG.stagnation_min if min_run is None else min_run
    win = CFG.window if window is None else window
    cur_grid = _grid_of(current_frame)
    cur_level = _level_of(current_frame)
    recent = list(history_frames[-max(1, win):])
    n = len(recent)
    run = 0
    # k=1 is the current frame itself; start at its predecessor.
    for k in range(2, n + 1):
        frame = recent[n - k]
        if frame is None or _level_of(frame) != cur_level or _grid_of(frame) != cur_grid:
            break
        run += 1
    return run if run >= floor else None


def count_recent_revisits(
    current_frame: Any,
    history_frames: Sequence[Any],
    *,
    window: int | None = None,
) -> int:
    """How many recent same-level frames (excluding the current one) re-present
    ``current_frame``'s grid EXACTLY.

    EXACT match only, deliberately. A near-match tolerance treats incremental
    single-object motion (clear one cell + set one cell = 2 changed cells) as a
    repeat, so it would flag a legitimate avatar marching across the board as
    "cycling" on every turn of genuine linear progress - contradictory, harmful
    advice on a core ARC-AGI-3 mechanic. Only a true return to a previously
    occupied exact state is a revisit. Pure.
    """
    if current_frame is None or not history_frames:
        return 0
    win = CFG.revisit_window if window is None else window
    cur_grid = _grid_of(current_frame)
    cur_level = _level_of(current_frame)
    recent = list(history_frames[-max(1, win):])
    n = len(recent)
    count = 0
    for k in range(2, n + 1):
        frame = recent[n - k]
        if frame is None or _level_of(frame) != cur_level:
            break
        if _grid_of(frame) == cur_grid:
            count += 1
    return count


def heuristic_action_target(
    valid_action_count: int | None,
    board_cells: int | None,
) -> int:
    """The clamped, GAME-AGNOSTIC per-level soft action target - a baseline
    PROXY. This is the ONLY target this module ever uses (see divergence 1).

    Defensible with no baseline of any kind: a base allowance, plus a few probe
    actions per available valid action, plus a bounded board-size allowance,
    clamped to ``[EFFNOTE_TARGET_MIN, EFFNOTE_TARGET_MAX]`` = ``[40, 100]``.
    Being generous keeps the escalation reminder from nagging a genuinely short
    level while still firing once a level drags well past a normal solve length.

    NOTE for readers of the prereg: on this rail the board is 64x64 and >=4
    valid actions are exposed, so the clamp binds and the proxy is 100 in
    practice. It is written as a function of observables anyway so that it
    degrades correctly on smaller boards / restricted action sets, and so that
    nothing here is conditioned on which game is being played. Pure.
    """
    target = CFG.target_base
    if valid_action_count and int(valid_action_count) > 0:
        target += CFG.target_per_action * int(valid_action_count)
    if board_cells and int(board_cells) > 0:
        target += min(CFG.target_board_cap, int(int(board_cells) ** 0.5))
    return max(CFG.target_min, min(CFG.target_max, int(target)))


# --------------------------------------------------------------------------- #
# the note (pure; bounded in CHARACTERS)
# --------------------------------------------------------------------------- #
# (a) The scoring rule, verbatim AND quantitative. Stated as the scorer really
#     behaves: the level term is capped, so MATCHING the human count is full
#     credit and beating it is worth nothing extra (efficiency_diagnosis sec2.3).
_HEADER = (
    "EFFICIENCY BUDGET - this level scores "
    "(human_baseline_actions/your_actions)^2 x 100, capped at 100: "
    "baseline=100, 2x over=25, 3x=11, 5x=4. Waste costs you quadratically."
)

# (e) The commit-don't-scan reminder.
_REMINDER = (
    "Commit to your best hypothesis and the shortest sequence that tests it; "
    "do not scan rows/columns or enumerate options. One idea, read the "
    "result, decide."
)


def _clamp(lines: list[tuple[int, str]], max_chars: int) -> str:
    """Render ``lines`` (``(keep_priority, text)``, emission order preserved),
    dropping WHOLE lines - lowest keep-priority first - until it fits.

    Never cuts mid-sentence. Keep-priority: 0 = the scoring rule (a), 1 = the
    budget/over-target line (b)+(c), 2 = the commit-don't-scan reminder (e),
    3.. = the stall detector lines (d), which are the most numerous and the
    most redundant with each other.
    """
    if not lines:
        return ""
    keep = list(lines)
    while len(keep) > 1 and len("\n".join(t for _, t in keep)) > max_chars:
        worst = max(range(len(keep)), key=lambda i: (keep[i][0], i))
        keep.pop(worst)
    text = "\n".join(t for _, t in keep)
    if len(text) > max_chars:  # single oversized line: cannot happen with the
        text = text[:max_chars]  # shipped strings, but the bound is a HARD bound
    return text


def build_efficiency_note(
    *,
    level_number: int | None,
    actions_this_level: int | None,
    target: int | None,
    net_zero_actions: int | None = None,
    stagnation_actions: int | None = None,
    revisit_count: int | None = None,
    max_chars: int | None = None,
) -> str:
    """Assemble the per-turn note (may be empty). Pure, bounded, no I/O.

    Quiet when there is nothing to say: no actions spent on this level yet AND
    no stall firing. Otherwise always carries (a) the scoring rule and (b) the
    live count vs the proxy target, and adds (c)/(d)/(e) as they apply.
    """
    cap = CFG.max_chars if max_chars is None else max_chars
    used = int(actions_this_level or 0)

    net_zero = int(net_zero_actions) if net_zero_actions else None
    stagnating = int(stagnation_actions) if stagnation_actions else None
    oscillating = (
        int(revisit_count)
        if revisit_count is not None and int(revisit_count) >= CFG.revisit_min
        else None
    )
    any_stall = (net_zero is not None) or (stagnating is not None) or (oscillating is not None)

    if used <= 0 and not any_stall:
        return ""

    tgt = int(target) if target and int(target) > 0 else None
    over = tgt is not None and used > tgt

    lines: list[tuple[int, str]] = [(0, _HEADER)]

    lvl = f"Level {level_number}" if level_number is not None else "This level"
    if tgt is not None and used > 0:
        if over:
            # (c) the over-target ratio
            lines.append((1,
                f"{lvl}: {used} actions used; a strong score needs about {tgt} "
                f"or fewer - you are {used / tgt:.1f}x over the typical target."))
        else:
            lines.append((1, f"{lvl}: {used} of about {tgt} typical actions used."))
    elif used > 0:
        lines.append((1, f"{lvl}: {used} actions used on this level so far."))

    # (d) the three pure stall detectors
    if stagnating is not None:
        lines.append((3,
            f"STALL: the board has not changed for your last {stagnating} "
            f"actions here - they did nothing."))
    if net_zero is not None:
        lines.append((4,
            f"NET-ZERO: your last {net_zero} actions round-tripped back to a "
            f"state you had already seen - no progress."))
    if oscillating is not None:
        lines.append((5,
            f"REVISIT: this exact board state has recurred {oscillating}x "
            f"recently - you are cycling."))

    # (e) the commit-don't-scan reminder, gated so it is not boilerplate
    if over or any_stall:
        lines.append((2, _REMINDER))

    if len(lines) == 1:
        return ""
    return _clamp(lines, cap)


# --------------------------------------------------------------------------- #
# per-session state + canary
# --------------------------------------------------------------------------- #
class EffNoteState:
    """Per-game counters. One instance per ``_HarnessGameSession``; no globals,
    no locks, no threads (each game runs in its own thread)."""

    __slots__ = (
        "game", "turns", "noted", "chars_sum", "chars_max", "over_target",
        "fire_net_zero", "fire_stagnation", "fire_revisit", "fire_any",
        "errors", "events", "last_level", "max_actions_seen",
    )

    def __init__(self, game: str) -> None:
        self.game = game
        self.turns = 0
        self.noted = 0
        self.chars_sum = 0
        self.chars_max = 0
        self.over_target = 0
        self.fire_net_zero = 0
        self.fire_stagnation = 0
        self.fire_revisit = 0
        self.fire_any = 0
        self.errors = 0
        self.events = 0
        self.last_level = None
        self.max_actions_seen = 0


def _emit(kind: str, st: EffNoteState, detail: str) -> None:
    if st.events >= CFG.max_events and kind != "game_end":
        return
    st.events += 1
    print(
        f"EFFNOTE v={_EVENT_V} kind={kind} game={st.game} "
        f"turns={st.turns} noted={st.noted} over={st.over_target} "
        f"nz={st.fire_net_zero} stag={st.fire_stagnation} rev={st.fire_revisit} "
        f"chars_max={st.chars_max} errors={st.errors} {detail}",
        flush=True,
    )


CANARY: dict[str, dict[str, Any]] = {}


def canary_report() -> dict[str, Any]:
    """End-of-run canary. Prints one greppable line and returns the numbers.

    K-E0 delivery  : ``noted``/``turns`` and the char bound (``chars_max``).
    K-E1 detectors : per-detector fire counts and the number of distinct games
                     each fired on.
    K-E3 cost      : ``chars_max`` vs ``EFFNOTE_MAX_CHARS`` - a CHARACTER bound,
                     never a token fraction (see the module docstring).
    """
    games = sorted(CANARY)
    keys = ("turns", "noted", "over_target", "fire_net_zero", "fire_stagnation",
            "fire_revisit", "fire_any", "chars_sum", "errors")
    tot = {k: sum(int(CANARY[g].get(k, 0)) for g in games) for k in keys}
    chars_max = max([int(CANARY[g].get("chars_max", 0)) for g in games], default=0)
    ngames = {
        k: sum(1 for g in games if int(CANARY[g].get(k, 0)) > 0)
        for k in ("fire_net_zero", "fire_stagnation", "fire_revisit")
    }
    turns = tot["turns"] or 1
    report = {
        "version": VERSION,
        "games": len(games),
        "chars_max": chars_max,
        "chars_mean": (tot["chars_sum"] / tot["noted"]) if tot["noted"] else 0.0,
        "note_rate": tot["noted"] / turns,
        "over_target_rate": tot["over_target"] / turns,
        "stall_rate": tot["fire_any"] / turns,
        "games_net_zero": ngames["fire_net_zero"],
        "games_stagnation": ngames["fire_stagnation"],
        "games_revisit": ngames["fire_revisit"],
        "max_chars_bound": CFG.max_chars,
        **tot,
    }
    print(
        f"EFFNOTE CANARY v={_EVENT_V} version={VERSION} games={len(games)} "
        f"turns={tot['turns']} noted={tot['noted']} "
        f"note_rate={report['note_rate']:.4f} "
        f"chars_mean={report['chars_mean']:.1f} chars_max={chars_max} "
        f"bound={CFG.max_chars} over_target={tot['over_target']} "
        f"over_rate={report['over_target_rate']:.4f} "
        f"stall_turns={tot['fire_any']} stall_rate={report['stall_rate']:.4f} "
        f"nz={tot['fire_net_zero']}/{ngames['fire_net_zero']}g "
        f"stag={tot['fire_stagnation']}/{ngames['fire_stagnation']}g "
        f"rev={tot['fire_revisit']}/{ngames['fire_revisit']}g "
        f"errors={tot['errors']} target=proxy-only",
        flush=True,
    )
    return report


# --------------------------------------------------------------------------- #
# note assembly from live harness objects
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


def level_and_actions(
    game: Any,
    action_num: int,
    current_frame: Any,
    history_entries: Any,
) -> tuple[int | None, int | None]:
    """``(level_number_1based, actions_this_level)``.

    Primary source is the live ``game.game_run`` per-level counter (the same
    array the scorer reads). Falls back to counting current-level frames in the
    history the model itself sees, then to ``action_num``.
    """
    run = getattr(game, "game_run", None)
    apl = getattr(run, "actions_per_level", None)
    lc = getattr(run, "levels_completed", None)
    if isinstance(apl, (list, tuple)) and apl and isinstance(lc, int):
        idx = min(max(0, lc), len(apl) - 1)
        try:
            return idx + 1, int(apl[idx])
        except (TypeError, ValueError):
            pass
    level = getattr(current_frame, "level", None) if current_frame is not None else None
    if level is None:
        level = 1
    if history_entries:
        count = 0
        for entry in history_entries:
            frame = getattr(entry, "frame", None)
            if frame is not None and getattr(frame, "level", None) == level:
                count += 1
        # the seed entry is a frame with no action behind it
        return level, max(0, count - 1)
    return level, max(0, int(action_num))


def note_for_turn(
    st: EffNoteState,
    game: Any,
    action_num: int,
    current_frame: Any,
    history_entries: Any,
    valid_actions: Any,
) -> str:
    """Compute this turn's note and update ``st``. Never raises to the caller's
    caller: the seam wraps this, but keep it defensive anyway."""
    st.turns += 1
    level_number, actions_this_level = level_and_actions(
        game, action_num, current_frame, history_entries)
    if actions_this_level:
        st.max_actions_seen = max(st.max_actions_seen, int(actions_this_level))

    frames = [getattr(e, "frame", None) for e in (history_entries or [])]
    if CFG.detectors:
        net_zero = detect_net_zero_cycle(current_frame, frames)
        stagnation = detect_stagnation(current_frame, frames)
        revisits = count_recent_revisits(current_frame, frames)
    else:
        net_zero = stagnation = None
        revisits = 0

    board_cells = 0
    if current_frame is not None:
        try:
            rows, cols = current_frame.shape
            board_cells = int(rows) * int(cols)
        except Exception:  # noqa: BLE001
            board_cells = 0
    target = heuristic_action_target(
        len(valid_actions) if valid_actions else 0, board_cells)

    note = build_efficiency_note(
        level_number=level_number,
        actions_this_level=actions_this_level,
        target=target,
        net_zero_actions=net_zero,
        stagnation_actions=stagnation,
        revisit_count=revisits,
    )

    fired = []
    if net_zero:
        st.fire_net_zero += 1
        fired.append(f"nz={net_zero}")
    if stagnation:
        st.fire_stagnation += 1
        fired.append(f"stag={stagnation}")
    if revisits >= CFG.revisit_min:
        st.fire_revisit += 1
        fired.append(f"rev={revisits}")
    if fired:
        st.fire_any += 1
    over = bool(actions_this_level and actions_this_level > target)
    if over:
        st.over_target += 1

    if note:
        st.noted += 1
        st.chars_sum += len(note)
        st.chars_max = max(st.chars_max, len(note))

    if fired or over:
        _emit(
            "note", st,
            f"anum={action_num} level={level_number} used={actions_this_level} "
            f"target={target} over={int(over)} chars={len(note)} "
            f"fired={','.join(fired) or 'none'}",
        )
    if level_number != st.last_level:
        st.last_level = level_number
    return note


# --------------------------------------------------------------------------- #
# patches
# --------------------------------------------------------------------------- #
def _apply_patches() -> int:
    global _APPLIED
    if _APPLIED:
        return 0

    import inference.framework.solver as solver_mod
    from inference.agent.tool_agent import ToolAgent

    patched = 0

    # --- seam 1: bind per-game state (and the game) to the analyzer --------
    _orig_play = solver_mod._HarnessGameSession.play

    def effnote_play(self):
        st = getattr(self, "_effnote_state", None)
        if st is None:
            st = EffNoteState(_game_label(self))
            self._effnote_state = st
        try:
            self.analyzer._effnote_state = st
            self.analyzer._effnote_game = self.game
        except Exception as exc:  # noqa: BLE001
            st.errors += 1
            log.debug("effnote: analyzer bind failed: %s", exc)
        try:
            return _orig_play(self)
        finally:
            try:
                CANARY[st.game] = {
                    "turns": st.turns, "noted": st.noted,
                    "chars_sum": st.chars_sum, "chars_max": st.chars_max,
                    "over_target": st.over_target,
                    "fire_net_zero": st.fire_net_zero,
                    "fire_stagnation": st.fire_stagnation,
                    "fire_revisit": st.fire_revisit,
                    "fire_any": st.fire_any,
                    "errors": st.errors,
                    "max_actions_seen": st.max_actions_seen,
                }
                _emit("game_end", st, f"max_actions_seen={st.max_actions_seen}")
            except Exception:  # noqa: BLE001
                pass

    solver_mod._HarnessGameSession.play = effnote_play
    patched += 1

    # --- seam 2: append the note to the USER turn --------------------------
    _orig_build_user_prompt = ToolAgent._build_user_prompt

    def effnote_build_user_prompt(self, action_num, *, valid_actions=None, **kwargs):
        text = _orig_build_user_prompt(
            self, action_num, valid_actions=valid_actions, **kwargs)
        st = getattr(self, "_effnote_state", None)
        if not isinstance(st, EffNoteState):
            return text
        try:
            note = note_for_turn(
                st,
                getattr(self, "_effnote_game", None),
                action_num,
                kwargs.get("current_frame"),
                kwargs.get("history_entries"),
                valid_actions,
            )
        except Exception as exc:  # noqa: BLE001 - the prompt may NEVER break
            st.errors += 1
            log.debug("effnote: note build failed: %s", exc)
            return text
        return f"{text}\n{note}" if note else text

    ToolAgent._build_user_prompt = effnote_build_user_prompt
    patched += 1

    _APPLIED = True
    return patched


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def apply(bm: Any = None) -> bool:
    """Install EFFNOTE v1. True on success (or already applied); False on
    flag-off / kill switch / any failure - in which case NOTHING is changed
    (vanilla duck).

    Arm flag:    EFFNOTE=1
    Kill switch: EFFNOTE_DISABLE=1
    Sub-flags:   EFFNOTE_MAX_CHARS (default 700, a CHARACTER bound)
                 EFFNOTE_DETECTORS (default on)
                 EFFNOTE_MIN_ROUNDTRIP / EFFNOTE_STAGNATION_MIN /
                 EFFNOTE_REVISIT_MIN / EFFNOTE_WINDOW / EFFNOTE_REVISIT_WINDOW
                 EFFNOTE_TARGET_* (the clamped game-agnostic proxy)
                 EFFNOTE_MAX_EVENTS (log-volume bound)
    """
    if _kill_switch():
        log.info("effnote %s: EFFNOTE_DISABLE=1 -> no-op", VERSION)
        return False
    if not _flag_on():
        log.info("effnote %s: EFFNOTE!=1 -> no-op (flag-gated arm)", VERSION)
        return False
    try:
        patched = _apply_patches()
        if bm is not None and hasattr(bm, "label"):
            bm.label = f"{bm.label}-effnote-{VERSION}"
        probe = heuristic_action_target(5, 64 * 64)
        print(
            f"effnote {VERSION}: ACTIVE ({patched} seams patched) - quantified "
            f"per-turn efficiency note, REPORT-ONLY (no action is ever blocked, "
            f"declined or injected). "
            f"Note = scoring rule (verbatim, quantitative) + live action count "
            f"vs a CLAMPED GAME-AGNOSTIC target proxy + over-target ratio + 3 "
            f"pure stall detectors (net-zero>={CFG.min_roundtrip}, "
            f"stagnation>={CFG.stagnation_min}, revisit>={CFG.revisit_min}) + "
            f"commit-don't-scan reminder. "
            f"target=proxy-only (NO baseline table, NO metadata read, NO game "
            f"id - proxy on this rail = {probe}). "
            f"cost bound = {CFG.max_chars} CHARACTERS (never a token fraction). "
            f"Zero LLM calls, hot action path untouched, vanilla fallback.",
            flush=True,
        )
        log.info("effnote %s installed (%d seams)", VERSION, patched)
        return True
    except Exception:  # noqa: BLE001 - any failure -> vanilla fallback
        log.exception("[effnote] apply failed -> stock duck harness (vanilla)")
        return False
