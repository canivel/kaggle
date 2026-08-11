"""Animation-awareness  -  sweep 2026-08-11 ADOPT #1, prereg
``learnings/war_room/animation_prereg_2026-08-11.md`` (SEALED before this file
was written).

THE DEFECT (verified in our own artifact, LM-free, offline):
``taaf/game.py:170`` defines ``GameState.frame -> Frame(data=self.raw.frame[-1])``
while ``arcengine`` renders a frame after every internal ``step()``, so one
action can come back as a short animation. ``GameState.animation_frames``
(``raw.frame[:-1]``) and ``GameState.all_frames`` (``raw.frame``) are defined
right below it and have **zero consumers** anywhere under
``src/ARC3-Inference/``. Our agent has never seen an intermediate frame.

WHAT THAT COSTS (``runs/animation/frame_audit.md``, 11,104 audited actions over
all 25 official games): 17/25 games return multi-frame responses (up to 118
frames for ONE action on sb26), and **401 actions  -  3.6% of all actions, 19.0%
of every action that looked like a no-op  -  had a settled board byte-identical to
the board before the action while an intermediate frame differed**. On ``ft09``
that is 281/352 actions and **99.3% of everything that looked like a no-op**.
The agent is shown "nothing happened" and must conclude the action is inert.
That is the state-aliasing / false-no-op class named in project memory.

THE FIX (this module): compute a small, fixed-schema, deterministic summary of
the discarded frames and hand it to the agent. **No raw frames are ever
emitted**  -  one 64x64 ASCII grid is ~1,400-2,000 tokens and sb26 can return 118
of them for a single action. The summary is ~45 tokens, and is emitted only
when there was actually an animation, so ordinary actions cost nothing.

SEAMS (3, all harness-side, all deterministic, zero LLM calls):
  1. ``solver._HarnessGameSession._execute_action`` -> attach
     ``payload["animation"]`` (absent when single-frame / all frames identical).
  2. ``solver._HarnessGameSession.step_env``        -> merge a batch's per-action
     summaries into the batch payload (vanilla keeps only the LAST action's
     payload, so a batch would otherwise lose every animation but the last).
  3. ``tool_agent.ToolAgent._compact_action_result`` -> carry the field through
     to the model (the vanilla compactor drops unknown keys), plus
     ``_summarize_step_sequence`` / ``_describe_last_outcome`` so the "did not
     show a confirmed board change; treat this as weak evidence" sentence is
     replaced by the truth on exactly the actions where it is a lie.

EXPLICITLY NOT HERE (prereg sec2.2, so it cannot be smuggled in later):
the hard no-op guard (sweep ADOPT #2  -  strictly downstream, separately gated,
and *harmful* on type-1 games without this arm), the ``animation()`` diff-timeline
retrieval tool, and the proactive turns-without-progress hint. One mechanism,
one flag.

House pattern (mirrors continuation_patch / compaction_patch): VERSION marker,
env flag gate + kill switch, blanket-guarded ``apply()``, runtime banner,
``bm.label`` stamp, greppable events + per-game jsonl sidecars, canary counters,
**vanilla fallback on ANY failure** (worst case: the stock duck harness). No
threading, no locks, no game-id logic, no per-game special casing.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger("animation")

VERSION = "v1"

# Event/sidecar schema stamp.
_EVENT_V = "1"

_APPLIED = False

# The four games our own audit (runs/animation/frame_audit.md) proves are
# type-1, i.e. where INVISIBLE actions exist. Used ONLY by the end-of-run canary
# report (prereg K-A2) to say whether the mechanism engaged where it must.
# The patch itself never reads a game id  -  behaviour is identical on all games.
_AUDIT_TYPE1_GAMES = ("ft09", "cd82", "sc25", "ls20")

# Token-cost accounting: the summary is a fixed-schema scalar dict, so its cost
# is bounded. This is the per-emission estimate used for the K-A3 canary; it is
# deliberately generous (a measured json.dumps of the dict is ~110 chars).
_TOKENS_PER_SUMMARY = 45


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def _flag_on() -> bool:
    return os.environ.get("ANIMATION_AWARE", "").strip() == "1"


def _kill_switch() -> bool:
    return os.environ.get("ANIMATION_DISABLE", "").strip() == "1"


def _only_invisible() -> bool:
    """ANIMATION_ONLY_INVISIBLE=1 -> emit a summary ONLY for the aliased case
    (board unchanged but something was rendered). Default 0: also emit the
    cheap motion summary, which is what tells the agent an animation is even a
    thing in this game."""
    return os.environ.get("ANIMATION_ONLY_INVISIBLE", "0").strip() == "1"


def _outcome_text() -> bool:
    """ANIMATION_OUTCOME_TEXT=0 -> leave ``_describe_last_outcome`` vanilla
    (ablation handle for seam 3b)."""
    return os.environ.get("ANIMATION_OUTCOME_TEXT", "1").strip() != "0"


# --------------------------------------------------------------------------- #
# canary counters (module-level; one process = one run)
# --------------------------------------------------------------------------- #
class _Counters:
    def __init__(self) -> None:
        self.actions = 0
        self.multi = 0
        self.invisible = 0
        self.summaries = 0
        self.errors = 0
        self.by_game: dict[str, dict[str, int]] = {}

    def game(self, name: str) -> dict[str, int]:
        slot = self.by_game.get(name)
        if slot is None:
            slot = {"actions": 0, "multi": 0, "invisible": 0}
            self.by_game[name] = slot
        return slot


COUNTERS = _Counters()


# --------------------------------------------------------------------------- #
# frame maths (pure, deterministic, no engine calls)
# --------------------------------------------------------------------------- #
def _norm(frame: Any) -> tuple[tuple[int, ...], ...]:
    """Normalise one raw arcengine frame to a hashable tuple-of-tuples.
    Mirrors ``solver._grid_from_state`` semantics (numpy or nested list)."""
    rows = frame.tolist() if hasattr(frame, "tolist") else frame
    return tuple(tuple(int(cell) for cell in row) for row in rows)


def _diff_cells(a: tuple, b: tuple) -> list[tuple[int, int]]:
    """Coordinates differing between two normalised frames. Shape mismatch ->
    empty (a resize is a board change the agent already sees)."""
    if len(a) != len(b):
        return []
    out: list[tuple[int, int]] = []
    for r, (ra, rb) in enumerate(zip(a, b)):
        if ra == rb or len(ra) != len(rb):
            continue
        for c, (ca, cb) in enumerate(zip(ra, rb)):
            if ca != cb:
                out.append((r, c))
    return out


def summarize_animation(
    raw_frames: Any,
    previous_grid: tuple | None,
    *,
    max_cells: int = 4096,
) -> dict[str, Any] | None:
    """The whole mechanism, in one pure function.

    ``raw_frames`` is ``state.raw.frame`` (the FULL list arcengine returned).
    ``previous_grid`` is the settled board BEFORE this action.

    Returns None when there is nothing to say (single frame, or every frame
    identical) so ordinary actions cost exactly zero tokens. Never raises for
    ill-shaped input  -  the caller treats None as "no animation".

    Fixed schema, scalars only, NO raw frames:
      frames          how many frames the engine rendered for this one action
      unique_frames   how many of them were distinct
      board_unchanged settled board identical to the pre-action board
      transient_cells largest number of cells any intermediate frame differed
                      from the settled board by (capped at ``max_cells``)
      transient_bbox  [row0, col0, row1, col1] bounding box of the union of
                      those transient cells (inclusive), or None
      signature       'reject_or_consumed' when the board is unchanged but the
                      engine rendered something (the previously INVISIBLE case)
                      'motion' otherwise
    """
    try:
        return _summarize_animation_inner(raw_frames, previous_grid, max_cells)
    except Exception as exc:  # noqa: BLE001 - a perception summary may NEVER raise
        log.debug("animation: summarize failed on ill-shaped frames: %s", exc)
        return None


def _summarize_animation_inner(
    raw_frames: Any,
    previous_grid: tuple | None,
    max_cells: int,
) -> dict[str, Any] | None:
    frames = list(raw_frames or [])
    if len(frames) < 2:
        return None
    norm = [_norm(f) for f in frames]
    settled = norm[-1]
    intermediates = norm[:-1]
    if all(f == settled for f in intermediates):
        return None

    board_unchanged = previous_grid is not None and settled == previous_grid

    best = 0
    r0 = c0 = 1 << 30
    r1 = c1 = -1
    for f in intermediates:
        cells = _diff_cells(f, settled)
        if not cells:
            continue
        if len(cells) > best:
            best = len(cells)
        for r, c in cells[:max_cells]:
            if r < r0:
                r0 = r
            if c < c0:
                c0 = c
            if r > r1:
                r1 = r
            if c > c1:
                c1 = c
    bbox = [r0, c0, r1, c1] if r1 >= 0 else None

    return {
        "frames": len(norm),
        "unique_frames": len(set(norm)),
        "board_unchanged": bool(board_unchanged),
        "transient_cells": min(best, max_cells),
        "transient_bbox": bbox,
        "signature": "reject_or_consumed" if board_unchanged else "motion",
    }


def merge_animations(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Collapse a batch's per-action summaries into one batch-level summary.
    Keeps the aliased case visible: if ANY action in the batch was
    reject_or_consumed, the batch signature is reject_or_consumed."""
    real = [s for s in summaries if isinstance(s, dict)]
    if not real:
        return None
    if len(real) == 1:
        return dict(real[0])
    invisible = [s for s in real if s.get("signature") == "reject_or_consumed"]
    lead = (invisible or real)[-1]
    out = dict(lead)
    out["animated_actions"] = len(real)
    out["invisible_actions"] = len(invisible)
    return out


# --------------------------------------------------------------------------- #
# model-facing text
# --------------------------------------------------------------------------- #
def animation_note(summary: dict[str, Any] | None) -> str:
    """One short sentence for the agent. Empty when there is nothing to say."""
    if not isinstance(summary, dict):
        return ""
    frames = summary.get("frames")
    cells = summary.get("transient_cells")
    if summary.get("signature") == "reject_or_consumed":
        return (
            f"The engine rendered {frames} frames for this action and {cells} cell(s) "
            f"changed mid-animation before returning to the SAME board you saw before "
            f"the action. The action was NOT ignored: something happened and was "
            f"undone or consumed (e.g. a rejected click, a spent attempt, a bounce). "
            f"Do not record this as 'no effect'."
        )
    return (
        f"The engine rendered {frames} frames for this action ({cells} transient "
        f"cell(s)); the board also changed, so the animation is motion, not a "
        f"hidden effect."
    )


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
def _game_label(session: Any) -> str:
    for getter in (
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


def _emit_event(session: Any, game: str, summary: dict[str, Any],
                action_display: str) -> None:
    """Greppable stdout line + best-effort per-game jsonl sidecar."""
    print(
        f"ANIMATION v={_EVENT_V} kind={summary.get('signature')} game={game} "
        f"action={action_display} frames={summary.get('frames')} "
        f"unique={summary.get('unique_frames')} "
        f"board_unchanged={1 if summary.get('board_unchanged') else 0} "
        f"transient_cells={summary.get('transient_cells')} "
        f"bbox={summary.get('transient_bbox')} "
        f"run_actions={COUNTERS.actions} run_multi={COUNTERS.multi} "
        f"run_invisible={COUNTERS.invisible}",
        flush=True,
    )
    state_path = getattr(session, "_animation_state_path", None)
    if state_path is None:
        return
    try:
        rec = {
            "kind": summary.get("signature"),
            "v": _EVENT_V,
            "game": game,
            "action": action_display,
            **{k: summary.get(k) for k in (
                "frames", "unique_frames", "board_unchanged",
                "transient_cells", "transient_bbox")},
        }
        path = Path(state_path).parent / f"{game}_animation_events.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
    except Exception as exc:  # noqa: BLE001 - sidecar is best-effort
        log.debug("animation sidecar write failed: %s", exc)


def canary_report(total_tokens: int | None = None) -> dict[str, Any]:
    """End-of-run canary (prereg sec3). Prints K-A1..K-A4 evidence and returns
    the same numbers so a notebook cell can assert on them."""
    engaged = sorted(
        g for g, s in COUNTERS.by_game.items() if s["invisible"] > 0
    )
    audit_hits = sorted(
        g for g in engaged if any(g.startswith(p) for p in _AUDIT_TYPE1_GAMES)
    )
    est_tokens = COUNTERS.summaries * _TOKENS_PER_SUMMARY
    frac = (est_tokens / total_tokens) if total_tokens else None
    report = {
        "version": VERSION,
        "actions": COUNTERS.actions,
        "multi_frame_actions": COUNTERS.multi,
        "invisible_actions": COUNTERS.invisible,
        "summaries_emitted": COUNTERS.summaries,
        "errors": COUNTERS.errors,
        "games_with_events": sorted(
            g for g, s in COUNTERS.by_game.items() if s["multi"] > 0),
        "games_with_invisible": engaged,
        "audit_type1_games_engaged": audit_hits,
        "animation_tokens_est": est_tokens,
        "animation_token_fraction": frac,
    }
    print(
        f"ANIMATION CANARY v={_EVENT_V} version={VERSION} "
        f"actions={report['actions']} multi={report['multi_frame_actions']} "
        f"invisible={report['invisible_actions']} "
        f"summaries={report['summaries_emitted']} errors={report['errors']} "
        f"games_with_events={len(report['games_with_events'])} "
        f"games_with_invisible={len(engaged)} "
        f"audit_type1_engaged={','.join(audit_hits) or 'NONE'} "
        f"tokens_est={est_tokens} "
        f"token_fraction={'' if frac is None else round(frac, 6)}",
        flush=True,
    )
    return report


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

    # --- seam 1: per-action summary ---------------------------------------
    _orig_execute_action = solver_mod._HarnessGameSession._execute_action

    def animation_execute_action(self, action, *, batch_index, batch_size,
                                 generated_tokens=None, flush_viewer_payload=True):
        previous_grid = None
        try:
            previous_grid = solver_mod._grid_from_state(self.game.current_state)
        except Exception as exc:  # noqa: BLE001 - never break the action path
            COUNTERS.errors += 1
            log.debug("animation: previous grid failed: %s", exc)

        payload = _orig_execute_action(
            self, action,
            batch_index=batch_index, batch_size=batch_size,
            generated_tokens=generated_tokens,
            flush_viewer_payload=flush_viewer_payload,
        )

        try:
            state = self.game.current_state
            raw_frames = getattr(getattr(state, "raw", None), "frame", None)
            summary = summarize_animation(raw_frames, previous_grid)
            game = _game_label(self)
            slot = COUNTERS.game(game)
            COUNTERS.actions += 1
            slot["actions"] += 1
            if summary is not None:
                COUNTERS.multi += 1
                slot["multi"] += 1
                if summary["board_unchanged"]:
                    COUNTERS.invisible += 1
                    slot["invisible"] += 1
                if _only_invisible() and not summary["board_unchanged"]:
                    return payload
                payload["animation"] = summary
                payload["animation_note"] = animation_note(summary)
                COUNTERS.summaries += 1
                # Feed the batch sink opened by seam 2 (absent -> no-op, so
                # this path stays valid for _execute_auto_reset).
                sink = getattr(self, "_animation_batch", None)
                if isinstance(sink, list):
                    sink.append(summary)
                _emit_event(self, game, summary,
                            str(payload.get("action_display") or action.id.name))
        except Exception as exc:  # noqa: BLE001 - never break the action path
            COUNTERS.errors += 1
            log.debug("animation: summary failed: %s", exc)
        return payload

    solver_mod._HarnessGameSession._execute_action = animation_execute_action
    patched += 1

    # --- seam 2: batch merge ----------------------------------------------
    _orig_step_env = solver_mod._HarnessGameSession.step_env

    def animation_step_env(self, arguments):
        # Vanilla builds final_payload from the LAST executed payload only, so
        # a batch drops every animation but the last. Seam 1 appends each
        # per-action summary into this per-call sink; we merge them back.
        self._animation_batch = []
        try:
            final_payload = _orig_step_env(self, arguments)
        finally:
            batch = getattr(self, "_animation_batch", None)
            self._animation_batch = None
        try:
            if isinstance(final_payload, dict) and batch:
                merged = merge_animations(batch)
                if merged is not None:
                    final_payload["animation"] = merged
                    final_payload["animation_note"] = animation_note(merged)
        except Exception as exc:  # noqa: BLE001
            COUNTERS.errors += 1
            log.debug("animation: batch merge failed: %s", exc)
        return final_payload

    solver_mod._HarnessGameSession.step_env = animation_step_env
    patched += 1

    # --- seam 3a: carry the field through the compactor -------------------
    _orig_compact = ToolAgent._compact_action_result

    def animation_compact_action_result(self, payload):
        compact = _orig_compact(self, payload)
        try:
            if isinstance(payload, dict) and isinstance(payload.get("animation"), dict):
                compact["animation"] = payload["animation"]
                note = payload.get("animation_note")
                if note:
                    compact["animation_note"] = note
        except Exception as exc:  # noqa: BLE001
            COUNTERS.errors += 1
            log.debug("animation: compact passthrough failed: %s", exc)
        return compact

    ToolAgent._compact_action_result = animation_compact_action_result
    patched += 1

    # --- seam 3b: fix the aliased outcome sentence ------------------------
    _orig_summarize_step = ToolAgent._summarize_step_sequence
    _orig_describe = ToolAgent._describe_last_outcome

    def animation_summarize_step_sequence(self, action_results):
        summary = _orig_summarize_step(self, action_results)
        try:
            if isinstance(summary, dict):
                anims = [
                    item.get("animation") for item in action_results
                    if isinstance(item, dict) and isinstance(item.get("animation"), dict)
                ]
                merged = merge_animations([a for a in anims if a])
                if merged is not None:
                    summary["animation"] = merged
        except Exception as exc:  # noqa: BLE001
            COUNTERS.errors += 1
            log.debug("animation: step summary failed: %s", exc)
        return summary

    def animation_describe_last_outcome(self, summary):
        text = _orig_describe(self, summary)
        if not _outcome_text():
            return text
        try:
            if isinstance(summary, dict) and not summary.get("board_changed"):
                anim = summary.get("animation")
                if isinstance(anim, dict) and anim.get("board_unchanged"):
                    return (
                        f"{text} HOWEVER the engine rendered "
                        f"{anim.get('frames')} frames for it and "
                        f"{anim.get('transient_cells')} cell(s) changed "
                        f"mid-animation before the board returned to its prior "
                        f"state  -  the action was NOT inert; something happened "
                        f"and was undone or consumed. Do not file it as "
                        f"'no effect'."
                    )
        except Exception as exc:  # noqa: BLE001
            COUNTERS.errors += 1
            log.debug("animation: outcome text failed: %s", exc)
        return text

    ToolAgent._summarize_step_sequence = animation_summarize_step_sequence
    ToolAgent._describe_last_outcome = animation_describe_last_outcome
    patched += 1

    # --- state_path binding for the per-game sidecars ---------------------
    _orig_analyze = ToolAgent.analyze

    def animation_analyze(self, state_path, *args, **kwargs):
        try:
            self._animation_state_path = Path(state_path)
        except Exception as exc:  # noqa: BLE001
            log.debug("animation: state_path bind failed: %s", exc)
        return _orig_analyze(self, state_path, *args, **kwargs)

    ToolAgent.analyze = animation_analyze

    _APPLIED = True
    return patched


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def apply(bm: Any = None) -> bool:
    """Install animation-awareness v1. Returns True on success (or already
    applied), False on flag-off / kill switch / any failure  -  in which case
    NOTHING is changed (vanilla fallback: stock duck harness).

    Flag gate:   ANIMATION_AWARE=1   (the arm flag)
    Kill switch: ANIMATION_DISABLE=1 -> no-op, returns False
    Sub-flags:   ANIMATION_ONLY_INVISIBLE=1 -> emit only the aliased case
                 ANIMATION_OUTCOME_TEXT=0   -> leave seam 3b vanilla
    """
    if _kill_switch():
        log.info("animation %s: ANIMATION_DISABLE=1 -> no-op", VERSION)
        return False
    if not _flag_on():
        log.info("animation %s: ANIMATION_AWARE!=1 -> no-op (flag-gated arm)", VERSION)
        return False
    try:
        patched = _apply_patches()
        if bm is not None and hasattr(bm, "label"):
            bm.label = f"{bm.label}-animation-{VERSION}"
        print(
            f"animation {VERSION}: ACTIVE ({patched} seams patched)  -  "
            f"per-action intermediate-frame summary from GameState.raw.frame "
            f"(taaf/game.py:170 discards all but frame[-1]; zero prior consumers). "
            f"Fixed scalar schema, NO raw frames, ~{_TOKENS_PER_SUMMARY} tok, "
            f"emitted only on animated actions. "
            f"only_invisible={'ON' if _only_invisible() else 'OFF (default)'}; "
            f"outcome_text={'ON (default)' if _outcome_text() else 'OFF'}; "
            f"NO no-op guard (prereg sec2.2: separately gated, downstream); "
            f"zero LLM calls, no locks, game-agnostic",
            flush=True,
        )
        log.info("animation %s installed (%d seams)", VERSION, patched)
        return True
    except Exception:  # noqa: BLE001 - any failure -> vanilla fallback
        log.exception("[animation] apply failed -> stock duck harness (vanilla)")
        return False
