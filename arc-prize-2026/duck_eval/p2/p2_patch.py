#!/usr/bin/env python3
"""P2 reset-anchored episodic retry -- the working-copy patch.

Prereg: ``learnings/war_room/p2_reset_retry_prereg_2026-08-22.md`` (SEALED 2026-08-22).
Gate P0.1 (RESET returns to level start) discharged on the real simulator, see prereg S2.

WHAT THIS PATCHES, AND WHY IT IS SMALLER THAN THE PREREG ASSUMED
---------------------------------------------------------------
The prereg's S3.4 anchor list assumed ``attempt()`` needed a host-side handler wired
through ``tool_agent.py`` alongside ``action()``.  Reading the shipped sandbox protocol
shows it does not: ``python_tool_sandbox.py`` runs the snippet in a CHILD process that
talks to the host over JSON lines, and ``action(...)`` is already a complete round-trip
primitive (child sends ``{"type":"action"}``, host replies ``action_result`` + refreshed
``state``).  ``RESET`` is a first-class action name (``inference/agent/action_names.py:14``)
and ``taaf/game.py:184`` guarantees it is *always* legal ("Legal action ids, with RESET (0)
always present").

Therefore ``attempt(seq)`` is composed ENTIRELY from the existing ``action()`` primitive,
inside the child process.  No new message type, no new host handler, no ``tool_agent.py``
change for the episode machinery itself.

This matters for bundle-drift risk, which is the campaign's most expensive failure class:

  * ``inference/agent/python_tool_sandbox.py`` is BYTE-IDENTICAL between the
    ``anim-20260807`` vehicle bundle and the vendored ``bundle_20260815``
    (md5 465f3e4fb9b1 both).  The episode patch lands only here.
  * ``inference/agent/tool_agent.py`` DIFFERS between those bundles (233 diff lines;
    different behaviour-flag architecture, different ``_PYTHON_TOOL_DESCRIPTION``).
    The prereg's anchors were verified against 08-15; the VEHICLE is anim-20260807.
    Anchors touching this file are therefore re-verified here against the vehicle.

Every anchor is asserted ``count == 1`` at apply time.  Any drift dies LOUDLY
(``P2FatalDrift``) -- never a silent stock run.  That is the sealed contract.

PARAMETERS (prereg S3.3, fixed pre-data, not tunable here)
    H   = 4    consecutive acting turns on one level without a clear -> retry arms
    K   = 5    episodes offered per retry turn
    CAP = 40   actions per episode
    retry disabled once k >= 4 levels cleared on that game
    RESET-after-WIN: NEVER (the engine full-resets after WIN)
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

H_STUCK_TURNS = 4
K_EPISODES = 5
EPISODE_ACTION_CAP = 40
RETRY_DISABLED_AT_LEVELS = 4

# The vehicle bundle. Certification item 4 requires anim-20260807.
VEHICLE_SANDBOX_MD5 = "465f3e4fb9b1"  # first 12 of md5(python_tool_sandbox.py)


class P2FatalDrift(RuntimeError):
    """Raised when an anchor is not present exactly once. Never swallowed."""


# --------------------------------------------------------------------------
# ANCHOR 1 -- python_tool_sandbox.py: define attempt() and export it.
# The anchor is the export line for action(); we insert the definition just
# before it and the export just after, so both land at the correct scope.
# --------------------------------------------------------------------------

_ANCHOR_SANDBOX_EXPORT = '        runtime_globals["action"] = action\n'

# NOTE: the injected source lands INSIDE the child bootstrap, which is an r""" literal.
# Triple-double-quoted docstrings would terminate that literal, so every explanatory line
# below is a # comment. (The smoke's ast.parse caught exactly this.)
_ATTEMPT_SRC = '''        def _p2_board_sig(frame):
            # Cheap, stable signature of a frame for the reset-to-level-start check.
            if frame is None:
                return None
            ascii_art = getattr(frame, "ascii", None)
            if ascii_art is None:
                return None
            try:
                text = ascii_art if isinstance(ascii_art, str) else str(ascii_art)
            except Exception:  # noqa: BLE001
                return None
            return hashlib.md5(text.encode("utf-8", "replace")).hexdigest()[:16]

        def attempt(seq):
            # Evaluate a candidate plan from the current level start, then RESET back.
            # Costs actions; costs no turn.  Returns a compact summary so one LLM turn
            # can price several independent plans instead of committing to one.
            # RESET-after-WIN is refused: if the episode clears the level we leave the
            # game advanced and report level_completed=True (the engine full-resets
            # after WIN, which would throw the level away).
            normalized = _normalize_actions(seq)
            if len(normalized) > P2_EPISODE_ACTION_CAP:
                raise ValueError(
                    "attempt() accepts at most %d actions; got %d."
                    % (P2_EPISODE_ACTION_CAP, len(normalized))
                )
            for item in normalized:
                if str(item.get("action", "")).strip().upper() == "RESET":
                    raise ValueError(
                        "attempt() issues its own RESET; do not include RESET in the sequence."
                    )

            start_frame = runtime_globals.get("current_frame")
            start_level = getattr(start_frame, "level", None)
            start_sig = _p2_board_sig(start_frame)

            taken = 0
            level_completed = False
            terminal_reason = "sequence_exhausted"
            total_reward = 0.0
            last_result = {}

            for item in normalized:
                last_result = action([item]) or {}
                taken += 1
                try:
                    total_reward += float(last_result.get("reward") or 0.0)
                except Exception:  # noqa: BLE001
                    pass
                if not last_result.get("executed", True):
                    terminal_reason = str(last_result.get("stop_reason") or "not_executed")
                    break
                if bool(last_result.get("level_completed")):
                    level_completed = True
                    terminal_reason = "level_completed"
                    break
                if bool(last_result.get("game_over")):
                    terminal_reason = "game_over"
                    break
                if bool(last_result.get("run_complete")) or bool(last_result.get("done")):
                    terminal_reason = "run_complete"
                    break

            # NEVER reset after a level clear -- the engine full-resets after WIN.
            reset_issued = False
            returned_to_start = None
            if not level_completed and terminal_reason != "run_complete":
                action(["RESET"])
                reset_issued = True
                end_sig = _p2_board_sig(runtime_globals.get("current_frame"))
                returned_to_start = (
                    None if (end_sig is None or start_sig is None) else bool(end_sig == start_sig)
                )

            end_frame = runtime_globals.get("current_frame")
            board_delta = "level %s -> %s after %d action(s); %s" % (
                start_level,
                getattr(end_frame, "level", None),
                taken,
                terminal_reason,
            )
            return {
                "level_completed": level_completed,
                "reward": total_reward,
                "actions_taken": taken,
                "terminal_reason": terminal_reason,
                "board_delta": board_delta[:200],
                "reset_issued": reset_issued,
                "returned_to_level_start": returned_to_start,
            }

'''

_SANDBOX_REPLACEMENT = _ATTEMPT_SRC + _ANCHOR_SANDBOX_EXPORT + '        runtime_globals["attempt"] = attempt\n'

# ANCHOR 2 -- the CHILD bootstrap needs hashlib + the cap constant.
#
# The child source is `_SANDBOX_BOOTSTRAP = textwrap.dedent(r"""...` with a 4-space
# base indentation that dedent() strips.  The anchor therefore carries that base
# indentation VERBATIM: the bare "import contextlib\n" is a substring of the indented
# line, so it would still count == 1 and pass the assert, but the replacement would
# splice column-0 lines into an indented block -> IndentationError at kernel import.
# The indentation is part of the anchor, not incidental to it.
_ANCHOR_SANDBOX_IMPORT = "    import contextlib\n"
_SANDBOX_IMPORT_REPLACEMENT = (
    "    import contextlib\n"
    "    import hashlib\n"
    "    P2_EPISODE_ACTION_CAP = %d  # [p2] sealed episode cap\n" % EPISODE_ACTION_CAP
)


# --------------------------------------------------------------------------
# ANCHOR 3 -- tool_agent.py: the STUCK TRIGGER + the D2 USE instrument.
#
# Sealed definition (prereg S3.2/S3.3): after H consecutive ACTING turns on the
# same level with no `level_completed`, the python tool result carries
# `retry_mode: on, episodes_available: K`.  Retry is disabled once k >= 4
# levels have cleared on that game (S3.3).
#
# FIREABILITY, MEASURED BEFORE THE BUILD (learnings/war_room/
# p2_trigger_fireability_2026-08-26.md): 19/25 games on the certified field
# floor -- the arm's own vehicle -- against a sealed D1 bar of >=15/25, and
# >=15/25 on four independent real corpora.  H would have to be raised past 7
# before delivery reaches the bar.  The trigger CAN fire; that is established,
# not assumed.  Negative control on file: 6/25 games correctly REFUSE, and they
# are exactly the prompt clearers (sb26 clears 7 levels and never arms).
#
# D2 IS THE REAL RISK AND IS INSTRUMENTED HERE, NOT INFERRED LATER.
# `feedback_advertise_where_model_reads.md`: a schema-only affordance delivered
# at 96.3% and got 1.3% USE against a 30% bar.  P1 died on exactly this and its
# read was unevaluable because nothing counted CALLS.  So we count calls by AST
# over the model's submitted code -- `attempt(...)` as a real Call node, not a
# substring a comment or docstring could fake -- and split the count by whether
# retry_mode was on when that code was submitted.
# --------------------------------------------------------------------------

_ANCHOR_AGENT_STEP = (
    '        step_executed = any(bool(item.get("executed")) for item in action_results)\n'
    "        if step_executed:\n"
    "            self._last_step_summary = self._summarize_step_sequence(action_results)\n"
    "            self._update_summarized_knowledge_from_step_summary()\n"
    "        return _ToolDispatchResult(\n"
)

_AGENT_STEP_REPLACEMENT = (
    '        step_executed = any(bool(item.get("executed")) for item in action_results)\n'
    "        if step_executed:\n"
    "            self._last_step_summary = self._summarize_step_sequence(action_results)\n"
    "            self._update_summarized_knowledge_from_step_summary()\n"
    "            self._p2_note_acting_turn(self._last_step_summary, state_path)\n"
    "        if self._p2_retry_armed(state_path):\n"
    '            payload["retry_mode"] = "on"\n'
    '            payload["episodes_available"] = %d\n'
    "        return _ToolDispatchResult(\n"
) % K_EPISODES

# The counter and the D2 instrument, as methods on the same class. Inserted
# immediately before _dispatch_tool, which is unique in the vehicle.
_ANCHOR_DISPATCH = (
    "    def _dispatch_tool(self, state_path: Path, name: str, arguments: dict[str, Any])"
    " -> _ToolDispatchResult:\n"
)

_P2_METHODS = '''    # ---- [p2] reset-anchored episodic retry: trigger + D2 use instrument ----

    def _p2_game_key(self, state_path) -> tuple:
        """Per-GAME identity. _ensure_session keys on state_path.parent, but the
        shipped layout can put every game's runtime_state in ONE `artifacts`
        dir (_resolve_run_artifact_location globs `*_runtime_state` and only
        derives a game stem when there is more than one). In that layout the
        parent dir is IDENTICAL across games, so keying the counter on it would
        carry `cleared` across the whole benchmark and permanently disable
        retry after the 4th clear anywhere. Key on the resolved (root, stem)."""
        try:
            root, stem = _resolve_run_artifact_location(Path(state_path))
        except Exception:  # noqa: BLE001
            return (str(state_path),)
        return (str(root), stem or Path(state_path).stem)

    def _p2_state(self, state_path=None) -> dict:
        if state_path is not None:
            key = self._p2_game_key(state_path)
            if getattr(self, "_p2_key", None) != key:
                self._p2_key = key
                self._p2 = None
        st = getattr(self, "_p2", None)
        if st is None:
            st = {
                "run": 0,             # consecutive acting turns on one uncleared level
                "level": None,        # the level that run is on
                "cleared": 0,         # levels cleared this game (retry disables at 4)
                "armed_turns": 0,     # acting turns where retry_mode was emitted
                "acting_turns": 0,
                "attempt_calls_armed": 0,
                "attempt_calls_unarmed": 0,
                "turns_calling_attempt_armed": 0,
                "ever_armed": False,
                "max_run": 0,
            }
            self._p2 = st
        return st

    def _p2_note_acting_turn(self, summary, state_path=None) -> None:
        """Increment/reset the stuck counter. Sealed definition: H consecutive
        ACTING turns on the SAME level with no level_completed."""
        st = self._p2_state(state_path)
        st["acting_turns"] += 1
        if not isinstance(summary, dict):
            return
        level = summary.get("level")
        if bool(summary.get("level_transition")):
            st["cleared"] += 1
            st["run"] = 0
            st["level"] = level
            return
        if level != st["level"]:
            st["level"] = level
            st["run"] = 1
        else:
            st["run"] += 1
        if st["run"] > st["max_run"]:
            st["max_run"] = st["run"]

    def _p2_is_armed(self, state_path=None) -> bool:
        """Pure predicate -- no bookkeeping, safe to call anywhere."""
        st = self._p2_state(state_path)
        return st["cleared"] < %d and st["run"] >= %d

    def _p2_retry_armed(self, state_path=None) -> bool:
        """Predicate + emission bookkeeping. Called once per tool result."""
        st = self._p2_state(state_path)
        armed = self._p2_is_armed()
        if armed:
            st["armed_turns"] += 1
            st["ever_armed"] = True
        if state_path is not None:
            self._p2_flush(state_path)
        return armed

    def _p2_count_attempt_calls(self, code: str, state_path=None) -> None:
        """D2: count REAL attempt(...) calls in the model's submitted code.

        AST, not substring: a mention inside a comment, string or docstring is
        not a use. Split by whether retry_mode was on, because D2 is defined
        over retry-mode turns.
        """
        import ast as _ast

        st = self._p2_state(state_path)
        armed = self._p2_is_armed()
        try:
            tree = _ast.parse(code)
        except SyntaxError:
            return
        n = 0
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                fn = node.func
                if isinstance(fn, _ast.Name) and fn.id == "attempt":
                    n += 1
        if not n:
            return
        if armed:
            st["attempt_calls_armed"] += n
            st["turns_calling_attempt_armed"] += 1
        else:
            st["attempt_calls_unarmed"] += n

    def _p2_flush(self, state_path) -> None:
        """Write the D2 report to the JOB DIR after every armed turn.

        P1 COMPLETED, was pulled twice, and its kernel log was 0 BYTES on both
        pulls -- its sealed certification was defined on log markers and was
        therefore UNEVALUABLE. execwm survived the same class only because its
        scorer read job-dir report files first. So this arm does not rely on
        stdout: it writes a small JSON per game, overwritten in place, so the
        read survives a truncated log and a mid-run crash alike.
        """
        try:
            root, stem = _resolve_run_artifact_location(Path(state_path))
            out = Path(root) / "p2"
            out.mkdir(parents=True, exist_ok=True)
            name = (stem or Path(state_path).stem) + ".json"
            (out / name).write_text(
                json.dumps(self._p2_report(), indent=1, default=str), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001
            pass  # never let instrumentation kill a run

    def _p2_report(self) -> dict:
        st = dict(self._p2_state())
        armed = st["armed_turns"]
        st["d2_use_rate"] = (st["turns_calling_attempt_armed"] / armed) if armed else None
        st["H"] = %d
        st["K"] = %d
        st["cap"] = %d
        return st

''' % (
    RETRY_DISABLED_AT_LEVELS,
    H_STUCK_TURNS,
    H_STUCK_TURNS,
    K_EPISODES,
    EPISODE_ACTION_CAP,
)

_DISPATCH_REPLACEMENT = _P2_METHODS + _ANCHOR_DISPATCH

# The D2 counter must see the code the model actually submitted.
_ANCHOR_CODE_READ = '        code = str(arguments.get("code", "")).rstrip()\n'
_CODE_READ_REPLACEMENT = (
    '        code = str(arguments.get("code", "")).rstrip()\n'
    "        self._p2_count_attempt_calls(code, state_path)\n"
)

# --------------------------------------------------------------------------
# ANCHOR 4 -- announce attempt() WHERE THE MODEL ACTUALLY READS: the tool
# description, which is where action() is announced too (prereg S3.2, and
# feedback_advertise_where_model_reads.md).
# --------------------------------------------------------------------------

_ANCHOR_TOOL_DESC = (
    '    "Use `print(...)` for compact output or assign final data to `result`."\n)'
)

_TOOL_DESC_REPLACEMENT = (
    '    "Use `print(...)` for compact output or assign final data to `result`. "\n'
    '    "`attempt(actions)` runs a candidate action sequence from the CURRENT LEVEL START, "\n'
    '    "reports what it reached, then RESETs back to that same level start so the level is "\n'
    '    "left unchanged -- letting you test several candidate plans in ONE turn instead of "\n'
    '    "committing to one. It costs actions, which are cheap, and no extra turn. It never "\n'
    '    "RESETs when the sequence clears the level: in that case the clear STANDS and it "\n'
    '    "reports level_completed. Episodes are capped at %d actions. When the tool result "\n'
    '    "shows `retry_mode: on`, `episodes_available` candidate sequences are offered now."\n)'
) % EPISODE_ACTION_CAP


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sub_once(text: str, anchor: str, replacement: str, *, where: str) -> str:
    count = text.count(anchor)
    if count != 1:
        raise P2FatalDrift(
            "P2 FATAL: anchor count != 1 (got %d) in %s for anchor %r" % (count, where, anchor[:70])
        )
    return text.replace(anchor, replacement)


def apply_patch(src_root: Path) -> dict:
    """Apply the P2 patch to a working-copy shadow of ARC3-Inference.

    ``src_root`` must be the directory that contains ``inference/``.
    Returns a dict of applied-anchor bookkeeping. Raises P2FatalDrift on any drift.
    """
    src_root = Path(src_root)
    sandbox = src_root / "inference" / "agent" / "python_tool_sandbox.py"
    if not sandbox.is_file():
        raise P2FatalDrift("P2 FATAL: %s not found" % sandbox)

    original = _read(sandbox)
    digest = hashlib.md5(original.encode("utf-8")).hexdigest()[:12]

    text = _sub_once(
        original, _ANCHOR_SANDBOX_IMPORT, _SANDBOX_IMPORT_REPLACEMENT, where="sandbox:imports"
    )
    text = _sub_once(
        text, _ANCHOR_SANDBOX_EXPORT, _SANDBOX_REPLACEMENT, where="sandbox:action-export"
    )

    # Must compile, or the kernel dies at import with no diagnosis.
    ast.parse(text)

    sandbox.write_text(text, encoding="utf-8")

    # ---- anchors 3 + 4: the trigger leg, in tool_agent.py ----
    agent = src_root / "inference" / "agent" / "tool_agent.py"
    if not agent.is_file():
        raise P2FatalDrift("P2 FATAL: %s not found" % agent)
    atext = _read(agent)
    atext = _sub_once(
        atext, _ANCHOR_CODE_READ, _CODE_READ_REPLACEMENT, where="agent:code-read"
    )
    atext = _sub_once(
        atext, _ANCHOR_AGENT_STEP, _AGENT_STEP_REPLACEMENT, where="agent:step-summary"
    )
    atext = _sub_once(
        atext, _ANCHOR_DISPATCH, _DISPATCH_REPLACEMENT, where="agent:dispatch-methods"
    )
    atext = _sub_once(
        atext, _ANCHOR_TOOL_DESC, _TOOL_DESC_REPLACEMENT, where="agent:tool-description"
    )
    ast.parse(atext)
    agent.write_text(atext, encoding="utf-8")

    return {
        "anchors_applied": 6,
        "sandbox_md5_before": digest,
        "sandbox_is_vehicle_generation": digest == VEHICLE_SANDBOX_MD5,
        "H": H_STUCK_TURNS,
        "K": K_EPISODES,
        "cap": EPISODE_ACTION_CAP,
        "banner": "[p2] reset-retry armed H=%d K=%d cap=%d"
        % (H_STUCK_TURNS, K_EPISODES, EPISODE_ACTION_CAP),
    }


if __name__ == "__main__":  # pragma: no cover
    import sys

    print(apply_patch(Path(sys.argv[1])))
