#!/usr/bin/env python3
"""One-shot generator: append the P2 trigger-leg anchors (3, 4) to p2_patch.py.

Idempotent: refuses to run twice (checks for the ANCHOR 3 banner).
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "p2_patch.py"
t = P.read_text(encoding="utf-8")

if "ANCHOR 3" in t:
    print("already applied; nothing to do")
    raise SystemExit(0)

Q = "'''"

TRIGGER = '''
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
    '        step_executed = any(bool(item.get("executed")) for item in action_results)\\n'
    "        if step_executed:\\n"
    "            self._last_step_summary = self._summarize_step_sequence(action_results)\\n"
    "            self._update_summarized_knowledge_from_step_summary()\\n"
    "        return _ToolDispatchResult(\\n"
)

_AGENT_STEP_REPLACEMENT = (
    '        step_executed = any(bool(item.get("executed")) for item in action_results)\\n'
    "        if step_executed:\\n"
    "            self._last_step_summary = self._summarize_step_sequence(action_results)\\n"
    "            self._update_summarized_knowledge_from_step_summary()\\n"
    "            self._p2_note_acting_turn(self._last_step_summary, state_path)\\n"
    "        if self._p2_retry_armed(state_path):\\n"
    '            payload["retry_mode"] = "on"\\n'
    '            payload["episodes_available"] = %d\\n'
    "        return _ToolDispatchResult(\\n"
) % K_EPISODES

# The counter and the D2 instrument, as methods on the same class. Inserted
# immediately before _dispatch_tool, which is unique in the vehicle.
_ANCHOR_DISPATCH = (
    "    def _dispatch_tool(self, state_path: Path, name: str, arguments: dict[str, Any])"
    " -> _ToolDispatchResult:\\n"
)

_P2_METHODS = QQQ    # ---- [p2] reset-anchored episodic retry: trigger + D2 use instrument ----

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

QQQ % (
    RETRY_DISABLED_AT_LEVELS,
    H_STUCK_TURNS,
    H_STUCK_TURNS,
    K_EPISODES,
    EPISODE_ACTION_CAP,
)

_DISPATCH_REPLACEMENT = _P2_METHODS + _ANCHOR_DISPATCH

# The D2 counter must see the code the model actually submitted.
_ANCHOR_CODE_READ = '        code = str(arguments.get("code", "")).rstrip()\\n'
_CODE_READ_REPLACEMENT = (
    '        code = str(arguments.get("code", "")).rstrip()\\n'
    "        self._p2_count_attempt_calls(code, state_path)\\n"
)

# --------------------------------------------------------------------------
# ANCHOR 4 -- announce attempt() WHERE THE MODEL ACTUALLY READS: the tool
# description, which is where action() is announced too (prereg S3.2, and
# feedback_advertise_where_model_reads.md).
# --------------------------------------------------------------------------

_ANCHOR_TOOL_DESC = (
    '    "Use `print(...)` for compact output or assign final data to `result`."\\n)'
)

_TOOL_DESC_REPLACEMENT = (
    '    "Use `print(...)` for compact output or assign final data to `result`. "\\n'
    '    "`attempt(actions)` runs a candidate action sequence from the CURRENT LEVEL START, "\\n'
    '    "reports what it reached, then RESETs back to that same level start so the level is "\\n'
    '    "left unchanged -- letting you test several candidate plans in ONE turn instead of "\\n'
    '    "committing to one. It costs actions, which are cheap, and no extra turn. It never "\\n'
    '    "RESETs when the sequence clears the level: in that case the clear STANDS and it "\\n'
    '    "reports level_completed. Episodes are capped at %d actions. When the tool result "\\n'
    '    "shows `retry_mode: on`, `episodes_available` candidate sequences are offered now."\\n)'
) % EPISODE_ACTION_CAP

'''.replace("QQQ", Q)

marker = "def _read(path: Path) -> str:"
assert t.count(marker) == 1, "marker not unique"
t = t.replace(marker, TRIGGER.lstrip("\n") + "\n" + marker)

# Extend apply_patch to apply anchors 3 and 4 and report them.
old_tail = '''    ast.parse(text)

    sandbox.write_text(text, encoding="utf-8")

    return {
        "anchors_applied": 2,'''
new_tail = '''    ast.parse(text)

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
        "anchors_applied": 6,'''
assert t.count(old_tail) == 1, "apply_patch tail not unique"
t = t.replace(old_tail, new_tail)

P.write_text(t, encoding="utf-8")
print("p2_patch.py extended ->", len(t.splitlines()), "lines")
