#!/usr/bin/env python3
"""One-shot: add T12 (cross-game leakage) and T13 (durable report) to the P2
trigger smoke, and give the method extractor the module-level helpers it needs.

Idempotent: no-ops if T12 is already present.
"""
from pathlib import Path

P = Path(__file__).resolve().parent / "p2_trigger_smoke.py"
t = P.read_text(encoding="utf-8")

if "T12a" in t:
    print("already applied")
    raise SystemExit(0)

old = """    wanted = {
        "_p2_state","""
new = """    wanted = {
        "_p2_game_key",
        "_p2_flush",
        "_p2_state","""
assert t.count(old) == 1, "wanted-set anchor"
t = t.replace(old, new)

old2 = """    ns: dict = {}
    exec(compile(module, "<p2-methods>", "exec"), ns)  # noqa: S102"""
new2 = '''    # The methods reference module-level names from tool_agent.py. Supply the
    # REAL _resolve_run_artifact_location by exec-ing its definition out of the
    # same patched source -- a hand-written stand-in would let the game-key
    # logic pass here and still be wrong on the rail.
    ns: dict = {"Path": Path, "json": __import__("json")}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_run_artifact_location":
            helper = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(helper)
            exec(compile(helper, "<p2-helpers>", "exec"), ns)  # noqa: S102
    ns.setdefault("RUNTIME_STATE_FILENAME", "runtime_state.json")
    exec(compile(module, "<p2-methods>", "exec"), ns)  # noqa: S102'''
assert t.count(old2) == 1, "ns anchor"
t = t.replace(old2, new2)

NEW_TESTS = '''        print("")
        print("-- T12: NEGATIVE CONTROL -- state must NOT leak across games --")
        # The shipped layout can place every game's runtime_state in ONE
        # `artifacts` dir, so _ensure_session (which keys on state_path.parent)
        # does NOT reset between games there. If the P2 counter inherited that
        # key, `cleared` would accumulate across the benchmark and retry would
        # be permanently DISABLED after the 4th clear anywhere in the run --
        # a silently-dead arm, the exact hard_noop_guard failure class.
        arts = tmp / "run1" / "artifacts"
        arts.mkdir(parents=True, exist_ok=True)
        g1 = arts / "aaa11111_runtime_state.json"
        g2 = arts / "bbb22222_runtime_state.json"
        for g in (g1, g2):
            g.write_text("{}", encoding="utf-8")
        check("T12a the two games SHARE a parent dir (the risky layout)",
              g1.parent == g2.parent)
        a = Agent()
        for lvl in range(1, RETRY_DISABLED_AT_LEVELS + 1):
            a._p2_note_acting_turn({"level": lvl, "level_transition": True}, g1)
        check("T12b game 1 reached the disable threshold",
              a._p2_state(g1)["cleared"] == RETRY_DISABLED_AT_LEVELS,
              str(a._p2_state(g1)["cleared"]))
        for _ in range(20):
            a._p2_note_acting_turn({"level": 9, "level_transition": False}, g1)
        check("T12c game 1 correctly REFUSES (k>=%d)" % RETRY_DISABLED_AT_LEVELS,
              a._p2_is_armed() is False)
        for _ in range(H_STUCK_TURNS):
            a._p2_note_acting_turn({"level": 1, "level_transition": False}, g2)
        check("T12d game 2 starts from a CLEAN counter",
              a._p2_state(g2)["cleared"] == 0, str(a._p2_state(g2)["cleared"]))
        check("T12e game 2 ARMS despite game 1 being disabled",
              a._p2_is_armed() is True)
        check("T12f game 2 did not inherit game 1's acting turns",
              a._p2_state(g2)["acting_turns"] == H_STUCK_TURNS,
              str(a._p2_state(g2)["acting_turns"]))

        print("")
        print("-- T13: the D2 report lands in the JOB DIR, not only the log --")
        # P1 COMPLETED, was pulled TWICE, and its kernel log was 0 BYTES both
        # times; its certification was defined on log markers and was therefore
        # UNEVALUABLE. A job-dir file survives exactly that.
        a = Agent()
        for _ in range(H_STUCK_TURNS):
            a._p2_note_acting_turn({"level": 1, "level_transition": False}, g1)
        a._p2_count_attempt_calls("attempt([1])", g1)
        armed_now = a._p2_retry_armed(g1)
        check("T13a armed, so a report is due", armed_now is True)
        rep_path = tmp / "run1" / "p2" / "aaa11111.json"
        check("T13b report file written", rep_path.is_file(),
              str([str(x) for x in (tmp / "run1").rglob("*.json")][:6]))
        if rep_path.is_file():
            import json as _json
            body = _json.loads(rep_path.read_text(encoding="utf-8"))
            check("T13c report carries d2_use_rate", "d2_use_rate" in body, str(sorted(body)))
            check("T13d report carries the armed attempt count",
                  body.get("attempt_calls_armed") == 1, str(body.get("attempt_calls_armed")))
            check("T13e report carries the sealed H", body.get("H") == H_STUCK_TURNS)
        check("T13f flush never raises on an unwritable path",
              a._p2_flush("") is None)

'''

lines = t.splitlines(keepends=True)
idx = next(i for i, ln in enumerate(lines) if "-- T11: DRIFT negative control" in ln)
lines.insert(idx, NEW_TESTS)
P.write_text("".join(lines), encoding="utf-8")
print("smoke extended; T12/T13 inserted before line", idx + 1)
