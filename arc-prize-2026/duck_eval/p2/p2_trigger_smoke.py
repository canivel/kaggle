#!/usr/bin/env python3
"""P2 TRIGGER-LEG smoke -- the stuck counter and the D2 use instrument.

Companion to ``p2_smoke.py`` (which covers the ``attempt()`` episode leg, 18/18).
This file covers ANCHORS 3 and 4: the H=4 stuck trigger, the ``retry_mode`` line
on the tool result, the ``attempt()`` announcement in the tool description, and
the D2 call counter.

WHY IT EXISTS. ``hard_noop_guard`` shipped armed and fired 0 times in 5,255 real
actions, and P1 shipped a patch that DELIVERED at 96.3% and got 1.3% USE.  Both
were unfalsifiable after the fact because nothing tested that the mechanism could
fire, and nothing counted use.  So: every test below either makes the trigger
FIRE or makes it REFUSE, and the refusals are as load-bearing as the firings.

The methods under test are extracted FROM THE PATCHED VEHICLE SOURCE, not
re-typed here -- if the patch and this smoke ever disagree, the smoke is wrong
by construction rather than silently testing a copy.

Run:  uv run python duck_eval/p2/p2_trigger_smoke.py
"""
from __future__ import annotations

import ast
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VEHICLE = (
    REPO
    / "runs"
    / "harness_diff_0813"
    / "ds"
    / "jakobbrggen_taaf-kaggle-source-anim-20260807-anim"
    / "src"
    / "ARC3-Inference"
)

sys.path.insert(0, str(REPO / "duck_eval" / "p2"))
from p2_patch import (  # noqa: E402
    EPISODE_ACTION_CAP,
    H_STUCK_TURNS,
    K_EPISODES,
    RETRY_DISABLED_AT_LEVELS,
    apply_patch,
)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(("  PASS  " if ok else "  FAIL  ") + name + (("  -- " + detail) if detail else ""))


def _extract_p2_methods(agent_src: str):
    """Pull the _p2_* methods out of the PATCHED tool_agent.py and bind them to
    a bare stub class. Tests the shipped code, not a transcription of it."""
    tree = ast.parse(agent_src)
    wanted = {
        "_p2_game_key",
        "_p2_flush",
        "_p2_state",
        "_p2_note_acting_turn",
        "_p2_is_armed",
        "_p2_retry_armed",
        "_p2_count_attempt_calls",
        "_p2_report",
    }
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            found[node.name] = node
    missing = wanted - set(found)
    if missing:
        raise AssertionError("patched source is missing methods: %s" % sorted(missing))

    module = ast.Module(body=[found[n] for n in sorted(found)], type_ignores=[])
    ast.fix_missing_locations(module)
    # The methods reference module-level names from tool_agent.py. Supply the
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
    exec(compile(module, "<p2-methods>", "exec"), ns)  # noqa: S102

    class Agent:
        pass

    for n in wanted:
        setattr(Agent, n, ns[n])
    return Agent, found


def turn(agent, level, cleared_now=False):
    """One acting turn on `level`; cleared_now = that turn cleared the level."""
    agent._p2_note_acting_turn({"level": level, "level_transition": cleared_now})


def main() -> int:
    print("P2 TRIGGER-LEG SMOKE (anchors 3+4)")
    print("=" * 70)

    if not VEHICLE.is_dir():
        print("FATAL: vehicle bundle not found at %s" % VEHICLE)
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="p2trig-"))
    try:
        root = tmp / "ARC3-Inference"
        shutil.copytree(VEHICLE, root)
        info = apply_patch(root)
        agent_path = root / "inference" / "agent" / "tool_agent.py"
        src = agent_path.read_text(encoding="utf-8")

        print("\n-- T1: the patch applies to the real vehicle and compiles --")
        check("T1a apply_patch reports 6 anchors", info["anchors_applied"] == 6, str(info["anchors_applied"]))
        check("T1b sandbox is the VEHICLE generation", bool(info["sandbox_is_vehicle_generation"]))
        try:
            ast.parse(src)
            check("T1c patched tool_agent.py compiles", True)
        except SyntaxError as exc:
            check("T1c patched tool_agent.py compiles", False, str(exc))

        Agent, _ = _extract_p2_methods(src)

        print("\n-- T2: the trigger FIRES at exactly H, not before --")
        a = Agent()
        for i in range(H_STUCK_TURNS - 1):
            turn(a, 1)
        check("T2a silent at H-1 (%d turns)" % (H_STUCK_TURNS - 1), a._p2_is_armed() is False)
        turn(a, 1)
        check("T2b ARMS at exactly H (%d turns)" % H_STUCK_TURNS, a._p2_is_armed() is True)
        turn(a, 1)
        check("T2c stays armed past H", a._p2_is_armed() is True)

        print("\n-- T3: NEGATIVE CONTROL -- a level clear DISARMS it --")
        a = Agent()
        for _ in range(H_STUCK_TURNS + 3):
            turn(a, 1)
        check("T3a armed while stuck", a._p2_is_armed() is True)
        turn(a, 1, cleared_now=True)
        check("T3b DISARMED the instant the level clears", a._p2_is_armed() is False)
        check("T3c counter reset to 0", a._p2_state()["run"] == 0, str(a._p2_state()["run"]))

        print("\n-- T4: NEGATIVE CONTROL -- a prompt clearer NEVER arms --")
        a = Agent()
        for lvl in range(1, 21):  # clears a level every single turn, 20 turns
            turn(a, lvl, cleared_now=True)
        check("T4a never armed across 20 clearing turns", a._p2_is_armed() is False)
        check("T4b max_run stayed 0", a._p2_state()["max_run"] == 0, str(a._p2_state()["max_run"]))
        a = Agent()
        for _ in range(6):  # 3-stuck-then-clear, repeated -- never reaches 4
            turn(a, 1)
            turn(a, 1)
            turn(a, 1)
            turn(a, 1, cleared_now=True)
        check("T4c 3-stuck-then-clear never arms", a._p2_is_armed() is False)
        check("T4d max_run capped at H-1", a._p2_state()["max_run"] == H_STUCK_TURNS - 1,
              str(a._p2_state()["max_run"]))

        print("\n-- T5: NEGATIVE CONTROL -- retry DISABLES once k >= %d --" % RETRY_DISABLED_AT_LEVELS)
        a = Agent()
        for lvl in range(1, RETRY_DISABLED_AT_LEVELS + 1):
            turn(a, lvl, cleared_now=True)
        check("T5a cleared counter == %d" % RETRY_DISABLED_AT_LEVELS,
              a._p2_state()["cleared"] == RETRY_DISABLED_AT_LEVELS, str(a._p2_state()["cleared"]))
        for _ in range(20):  # now get very stuck
            turn(a, 99)
        check("T5b REFUSES to arm after k>=%d despite 20 stuck turns" % RETRY_DISABLED_AT_LEVELS,
              a._p2_is_armed() is False)
        check("T5c ... and the run counter still tracked it", a._p2_state()["run"] == 20,
              str(a._p2_state()["run"]))

        print("\n-- T6: a level CHANGE without a clear restarts the run at 1 --")
        a = Agent()
        for _ in range(3):
            turn(a, 1)
        turn(a, 2)  # level moved, but no level_transition reported
        check("T6a run restarted at 1 on a level change", a._p2_state()["run"] == 1,
              str(a._p2_state()["run"]))
        check("T6b therefore not armed", a._p2_is_armed() is False)

        print("\n-- T7: D2 instrument counts REAL attempt() calls (AST, not substring) --")
        a = Agent()
        for _ in range(H_STUCK_TURNS):
            turn(a, 1)
        a._p2_count_attempt_calls("attempt([{'action':'ACTION1'}])")
        st = a._p2_state()
        check("T7a counts a real call while armed", st["attempt_calls_armed"] == 1, str(st["attempt_calls_armed"]))
        check("T7b counts the TURN once", st["turns_calling_attempt_armed"] == 1)
        a._p2_count_attempt_calls("# attempt([1]) in a comment\nx = 'attempt([2])'\nprint(x)")
        st = a._p2_state()
        check("T7c REFUSES a comment and a string literal", st["attempt_calls_armed"] == 1,
              "got %d" % st["attempt_calls_armed"])
        a._p2_count_attempt_calls("attempt([1]); attempt([2]); attempt([3])")
        st = a._p2_state()
        check("T7d counts 3 calls in one turn", st["attempt_calls_armed"] == 4, str(st["attempt_calls_armed"]))
        check("T7e ... as ONE calling turn", st["turns_calling_attempt_armed"] == 2,
              str(st["turns_calling_attempt_armed"]))
        a._p2_count_attempt_calls("this is not python (((")
        check("T7f survives unparseable code", True)
        a._p2_count_attempt_calls("action([{'action':'ACTION1'}])")
        st = a._p2_state()
        check("T7g does NOT count action() as attempt()", st["attempt_calls_armed"] == 4,
              str(st["attempt_calls_armed"]))

        print("\n-- T8: D2 splits armed vs unarmed (the whole point of the gate) --")
        a = Agent()
        a._p2_count_attempt_calls("attempt([1])")  # run=0 -> unarmed
        st = a._p2_state()
        check("T8a a call while UNARMED is booked separately", st["attempt_calls_unarmed"] == 1
              and st["attempt_calls_armed"] == 0, str(st))
        for _ in range(H_STUCK_TURNS):
            turn(a, 1)
        a._p2_count_attempt_calls("attempt([1])")
        st = a._p2_state()
        check("T8b a call while ARMED is booked as armed", st["attempt_calls_armed"] == 1, str(st))

        print("\n-- T9: the D2 use RATE is computable (P1 died because it was not) --")
        a = Agent()
        for _ in range(H_STUCK_TURNS):
            turn(a, 1)
        for i in range(10):
            a._p2_retry_armed()  # 10 armed emissions
            if i < 3:
                a._p2_count_attempt_calls("attempt([1])")
        rep = a._p2_report()
        check("T9a armed_turns counted", rep["armed_turns"] == 10, str(rep["armed_turns"]))
        check("T9b d2_use_rate == 3/10", abs(rep["d2_use_rate"] - 0.3) < 1e-9, str(rep["d2_use_rate"]))
        check("T9c report carries the sealed constants",
              rep["H"] == H_STUCK_TURNS and rep["K"] == K_EPISODES and rep["cap"] == EPISODE_ACTION_CAP)
        a2 = Agent()
        check("T9d use_rate is None (not 0.0) when never armed", a2._p2_report()["d2_use_rate"] is None)

        print("\n-- T10: the affordance is announced WHERE THE MODEL READS --")
        check("T10a attempt() named in the tool description", "`attempt(actions)`" in src)
        check("T10b the description explains RESET-to-level-start",
              "CURRENT LEVEL START" in src and "RESETs back" in src)
        check("T10c the description states the RESET-after-WIN exception",
              "never" in src.lower() and "level_completed" in src)
        check("T10d the episode cap is stated", "capped at %d actions" % EPISODE_ACTION_CAP in src)
        check("T10e retry_mode is explained in the description", "`retry_mode: on`" in src)
        check("T10f the tool result actually emits retry_mode",
              '"retry_mode"] = "on"' in src and '"episodes_available"] = %d' % K_EPISODES in src)

        print("")
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

        print("\n-- T11: DRIFT negative control -- a drifted vehicle must die LOUDLY --")
        drift = tmp / "drift"
        shutil.copytree(VEHICLE, drift / "ARC3-Inference")
        ta = drift / "ARC3-Inference" / "inference" / "agent" / "tool_agent.py"
        ta.write_text(
            ta.read_text(encoding="utf-8").replace(
                '        code = str(arguments.get("code", "")).rstrip()\n', "", 1
            ),
            encoding="utf-8",
        )
        try:
            apply_patch(drift / "ARC3-Inference")
            check("T11a drifted vehicle dies loudly", False, "apply_patch SUCCEEDED on drift")
        except Exception as exc:  # noqa: BLE001
            check("T11a drifted vehicle dies loudly", type(exc).__name__ == "P2FatalDrift", type(exc).__name__)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 70)
    print("PASS %d  FAIL %d" % (len(PASS), len(FAIL)))
    for f in FAIL:
        print("  FAILED: " + f)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
