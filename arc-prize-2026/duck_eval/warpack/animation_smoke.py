"""Animation-awareness smoke -- CPU only, no GPU, no LLM, no network.

Runtime-tests the ASSEMBLED notebook notebooks/duckanimation-eval/
arc3-duck-animation-eval.ipynb AND the patched code path against the REAL
offline arcengine (feedback_test_before_submit: v38 scored 0.00 from a missing
import -- always runtime-test the exact artifact).

  S*   structural: 17 cells, eval+animation-flag prefix on cell 2, heavy gates
       on cells 4/6/8/10, animation graft (NOT warpack/ledger/sentinel/
       compaction) in cell 12, (f) continuation default block still riding,
       post-run canary in cell 14, kernel-metadata byte-parity with war-eval.
  U*   unit: summarize_animation / merge_animations / animation_note on
       synthetic frames -- including the exact type-1 shape our audit found
       (settled == previous settled, intermediate differs).
  I*   integration: exec cell 2 + cell 12 (real notebook source) against the
       module copy the kernel loads at runtime, then drive the REAL patched
       solver._HarnessGameSession._execute_action over a REAL offline ft09
       (type-1) and tr87 (single-frame) engine, asserting the payload gains
       an animation summary exactly where the audit says it must and nowhere
       else. Also exercises the ToolAgent seams and the canary report.
  K*   kill switch + flag gate (subprocess): ANIMATION_DISABLE=1 and
       ANIMATION_AWARE unset both leave the harness byte-vanilla.

Run:  .venv/Scripts/python.exe duck_eval/warpack/animation_smoke.py [--warkit <dir>]
"""
from __future__ import annotations

import ast
import asyncio
import contextlib
import inspect
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
NB_PATH = REPO / "notebooks" / "duckanimation-eval" / "arc3-duck-animation-eval.ipynb"
META_PATH = REPO / "notebooks" / "duckanimation-eval" / "kernel-metadata.json"
BASE_META_PATH = REPO / "notebooks" / "duckwar-eval" / "kernel-metadata.json"
WARKIT_DEFAULT = REPO / "duck_eval" / "warpack" / "_kaggle_dataset"
ENV_FILES = REPO / "kaggle-data" / "environment_files"

sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "animation-smoke")
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}" + (f"  [{detail[:110]}]" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail[:400]}")


def cell_src(nb: dict, i: int) -> str:
    return "".join(nb["cells"][i]["source"])


def exec_cell(src: str, ns: dict) -> str:
    buf = io.StringIO()
    code = compile(src, "<cell>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    with contextlib.redirect_stdout(buf):
        if code.co_flags & inspect.CO_COROUTINE:
            asyncio.run(eval(code, ns))  # noqa: S307 - our own notebook source
        else:
            exec(code, ns)  # noqa: S102 - our own notebook source
    out = buf.getvalue()
    if out.strip():
        for line in out.splitlines():
            print(f"    | {line}")
    return out


# --------------------------------------------------------------------------- #
# S: structural
# --------------------------------------------------------------------------- #
def structural(nb: dict) -> None:
    print("S: structural checks on the assembled notebook")
    check("S1 17 cells", len(nb["cells"]) == 17, str(len(nb["cells"])))
    c2 = cell_src(nb, 2)
    check("S2 cell 2 forces the offline bench (eval line first)",
          c2.startswith('import os; os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"'))
    check("S2b cell 2 stamps ANIMATION_AWARE=1 + seed + banner",
          'os.environ["ANIMATION_AWARE"] = "1"' in c2
          and 'os.environ["ANIMATION_EVAL_SEED"] = "1"' in c2
          and "animation-eval: SEED=1" in c2)
    gated = all(cell_src(nb, i).lstrip("#").strip().startswith("Warpack fast-submit gate")
                and "if RUN_HEAVY:" in cell_src(nb, i) for i in (4, 6, 8, 10))
    check("S3 heavy cells 4/6/8/10 gated on RUN_HEAVY", gated)
    c12 = cell_src(nb, 12)
    check("S4 cell 12 is the animation graft",
          "import animation_patch" in c12 and "animation_patch.apply(bm)" in c12
          and "if not RUN_HEAVY:" in c12)
    check("S4b cell 12 ships NO warpack/ledger/sentinel/compaction",
          "warpack_patch" not in c12 and "ledger_patch" not in c12
          and "budget_sentinel_patch" not in c12 and "compaction_patch" not in c12)
    check("S4c cell 12 keeps the (f) continuation default block",
          "import continuation_patch" in c12)
    check("S4d NO no-op guard anywhere in the notebook (prereg sec2.2)",
          not any("noop_guard" in cell_src(nb, i) for i in range(len(nb["cells"]))))
    c14 = cell_src(nb, 14)
    check("S5 cell 14 keeps the fast-submit path", "_write_dummy_submission" in c14)
    check("S5b cell 14 carries the post-run animation canary",
          "animation_patch as _anim" in c14 and "_anim.canary_report()" in c14)
    check("S5c no non-ASCII in cells 2/12/14 (round-trip safe)",
          not any(ord(ch) > 127 for ch in c2 + c12 + c14))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_META_PATH.read_text(encoding="utf-8"))
    delta = {k for k in set(meta) | set(base) if meta.get(k) != base.get(k)}
    check("S6 kernel-metadata matches war-eval except id/title/code_file",
          delta == {"id", "title", "code_file"}
          and meta["id"] == "canivel/arc3-duck-animation-eval",
          f"delta={sorted(delta)}")


# --------------------------------------------------------------------------- #
# U: units
# --------------------------------------------------------------------------- #
def units(mod) -> None:
    print("U: unit checks on the pure summary functions")
    A = ((0, 0), (0, 0))
    B = ((0, 1), (0, 0))
    C = ((1, 1), (1, 1))

    check("U1 single frame -> None", mod.summarize_animation([A], A) is None)
    check("U2 all frames identical -> None",
          mod.summarize_animation([A, A, A], A) is None)

    # THE type-1 shape: settled == previous settled, an intermediate differs.
    s = mod.summarize_animation([B, A], A)
    check("U3 type-1 detected as reject_or_consumed",
          isinstance(s, dict) and s["signature"] == "reject_or_consumed"
          and s["board_unchanged"] is True and s["frames"] == 2
          and s["transient_cells"] == 1 and s["transient_bbox"] == [0, 1, 0, 1],
          json.dumps(s))

    # Motion: the board DID change vs the previous settled frame.
    s2 = mod.summarize_animation([B, C], A)
    check("U4 board-changed animation is 'motion', not reject",
          isinstance(s2, dict) and s2["signature"] == "motion"
          and s2["board_unchanged"] is False, json.dumps(s2))

    check("U5 no raw frames anywhere in the summary schema",
          set(s) == {"frames", "unique_frames", "board_unchanged",
                     "transient_cells", "transient_bbox", "signature"},
          str(sorted(s)))
    check("U6 summary is token-bounded (<220 chars serialized)",
          len(json.dumps(s)) < 220, f"{len(json.dumps(s))} chars")

    merged = mod.merge_animations([s2, s, s2])
    check("U7 batch merge keeps the aliased case visible",
          merged["signature"] == "reject_or_consumed"
          and merged["animated_actions"] == 3 and merged["invisible_actions"] == 1,
          json.dumps(merged))
    check("U8 merge of empty -> None", mod.merge_animations([]) is None)

    note = mod.animation_note(s)
    check("U9 reject note tells the agent NOT to record 'no effect'",
          "NOT ignored" in note and "no effect" in note, note[:90])
    check("U10 no-summary note is empty", mod.animation_note(None) == "")

    # Robustness: ill-shaped input must never raise.
    ok = True
    for bad in ([], None, [A, ((0,), (0, 0))], [[[0, "x"]], [[0, 0]]]):
        try:
            mod.summarize_animation(bad, A)
        except Exception:  # noqa: BLE001
            ok = False
    check("U11 ill-shaped frame input never raises", ok)


# --------------------------------------------------------------------------- #
# I: integration against the REAL offline engine
# --------------------------------------------------------------------------- #
class _StubSession:
    """Minimal ``_HarnessGameSession`` stand-in: exactly the attributes the
    VANILLA ``_execute_action`` touches. The game is a REAL taaf ``GameAPI``
    on a REAL offline arcengine env, so the frames under test are real."""

    def __init__(self, game) -> None:
        self.game = game
        self.analyzer = SimpleNamespace(total_tokens=0)
        self.token_baseline = 0
        self.history_entries: list = []
        self.last_engine_action = None
        self.viewer_events: list = []

    @property
    def action_count(self) -> int:
        return len(self.history_entries)

    def write_runtime_state(self) -> None:
        return None

    def timing_payload(self) -> dict:
        return {"run_elapsed_seconds": 0.0, "time_remaining_seconds": None}

    def _append_action_viewer_event(self, payload, frame) -> None:
        self.viewer_events.append(payload)

    def write_viewer_payload(self) -> None:
        return None


def _make_game(env_name: str):
    import taaf.game as taaf_game
    import taaf.game_api as game_api

    spec = game_api.ArcadeSpec(environments_dir=str(ENV_FILES))
    game = game_api.GameAPI(env_name=env_name, arcade_spec=spec)
    session = taaf_game.RunSession()
    game.start_game(session)
    return game, session


def _drive(solver_mod, game, actions) -> list[dict]:
    import arcengine

    stub = _StubSession(game)
    payloads: list[dict] = []
    for name, data in actions:
        try:
            action = arcengine.ActionInput(
                id=arcengine.GameAction.from_name(name), data=dict(data))
            payload = solver_mod._HarnessGameSession._execute_action(
                stub, action, batch_index=1, batch_size=1, generated_tokens=0)
        except Exception:  # noqa: BLE001 - engine refusals are expected mid-script
            continue
        payloads.append(payload)
    return payloads


def integration(warkit: Path) -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    structural(nb)

    src = warkit / "animation_patch.py"
    assert src.is_file(), f"no animation_patch.py under {warkit}"
    print(f"module source under test: {src}")

    tmp_root = Path(tempfile.mkdtemp(prefix="animsmoke-"))
    run_dir = tmp_root / "run"
    run_dir.mkdir(parents=True)
    shutil.copy(src, run_dir / "animation_patch.py")
    shutil.copy(warkit / "continuation_patch.py", run_dir / "continuation_patch.py")

    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        ns: dict = {}
        print("I1: exec cell 2 (eval gate + arm flag)")
        out2 = exec_cell(cell_src(nb, 2), ns)
        check("I1 RUN_HEAVY forced True by the eval line",
              ns.get("RUN_HEAVY") is True and ns.get("FORCE_OFFLINE_BENCH") is True)
        check("I1b ANIMATION_AWARE=1 stamped + banner printed",
              os.environ.get("ANIMATION_AWARE") == "1"
              and "animation-eval: SEED=1" in out2)

        print("I2: exec cell 12 (animation graft) against the runtime module copy")
        ns["bm"] = SimpleNamespace(label="animation-smoke")
        out12 = exec_cell(cell_src(nb, 12), ns)
        check("I2 animation graft applied (banner printed)",
              "animation v1: ACTIVE" in out12 and "graft applied" in out12)
        check("I2b banner states NO no-op guard (prereg sec2.2)",
              "NO no-op guard" in out12)
        check("I2c bm.label carries the animation marker",
              "-animation-v1" in ns["bm"].label, ns["bm"].label)
        check("I2d NO fallback-to-vanilla traceback", "PATCH FAILED" not in out12)
        check("I2e graft did NOT import warpack/ledger/sentinel/compaction",
              not {"warpack_patch", "ledger_patch", "ledger_core",
                   "budget_sentinel_patch", "compaction_patch"} & set(sys.modules))
        check("I2f (f) continuation still applied alongside",
              "continuation v1" in out12)

        import animation_patch as mod
        import inference.framework.solver as solver_mod
        from inference.agent.tool_agent import ToolAgent

        units(mod)

        print("I3: REAL offline ft09 (audit: type-1, 99.3% of its no-ops invisible)")
        mod.COUNTERS.__init__()
        game, sess = _make_game("ft09")
        import random
        rng = random.Random("animation_smoke:ft09")
        script = [("ACTION6", {"x": rng.randrange(64), "y": rng.randrange(64)})
                  for _ in range(60)]
        payloads = _drive(solver_mod, game, script)
        game.finish_game()
        sess.close()
        anims = [p for p in payloads if isinstance(p.get("animation"), dict)]
        invis = [p for p in anims
                 if p["animation"]["signature"] == "reject_or_consumed"]
        check("I3 ft09 produced animation summaries",
              len(anims) > 0, f"{len(anims)}/{len(payloads)} actions")
        check("I3b ft09 produced INVISIBLE (reject_or_consumed) summaries -- "
              "the aliased class the audit measured",
              len(invis) > 0, f"{len(invis)}/{len(payloads)} actions")
        check("I3c every INVISIBLE payload reports board_changed False "
              "(this is exactly the false-no-op the agent used to see)",
              all(p.get("board_changed") is False for p in invis))
        check("I3d every INVISIBLE payload carries the corrective note",
              all("NOT ignored" in str(p.get("animation_note")) for p in invis))
        check("I3e counters agree with the payloads",
              mod.COUNTERS.invisible == len(invis)
              and mod.COUNTERS.multi == len([
                  p for p in payloads if isinstance(p.get("animation"), dict)]),
              f"counters multi={mod.COUNTERS.multi} invisible={mod.COUNTERS.invisible}")
        check("I3f zero exceptions in the patched action path (K-A4)",
              mod.COUNTERS.errors == 0, str(mod.COUNTERS.errors))
        check("I3g NO raw frames leaked into any payload",
              all(set(p["animation"]) == {
                  "frames", "unique_frames", "board_unchanged",
                  "transient_cells", "transient_bbox", "signature"} for p in anims))

        print("I4: REAL offline tr87 (audit: single-frame, 0 animations)")
        before = dict(mod.COUNTERS.by_game)
        game2, sess2 = _make_game("tr87")
        avail = [a for a in game2.current_state.available_actions if a != 0]
        import arcengine
        by_value = {a.value: a for a in arcengine.GameAction}
        rng2 = random.Random("animation_smoke:tr87")
        script2 = []
        for _ in range(40):
            act = by_value[rng2.choice(avail)]
            script2.append((act.name,
                            {"x": rng2.randrange(64), "y": rng2.randrange(64)}
                            if act.name == "ACTION6" else {}))
        payloads2 = _drive(solver_mod, game2, script2)
        game2.finish_game()
        sess2.close()
        anims2 = [p for p in payloads2 if isinstance(p.get("animation"), dict)]
        check("I4 tr87 emits ZERO summaries -> zero token cost on single-frame "
              "games (matches the audit)",
              len(anims2) == 0 and len(payloads2) > 0,
              f"{len(anims2)}/{len(payloads2)}")
        check("I4b tr87 payloads are otherwise byte-vanilla in shape",
              all("animation_note" not in p for p in payloads2))
        _ = before

        print("I5: ToolAgent seams")
        sample = invis[0]["animation"]
        compact = ToolAgent._compact_action_result(
            object.__new__(ToolAgent),
            {"executed": True, "board_changed": False, "animation": sample,
             "animation_note": mod.animation_note(sample)})
        check("I5 compactor carries the animation field to the model",
              compact.get("animation") == sample and "animation_note" in compact)
        summary = ToolAgent._summarize_step_sequence(
            object.__new__(ToolAgent),
            [{"executed": True, "action_num": 3, "board_changed": False,
              "animation": sample}])
        check("I5b step summary picks the animation up",
              isinstance(summary, dict) and summary.get("animation") is not None)
        text = ToolAgent._describe_last_outcome(object.__new__(ToolAgent), summary)
        check("I5c outcome text corrects the 'weak evidence' reading",
              "was NOT inert" in text and "no effect" in text, text[-120:])
        vanilla_text = ToolAgent._describe_last_outcome(
            object.__new__(ToolAgent),
            {"executed_count": 1, "board_changed": True, "level": 1})
        check("I5d outcome text untouched when the board DID change",
              "was NOT inert" not in vanilla_text, vanilla_text[:90])

        print("I6: canary report (prereg sec3)")
        rep = mod.canary_report(total_tokens=1_000_000)
        check("I6 canary reports nonzero invisible on an audit type-1 game (K-A2)",
              rep["invisible_actions"] > 0
              and any(g.startswith("ft09") for g in rep["audit_type1_games_engaged"]),
              json.dumps(rep["audit_type1_games_engaged"]))
        check("I6b canary reports zero errors (K-A4)", rep["errors"] == 0)
        check("I6c token fraction is far under the 1% bound (K-A3)",
              rep["animation_token_fraction"] < 0.01,
              str(rep["animation_token_fraction"]))
        check("I6d per-game sidecar/event accounting present",
              len(rep["games_with_events"]) >= 1, str(rep["games_with_events"]))
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# K: flag gate + kill switch (subprocess)
# --------------------------------------------------------------------------- #
def run_gate_child(warkit: Path) -> None:
    """No ANIMATION_AWARE (or ANIMATION_DISABLE=1) -> nothing is patched."""
    sys.path.insert(0, str(warkit))
    import animation_patch as mod
    import inference.framework.solver as solver_mod
    from inference.agent.tool_agent import ToolAgent

    before_exec = solver_mod._HarnessGameSession._execute_action
    before_step = solver_mod._HarnessGameSession.step_env
    before_compact = ToolAgent._compact_action_result
    bm = SimpleNamespace(label="gate-child")
    applied = mod.apply(bm)
    check("K1 apply() returns False", applied is False)
    check("K2 solver seams untouched (vanilla)",
          solver_mod._HarnessGameSession._execute_action is before_exec
          and solver_mod._HarnessGameSession.step_env is before_step)
    check("K3 ToolAgent seam untouched (vanilla)",
          ToolAgent._compact_action_result is before_compact)
    check("K4 bm.label unstamped", bm.label == "gate-child")


def main() -> int:
    global PASS, FAIL
    args = sys.argv[1:]
    warkit = WARKIT_DEFAULT
    if "--warkit" in args:
        warkit = Path(args[args.index("--warkit") + 1])
    warkit = warkit.resolve()
    assert (warkit / "animation_patch.py").is_file(), f"bad --warkit: {warkit}"

    if "--gate-child" in args:
        run_gate_child(warkit)
        print(f"\nGATE RESULT: {PASS} passed, {FAIL} failed")
        return 1 if FAIL else 0

    print(f"animation-awareness smoke | nb={NB_PATH}")
    integration(warkit)

    for label, env_delta in (
        ("flag OFF (ANIMATION_AWARE unset)", {"ANIMATION_AWARE": None}),
        ("kill switch (ANIMATION_DISABLE=1)",
         {"ANIMATION_AWARE": "1", "ANIMATION_DISABLE": "1"}),
    ):
        print(f"K: {label} (subprocess)")
        env = dict(os.environ)
        for k, v in env_delta.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        env.pop("WARPACK_FORCE_OFFLINE_BENCH", None)
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--gate-child",
             "--warkit", str(warkit)],
            capture_output=True, text=True, env=env, timeout=900)
        for line in child.stdout.splitlines():
            if line.strip().startswith(("PASS", "FAIL", "GATE")):
                print(f"  {line.strip()}")
        PASS += child.stdout.count("  PASS")
        FAIL += child.stdout.count("  FAIL")
        if child.returncode not in (0, 1):
            FAIL += 1
            print(child.stdout[-2000:])
            print(child.stderr[-2000:])

    print(f"\nRESULT: {PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
