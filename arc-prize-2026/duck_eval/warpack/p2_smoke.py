"""P2 batch-gate smoke -- structure, legality, units, flag gate, and the REAL
offline arcengine integration, including the LC-PRESERVATION test.

``feedback_test_before_submit``: v38 scored 0.00 from a missing import. Every
module in this family is runtime-tested against the real engine before it can
be considered shippable -- including one that is NOT being pushed, because the
whole point of leaving it in ``_kaggle_dataset/`` is that a later slot can pick
it up without re-deriving anything.

The load-bearing test here is **L-C**: the recorded level-completing batches
(ar25 L1 = 5 stale LEFTs then 10 productive DOWNs, the last of which completes;
sp80 L1 = 3 stale RIGHTs then SPACE) must pass through the shipped rule
COMPLETELY. That is the property ``p2_replay.py`` proves over 4 recorded runs;
this asserts it directly on the shipped predicate.

Usage:
  .venv/Scripts/python.exe duck_eval/warpack/p2_smoke.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLE = REPO / "duck_eval" / "taaf_bundle" / "src"
WARKIT = REPO / "duck_eval" / "warpack" / "_kaggle_dataset"
ENV_FILES = REPO / "kaggle-data" / "environment_files"

sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "p2-smoke")
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:9/v1")

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}" + (f"  [{detail[:130]}]" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail[:400]}")


class _F:
    """Frame stand-in: the detectors only ever test grid/level equality."""
    __slots__ = ("grid", "level", "shape", "step")

    def __init__(self, grid, level=1, shape=(64, 64)):
        self.grid, self.level, self.shape = grid, level, shape
        self.step = 0


# --------------------------------------------------------------------------- #
# S: structure + legality of the shipped source
# --------------------------------------------------------------------------- #
def structural() -> None:
    print("\nS: structure + legality")
    src_path = WARKIT / "p2_batchgate_patch.py"
    check("S1 the module exists in the kaggle dataset dir", src_path.is_file())
    code = src_path.read_text(encoding="utf-8")
    raw = src_path.read_bytes()
    check("S2 the module is ASCII-only (the dataset upload path is cp1252)",
          all(b < 128 for b in raw),
          f"{len(raw)} B, {sum(1 for b in raw if b >= 128)} non-ascii")
    check("S3 the detectors are IMPORTED from effnote_patch, not re-implemented",
          "import effnote_patch as _EN" in code
          and "def detect_net_zero_cycle" not in code
          and "def detect_stagnation" not in code
          and "def count_recent_revisits" not in code)
    check("S4 arm flag + kill switch + vanilla fallback are all present",
          'os.environ.get("P2_BATCHGATE"' in code
          and 'os.environ.get("P2_BATCHGATE_DISABLE"' in code
          and "stock duck harness (vanilla)" in code)
    # Legality is a property of the CODE, not of the prose. Strip docstrings
    # and comments before asserting, or the module's own explanation of what it
    # refuses to do would fail its own check.
    tree = __import__("ast").parse(code)
    ast_mod = __import__("ast")
    doc_nodes = set()
    for node in ast_mod.walk(tree):
        if isinstance(node, (ast_mod.Module, ast_mod.FunctionDef,
                             ast_mod.AsyncFunctionDef, ast_mod.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast_mod.Expr) \
                    and isinstance(body[0].value, ast_mod.Constant) \
                    and isinstance(body[0].value.value, str):
                doc_nodes.add(id(body[0].value))
    strings = [n.value for n in ast_mod.walk(tree)
               if isinstance(n, ast_mod.Constant) and isinstance(n.value, str)
               and id(n) not in doc_nodes]
    names = {n.attr for n in ast_mod.walk(tree) if isinstance(n, ast_mod.Attribute)}
    names |= {n.id for n in ast_mod.walk(tree) if isinstance(n, ast_mod.Name)}
    blob = " ".join(strings) + " " + " ".join(sorted(names))
    check("S5 NO game id and NO baseline is read anywhere in the CODE "
          "(docstrings/comments stripped before asserting)",
          "base_actions_per_level" not in blob
          and "metadata.json" not in blob
          and "rglob" not in blob
          and not re.search(r"\b(ar25|sp80|tu93|m0r0|ft09|re86)\b", blob),
          blob[:120])
    check("S6 no duplicate-game replay gate (caoyupeng's external_game_id)",
          "external_game_id" not in code and "-dup" not in code)
    check("S7 the raw POSITION cap ships OFF (the replay proves it cuts "
          "level-completing actions)",
          '_env_int("P2_CAP", 0)' in code)
    check("S8 py_compile is clean",
          subprocess.run([sys.executable, "-m", "py_compile", str(src_path)],
                         capture_output=True).returncode == 0)
    check("S9 zero LLM / network calls in the module",
          not any(t in code for t in ("requests.", "httpx", "openai",
                                      "urlopen", "socket.")))


# --------------------------------------------------------------------------- #
# U: units on the pure functions
# --------------------------------------------------------------------------- #
def units(mod) -> None:
    print("\nU: pure predicate + gate")
    A = mod.should_abort_remainder

    check("U1 the first action of a batch is NEVER cut (executed=0 => None), "
          "so at least one action always executes and the stock aggregation "
          "path is always the one that runs",
          A(executed=0, consecutive_noops=99, consecutive_stale=99,
            cap=1, noop_run=1, stale_run=1) is None)
    check("U2 stale_run fires exactly at k",
          A(executed=3, consecutive_noops=0, consecutive_stale=5,
            stale_run=6, cap=0, noop_run=0) is None
          and A(executed=3, consecutive_noops=0, consecutive_stale=6,
                stale_run=6, cap=0, noop_run=0) == "stale_run>=6")
    check("U3 noop_run fires exactly at k",
          A(executed=3, consecutive_noops=1, consecutive_stale=0,
            noop_run=2, cap=0, stale_run=0) is None
          and A(executed=3, consecutive_noops=2, consecutive_stale=0,
                noop_run=2, cap=0, stale_run=0) == "noop_run>=2")
    check("U4 every limit at 0 disables the cap entirely",
          A(executed=99, consecutive_noops=99, consecutive_stale=99,
            cap=0, noop_run=0, stale_run=0) is None)
    check("U5 the position cap fires on position alone (ablation handle)",
          A(executed=2, consecutive_noops=0, consecutive_stale=0,
            cap=2, noop_run=0, stale_run=0) == "cap>=2")

    # ---- THE LC-PRESERVATION TEST -------------------------------------- #
    print("\nL-C: the recorded level-completing batches survive the shipped rule")

    def run_batch(pattern, *, stale_run, noop_run=0, cap=0, carry_stale=0):
        """pattern: list of 'new' | 'stale' | 'noop'. Returns index of the
        first cut action, or None if the whole batch executes."""
        executed = 0
        stale = carry_stale
        noop = 0
        for i, kind in enumerate(pattern):
            if A(executed=executed, consecutive_noops=noop,
                 consecutive_stale=stale, cap=cap, noop_run=noop_run,
                 stale_run=stale_run) is not None:
                return i
            executed += 1
            if kind == "new":
                stale = 0
                noop = 0
            elif kind == "stale":
                stale += 1
                noop = 0
            else:
                stale += 1
                noop += 1
        return None

    # ar25 L1: LEFT x5 all land on already-visited states, then DOWN x10, the
    # last of which COMPLETES. 15 actions, lc at position 15.
    ar25 = ["stale"] * 5 + ["new"] * 10
    # tu93 L1: 16 actions, lc at position 16, opening with re-traversal.
    tu93 = ["stale"] * 4 + ["new"] * 12
    # sp80 L1 (animation_v1): RIGHT x3 stale then SPACE completes.
    sp80 = ["stale"] * 3 + ["new"]
    # sp80 L1 (effnote_v1): RIGHT x3, UP x2, SPACE completes.
    sp80b = ["stale"] * 3 + ["new"] * 3

    shipped = {"stale_run": mod.CFG.stale_run, "noop_run": mod.CFG.noop_run,
               "cap": mod.CFG.cap}
    for name, pat in (("ar25 L1 (15)", ar25), ("tu93 L1 (16)", tu93),
                      ("sp80 L1 (4)", sp80), ("sp80 L1 effnote (6)", sp80b)):
        cut = run_batch(pat, **shipped)
        check(f"LC {name} executes COMPLETELY under the shipped rule "
              f"(stale_run={shipped['stale_run']}, noop_run={shipped['noop_run']}, "
              f"cap={shipped['cap']})",
              cut is None, f"cut at index {cut}")

    check("LC-neg a POSITION cap of 2 DOES destroy ar25 L1 -- the reason the "
          "brief's hard cap is refused, asserted rather than argued",
          run_batch(ar25, stale_run=0, noop_run=0, cap=2) == 2)
    check("LC-neg2 abort-on-first-revisit (stale_run=1) also destroys ar25 L1 "
          "-- the same wall that forced P1_ABORT_REVISIT=0 on 08-12",
          run_batch(ar25, stale_run=1, noop_run=0, cap=0) == 1)
    check("LC-pos a genuinely stuck batch IS cut: 12 consecutive stale actions "
          "with the shipped rule",
          run_batch(["stale"] * 12, **shipped) is not None)

    print("\nG: the gate (shipped EFFNOTE detectors at P2 thresholds)")
    g0 = [_F(f"g{i}") for i in range(20)]
    check("G1 a history with no repeats does NOT arm the gate",
          mod.detectors_for(g0[-1], g0) == {})
    flat = [_F("same") for _ in range(20)]
    fired = mod.detectors_for(flat[-1], flat)
    check("G2 a byte-identical run arms stagnation + revisit",
          "stag" in fired and "rev" in fired, json.dumps(fired))
    cyc = [_F("a"), _F("b"), _F("c"), _F("d"), _F("e"), _F("f"), _F("g"), _F("a")]
    fired2 = mod.detectors_for(cyc[-1], cyc)
    check("G3 a >=6-action round trip arms net-zero", "nz" in fired2,
          json.dumps(fired2))
    lvl = [_F("x", level=1)] * 3 + [_F("y", level=2)]
    check("G4 the scan stops at a LEVEL boundary (no cross-level stall)",
          mod.detectors_for(lvl[-1], lvl) == {})
    check("G5 the fingerprint is stable and order-sensitive",
          mod.board_fingerprint([[1, 2], [3, 4]])
          == mod.board_fingerprint([[1, 2], [3, 4]])
          and mod.board_fingerprint([[1, 2], [3, 4]])
          != mod.board_fingerprint([[3, 4], [1, 2]]))


# --------------------------------------------------------------------------- #
# K: flag gate + kill switch, in a clean subprocess
# --------------------------------------------------------------------------- #
def flag_gate() -> None:
    print("\nK: flag gate + kill switch (subprocess, clean interpreter)")
    prog = (
        "import sys;"
        f"sys.path[:0]=[r'{BUNDLE / 'ARC3-Inference'}',"
        f"r'{BUNDLE / 'tufa-arc-agi-framework' / 'src'}',r'{WARKIT}'];"
        "import inference.framework.solver as s;"
        "v=(s._HarnessGameSession.play,s._HarnessGameSession.step_env,"
        "s._HarnessGameSession._execute_action);"
        "import p2_batchgate_patch as p;"
        "r=p.apply(None);"
        "n=(s._HarnessGameSession.play,s._HarnessGameSession.step_env,"
        "s._HarnessGameSession._execute_action);"
        "print('APPLIED',r,'UNCHANGED',v==n)"
    )
    for name, env, want in (
        ("K1 flag OFF -> nothing patched", {}, "APPLIED False UNCHANGED True"),
        ("K2 kill switch beats the arm flag",
         {"P2_BATCHGATE": "1", "P2_BATCHGATE_DISABLE": "1"},
         "APPLIED False UNCHANGED True"),
        ("K3 arm flag ON -> the three seams are patched",
         {"P2_BATCHGATE": "1"}, "APPLIED True UNCHANGED False"),
    ):
        e = dict(os.environ)
        e.pop("P2_BATCHGATE", None)
        e.pop("P2_BATCHGATE_DISABLE", None)
        e.update(env)
        r = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                           text=True, env=e, cwd=str(REPO))
        out = (r.stdout or "") + (r.stderr or "")
        check(name, want in out, out.strip()[-260:])


# --------------------------------------------------------------------------- #
# I: the REAL offline arcengine
# --------------------------------------------------------------------------- #
class _StubSolver:
    max_runtime_s_per_game = None
    max_actions_per_game = None
    job_dir = None          # write_viewer_payload short-circuits on None

    def soft_time_remaining_seconds(self):
        return None


def integration() -> None:
    print("\nI: REAL offline arcengine integration")
    tmp_root = Path(tempfile.mkdtemp(prefix="p2smoke-"))
    run_dir = tmp_root / "run"
    run_dir.mkdir(parents=True)
    for f in ("p2_batchgate_patch.py", "effnote_patch.py"):
        shutil.copy(WARKIT / f, run_dir / f)

    cwd0 = Path.cwd()
    os.chdir(run_dir)
    sys.path.insert(0, str(run_dir))
    try:
        import arcengine
        import taaf.game as taaf_game
        import taaf.game_api as game_api
        import inference.framework.solver as solver_mod
        from types import SimpleNamespace

        vanilla_step = solver_mod._HarnessGameSession.step_env
        vanilla_exec = solver_mod._HarnessGameSession._execute_action

        os.environ["P2_BATCHGATE"] = "1"
        import p2_batchgate_patch as mod
        bm = SimpleNamespace(label="p2-smoke")
        applied = mod.apply(bm)
        check("I1 apply() returns True and stamps bm.label",
              applied and "-p2-v1" in bm.label, bm.label)
        check("I2 the action-path seams really were replaced",
              solver_mod._HarnessGameSession.step_env is not vanilla_step
              and solver_mod._HarnessGameSession._execute_action is not vanilla_exec)

        spec = game_api.ArcadeSpec(environments_dir=str(ENV_FILES))
        game = game_api.GameAPI(env_name="ft09", arcade_spec=spec)
        sess = taaf_game.RunSession()
        game.start_game(sess)

        session = object.__new__(solver_mod._HarnessGameSession)
        session.solver = _StubSolver()
        session.game = game
        session.analyzer = SimpleNamespace()
        session.game_index = 0
        session.pass_index = 0
        session.state_path = run_dir / "state.json"
        session.transcript_path = run_dir / "t.txt"
        session.analysis_html_relpath = "a.html"
        session.stop_event = __import__("threading").Event()
        session.viewer_data_path = run_dir / "viewer.json"
        session.started_at = __import__("time").monotonic()
        session.history_entries = []
        session.viewer_events = []
        session.analysis_step = 0
        session.last_engine_action = None
        session.token_baseline = 0
        session._viewer_events_flushed = 0
        session.seed_initial_history()

        valid = solver_mod._engine_action_names(game)
        model_actions = solver_mod.to_model_actions(valid)

        def act(name, row=0, col=0):
            """A well-formed action dict. MOUSE (ACTION6) is the only action
            ft09 exposes from its opening state and it REQUIRES row/col --
            omitting them makes ``_normalize_actions`` reject the whole batch
            (and is, incidentally, the same class of defect as the 2026-05
            ``sel.set_data`` bug)."""
            d = {"action": name}
            if name == "MOUSE":
                d["row"], d["col"] = row, col
            return d

        move = model_actions[0]

        print("  I3: an UNGATED batch passes through COMPLETELY untouched")
        p = session.step_env(
            {"actions": [act(move, 10 + i, 10 + i) for i in range(4)]})
        st = session._p2_state
        check("I3 4 requested -> 4 executed, no gate, no abort, no p2 fields",
              p.get("executed_count") == 4 and st.saved == 0
              and st.aborts == 0 and "p2_saved" not in p,
              json.dumps({k: p.get(k) for k in
                          ("executed_count", "requested_count", "stop_reason")}))
        check("I3b the batch was NOT gated (no stall in a fresh history)",
              st.gated_batches == 0 and st.batch_gated is False)

        print("  I4: a GATED, genuinely stuck batch IS truncated")
        # Drive the board into a byte-identical run so the shipped detectors
        # arm on the REAL engine's own frames, then issue a long batch.
        stuck = None
        for cand in model_actions:
            probe = session.step_env({"actions": [act(cand, 0, 0)]})
            if not probe.get("board_changed"):
                stuck = cand
                break
        if stuck is None:
            check("I4 SKIPPED - the engine exposes no inert action from this "
                  "state", True, "no no-op action available")
        else:
            for _ in range(12):
                session.step_env({"actions": [act(stuck, 0, 0)]})
            before = st.saved
            p2 = session.step_env(
                {"actions": [act(stuck, 0, 0) for _ in range(10)]})
            check("I4 the gate armed on the REAL engine frames",
                  st.batch_gated is True, mod.gate_reason(st.batch_fired))
            check("I4b the batch was truncated and the remainder was NOT charged",
                  st.saved > before and p2.get("executed_count", 99) < 10,
                  json.dumps({k: p2.get(k) for k in
                              ("executed_count", "requested_count",
                               "stop_reason", "p2_saved")}))
            check("I4c the payload is a CLEAN early stop, not an error",
                  p2.get("stop_reason") == "p2_batch_gate"
                  and p2.get("stopped_early") is True
                  and "error" not in p2, json.dumps(list(p2)[:16]))
            check("I4d at least one action ALWAYS executes (no empty batch, so "
                  "the stock aggregation path is the one that ran)",
                  p2.get("executed_count", 0) >= 1)
            check("I4e the abort never escapes to the caller as an exception",
                  st.errors == 0, str(st.errors))

        print("  I5: canary")
        mod.CANARY[st.game] = {
            "batches": st.batches, "gated_batches": st.gated_batches,
            "gated_multi": st.gated_multi, "requested": st.requested,
            "executed": st.executed, "saved": st.saved, "aborts": st.aborts,
            "fire_nz": st.fire_nz, "fire_stag": st.fire_stag,
            "fire_rev": st.fire_rev, "errors": st.errors,
            "gated_actions": st.gated_actions,
            "ungated_actions": st.ungated_actions,
            "gated_requested": st.gated_requested,
            "ungated_requested": st.ungated_requested,
        }
        rep = mod.canary_report()
        check("I5 the canary carries the delivery + behavioural counters",
              {"saved_rate", "gate_rate", "G1_gated_batch_size"} <= set(rep),
              json.dumps({k: rep[k] for k in
                          ("batches", "gated_batches", "saved", "errors")}))
        check("I5b requested >= executed always (the cap only ever removes)",
              rep["requested"] >= rep["executed"],
              f"{rep['requested']} >= {rep['executed']}")
        check("I5c no token metric of any kind (the denominator that killed "
              "the animation arm)",
              not any("token" in k for k in rep))

        game.finish_game()
        sess.close()
        check("I6 arcengine really was the engine under test",
              hasattr(arcengine, "GameAction"))
    finally:
        os.chdir(cwd0)


def main() -> int:
    structural()
    sys.path.insert(0, str(WARKIT))
    import p2_batchgate_patch as mod
    units(mod)
    flag_gate()
    integration()
    print(f"\n{PASS} passed / {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
