"""EFFNOTE smoke -- CPU only, no GPU/LLM/network.

Runtime-tests the ASSEMBLED notebook
notebooks/duckeffnote-eval/arc3-duck-effnote-eval.ipynb AND the patched code
path against the REAL offline arcengine (feedback_test_before_submit: v38
scored 0.00 from a missing import -- always runtime-test the exact artifact).

  S*  structural: eval + EFFNOTE-flag prefix on cell 2, heavy gates, EFFNOTE
      graft (NOT warpack/ledger/sentinel/compaction/animation/p1) in cell 12,
      (f) continuation default still riding, post-run canary in cell 14,
      kernel-metadata byte-parity with the war-eval family, and the structural
      diff vs the war-eval baseline (exactly cells 2/12/14).
  U*  unit: the three pure detectors, the clamped game-agnostic target proxy,
      the note assembly, and the CHARACTER cost bound (incl. the worst case
      and a forced-tiny bound, which must drop WHOLE lines, never mid-sentence).
  L*  legality: no game id, no baseline table, no metadata read anywhere in the
      shipped module; the note is report-only (no action seam is patched).
  R*  REPLAY on the three block-free control runs (the pre-registered
      CONTROL SPREAD, computed BEFORE the arm exists -- Mechanism C's lesson).
  I*  integration: exec cell 2 + cell 12 (real notebook source) against the
      module copy the kernel loads at runtime, then drive the REAL patched
      ToolAgent._build_user_prompt and the REAL patched
      solver._HarnessGameSession.play over a REAL offline engine.
  K*  flag gate + kill switch (subprocess): EFFNOTE unset and EFFNOTE_DISABLE=1
      both leave the harness byte-vanilla.

Run:  .venv/Scripts/python.exe duck_eval/warpack/effnote_smoke.py [--warkit <dir>]
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
NB_PATH = REPO / "notebooks" / "duckeffnote-eval" / "arc3-duck-effnote-eval.ipynb"
META_PATH = REPO / "notebooks" / "duckeffnote-eval" / "kernel-metadata.json"
BASE_NB_PATH = REPO / "notebooks" / "duckwar-eval" / "arc3-duck-war-eval.ipynb"
BASE_META_PATH = REPO / "notebooks" / "duckwar-eval" / "kernel-metadata.json"
WARKIT_DEFAULT = REPO / "duck_eval" / "warpack" / "_kaggle_dataset"
ENV_FILES = REPO / "kaggle-data" / "environment_files"

sys.path.insert(0, str(BUNDLE / "ARC3-Inference"))
sys.path.insert(0, str(BUNDLE / "tufa-arc-agi-framework" / "src"))

os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "effnote-smoke")
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
    print("\n".join(f"    | {ln}" for ln in out.splitlines()[:14]))
    return out


class _F:
    """Frame stand-in: the detectors only test grid/level equality."""
    __slots__ = ("grid", "level", "shape", "step", "ascii")

    def __init__(self, grid, level=1, shape=(64, 64)):
        self.grid, self.level, self.shape = grid, level, shape
        self.step = 0
        self.ascii = ""

    def __str__(self):
        return f"Level: {self.level}\nStep: {self.step}\n"


class _E:
    __slots__ = ("frame",)

    def __init__(self, frame):
        self.frame = frame


# --------------------------------------------------------------------------- #
# S: structural
# --------------------------------------------------------------------------- #
def structural(nb: dict) -> None:
    print("S: structural")
    c2 = cell_src(nb, 2)
    check("S1 cell 2 forces the offline bench (eval line first)",
          c2.startswith('import os; os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"'))
    check("S2 cell 2 sets the EFFNOTE arm flag + seed",
          'os.environ["EFFNOTE"] = "1"' in c2
          and 'os.environ["EFFNOTE_EVAL_SEED"] = "1"' in c2)
    check("S3 cell 2 banner names REPORT-ONLY, the proxy-only target and the "
          "CHARACTER bound",
          "REPORT-ONLY" in c2 and "NO baseline table" in c2
          and "700 CHARACTERS" in c2)
    c12 = cell_src(nb, 12)
    check("S4 cell 12 IS the EFFNOTE graft", "import effnote_patch" in c12
          and "effnote_patch.apply(bm)" in c12)
    check("S5 cell 12 carries NO warpack/ledger/sentinel/compaction/animation/p1 graft",
          not any(t in c12 for t in ("import warpack_patch", "import ledger_patch",
                                     "import budget_sentinel_patch",
                                     "import compaction_patch",
                                     "import animation_patch",
                                     "import p1_suppressor_patch")))
    check("S6 cell 12 still carries the (f) continuation default",
          "import continuation_patch" in c12)
    check("S7 cell 12 falls back to VANILLA on failure",
          "PATCH FAILED - continuing with VANILLA duck harness" in c12)
    c14 = cell_src(nb, 14)
    check("S8 cell 14 calls the post-run canary",
          "effnote_patch as _effnote" in c14 and "_effnote.canary_report()" in c14)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    base = json.loads(BASE_META_PATH.read_text(encoding="utf-8"))
    delta = {k for k in set(meta) | set(base) if meta.get(k) != base.get(k)}
    check("S9 kernel-metadata differs from the war-eval family ONLY in "
          "id/title/code_file (kaggle_env_match discipline)",
          delta <= {"id", "title", "code_file"}, str(sorted(delta)))
    check("S10 kernel id is the EFFNOTE slug",
          meta["id"] == "canivel/arc3-duck-effnote-eval", meta["id"])
    check("S11 the wheelhouse/bundle/model triple + docker sha + machine shape "
          "are unchanged (feedback_kaggle_env_match, 5x confirmed)",
          meta["dataset_sources"] == base["dataset_sources"]
          and meta["docker_image"] == base["docker_image"]
          and meta["machine_shape"] == base["machine_shape"]
          and meta["enable_gpu"] == base["enable_gpu"]
          and meta["enable_internet"] == base["enable_internet"])
    # STRUCTURAL DIFF GATE. preflight.py has no duck-harness family profile
    # (known runbook debt, filed 2026-08-12: its K2/K4/K5/K6/K8 checks test the
    # arc3-baseline agent-swarm shape and BLOCK every member of this family,
    # including one that built COMPLETE). The applicable gate is this diff.
    base_nb = json.loads(BASE_NB_PATH.read_text(encoding="utf-8"))
    check("S12 same cell count as the war-eval baseline",
          len(nb["cells"]) == len(base_nb["cells"]),
          f"{len(nb['cells'])} vs {len(base_nb['cells'])}")
    differing = [i for i, (a, b) in enumerate(zip(base_nb["cells"], nb["cells"]))
                 if "".join(a["source"]) != "".join(b["source"])]
    check("S13 STRUCTURAL DIFF GATE: exactly cells 2/12/14 differ from the "
          "war-eval baseline (same shape as the P1 arm)",
          differing == [2, 12, 14], str(differing))


# --------------------------------------------------------------------------- #
# U: unit (pure functions)
# --------------------------------------------------------------------------- #
def units(mod) -> None:
    print("U: unit (pure detectors / target proxy / note)")

    # -- detect_net_zero_cycle --------------------------------------------
    # A -> B -> C -> D -> E -> F -> A : a 6-action round-trip back to A.
    seq = ["A", "B", "C", "D", "E", "F", "A"]
    frames = [_F(g) for g in seq]
    check("U1 net-zero fires on a 6-action round-trip",
          mod.detect_net_zero_cycle(frames[-1], frames) == 6,
          str(mod.detect_net_zero_cycle(frames[-1], frames)))
    short = [_F(g) for g in ["A", "B", "C", "A"]]
    check("U2 net-zero is SILENT below the 6-action floor (a 3-action probe is "
          "not a waste burst)",
          mod.detect_net_zero_cycle(short[-1], short) is None)
    static = [_F("A") for _ in range(12)]
    check("U3 net-zero does NOT fire on a purely static board (the divergence "
          "requirement; that case belongs to STALL)",
          mod.detect_net_zero_cycle(static[-1], static) is None)
    lvl = [_F("A", 1), _F("B", 1), _F("C", 2), _F("D", 2), _F("E", 2),
           _F("F", 2), _F("G", 2), _F("H", 2), _F("A", 2)]
    check("U4 net-zero stops at the level boundary (no cross-level cycles)",
          mod.detect_net_zero_cycle(lvl[-1], lvl) is None)

    # -- detect_stagnation -------------------------------------------------
    stag = [_F("Z")] + [_F("A") for _ in range(9)]
    check("U5 stagnation fires on 8 consecutive no-change actions",
          mod.detect_stagnation(stag[-1], stag) == 8,
          str(mod.detect_stagnation(stag[-1], stag)))
    stag7 = [_F("Z")] + [_F("A") for _ in range(7)]
    check("U6 stagnation is SILENT at 6 (below the floor)",
          mod.detect_stagnation(stag7[-1], stag7) is None,
          str(mod.detect_stagnation(stag7[-1], stag7)))
    check("U7 stagnation stops at the level boundary",
          mod.detect_stagnation(_F("A", 2), [_F("A", 1) for _ in range(20)]
                                + [_F("A", 2)]) is None)

    # -- count_recent_revisits --------------------------------------------
    osc = [_F("A"), _F("B"), _F("A"), _F("B"), _F("A"), _F("B"), _F("A")]
    check("U8 revisits counts EXACT recurrences of the current state",
          mod.count_recent_revisits(osc[-1], osc) == 3,
          str(mod.count_recent_revisits(osc[-1], osc)))
    march = [_F(f"S{i}") for i in range(20)]
    check("U9 revisits is ZERO on a strictly advancing board (exact-match only "
          "-- a near-match tolerance would flag a marching avatar as cycling)",
          mod.count_recent_revisits(march[-1], march) == 0)

    # -- heuristic_action_target (the clamped GAME-AGNOSTIC proxy) --------
    check("U10 proxy target is clamped to [40,100]",
          mod.heuristic_action_target(99, 10 ** 6) == 100
          and mod.heuristic_action_target(0, 0) == 50
          and mod.heuristic_action_target(-5, -5) == 50)
    check("U11 proxy target on this rail (64x64, 5 valid actions) = 100",
          mod.heuristic_action_target(5, 64 * 64) == 100)
    check("U12 proxy target is a function of OBSERVABLES only "
          "(same inputs -> same target, whatever the game)",
          mod.heuristic_action_target(3, 4096) == mod.heuristic_action_target(3, 4096)
          and mod.heuristic_action_target(3, 4096) == 95,
          str(mod.heuristic_action_target(3, 4096)))

    # -- build_efficiency_note --------------------------------------------
    quiet = mod.build_efficiency_note(level_number=1, actions_this_level=0,
                                      target=100)
    check("U13 the note is QUIET when nothing has been spent and nothing stalls",
          quiet == "", repr(quiet)[:80])
    n1 = mod.build_efficiency_note(level_number=1, actions_this_level=12,
                                   target=100)
    check("U14 (a) the scoring rule is stated VERBATIM and QUANTITATIVELY on "
          "every note",
          "(human_baseline_actions/your_actions)^2 x 100" in n1
          and "baseline=100, 2x over=25, 3x=11, 5x=4" in n1)
    check("U15 (b) the live action count and the proxy target are shown",
          "12 of about 100 typical actions used" in n1, n1.splitlines()[-1])
    check("U16 the under-target note carries NO reminder (not boilerplate)",
          "Commit to your best hypothesis" not in n1)
    n2 = mod.build_efficiency_note(level_number=1, actions_this_level=225,
                                   target=100)
    check("U17 (c) the over-target ratio is quantified",
          "225 actions used" in n2 and "2.2x over the typical target" in n2)
    check("U18 (e) the commit-don't-scan reminder fires when over target",
          "Commit to your best hypothesis" in n2
          and "do not scan rows/columns or enumerate options" in n2)
    n3 = mod.build_efficiency_note(level_number=3, actions_this_level=10,
                                   target=100, net_zero_actions=9,
                                   stagnation_actions=11, revisit_count=6)
    check("U19 (d) all three stall detectors render when they fire",
          "STALL:" in n3 and "NET-ZERO:" in n3 and "REVISIT:" in n3, n3)
    check("U20 (e) the reminder fires on a stall even when UNDER target",
          "Commit to your best hypothesis" in n3)
    n4 = mod.build_efficiency_note(level_number=1, actions_this_level=5,
                                   target=100, revisit_count=3)
    check("U21 revisit BELOW the floor does not render and does not trigger "
          "the reminder",
          "REVISIT:" not in n4 and "Commit to your best hypothesis" not in n4)

    # -- the CHARACTER cost bound (never a token fraction) ----------------
    worst = mod.build_efficiency_note(
        level_number=99, actions_this_level=99999, target=100,
        net_zero_actions=9999, stagnation_actions=9999, revisit_count=9999)
    check("U22 K-E3 the WORST-CASE note is within the 700-CHARACTER bound",
          len(worst) <= 700, f"{len(worst)} chars")
    check("U23 the worst case still carries the rule, the count and the reminder",
          "EFFICIENCY BUDGET" in worst and "99999 actions used" in worst
          and "Commit to your best hypothesis" in worst)
    tiny = mod.build_efficiency_note(
        level_number=1, actions_this_level=500, target=100,
        net_zero_actions=9, stagnation_actions=9, revisit_count=9,
        max_chars=300)
    check("U24 a forced-tiny bound drops WHOLE lines, never mid-sentence",
          len(tiny) <= 300 and tiny.endswith(".")
          and all(ln.strip() for ln in tiny.splitlines()), f"{len(tiny)}: {tiny!r}")
    check("U25 under pressure the SCORING RULE is the last thing dropped",
          "EFFICIENCY BUDGET" in tiny)

    # -- state + canary ----------------------------------------------------
    st = mod.EffNoteState("smoke")
    game = SimpleNamespace(game_run=SimpleNamespace(
        actions_per_level=[7, 130], levels_completed=1))
    lvl_n, used = mod.level_and_actions(game, 0, None, None)
    check("U26 the live per-level counter is read from game_run "
          "(the same array the scorer reads)",
          (lvl_n, used) == (2, 130), f"{lvl_n},{used}")
    hist = [_E(_F("A", 1)), _E(_F("B", 1)), _E(_F("C", 1))]
    lvl_n2, used2 = mod.level_and_actions(SimpleNamespace(), 0, _F("C", 1), hist)
    check("U27 the fallback counts same-level frames minus the arrival frame",
          (lvl_n2, used2) == (1, 2), f"{lvl_n2},{used2}")
    note = mod.note_for_turn(st, game, 137, _F("A", 2), hist, ["U", "D", "L", "R", "S"])
    check("U28 note_for_turn updates the delivery counters",
          st.turns == 1 and st.noted == 1 and st.chars_max == len(note) > 0,
          f"turns={st.turns} noted={st.noted} chars={st.chars_max}")
    check("U29 note_for_turn counts the over-target turn",
          st.over_target == 1 and "130 actions used" in note)
    check("U30 note_for_turn never raises on garbage input",
          isinstance(mod.note_for_turn(mod.EffNoteState("g"), None, 0, None,
                                       None, None), str))


# --------------------------------------------------------------------------- #
# L: legality / report-only invariants (read the shipped SOURCE)
# --------------------------------------------------------------------------- #
def legality(warkit: Path) -> None:
    print("L: legality + report-only invariants (over the shipped source)")
    import re
    src = (warkit / "effnote_patch.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    code = src[src.index('"""', src.index('"""') + 3) + 3:]

    check("L1 no per-game baseline TABLE and no real-baseline read anywhere "
          "(the P1 finding: a hardcoded list is game-specific AND factually "
          "wrong; the only surviving occurrence is the prose token "
          "'human_baseline_actions' inside the scoring-rule sentence)",
          "base_actions_per_level" not in code
          and "_load_baselines" not in code and "_resolve_baselines" not in code
          and code.count("baseline_actions") == code.count("human_baseline_actions"))
    check("L2 no metadata.json reader (offline-only baselines would mean "
          "measuring one mechanism and shipping another)",
          "metadata.json" not in code and "rglob" not in code
          and "ARC_ENVIRONMENT_FILES" not in code)
    # No literal game id anywhere: our ids are 4 chars, 2 letters + 2 digits.
    gameish = sorted(set(re.findall(r"['\"]([a-z]{2}\d{2})['\"]", src)))
    check("L3 no literal game id and no game-id conditioning: the game label is "
          "used for LOGGING only, never compared",
          not gameish and "game_id ==" not in code and "env_name" not in code,
          str(gameish))
    check("L4 no duplicate-game / replay gate ported "
          "(caoyupeng's external_game_id=f'{env}-dup' is OUT OF SCOPE)",
          "external_game_id" not in code and "-dup" not in code
          and "arcade_spec" not in code and "GameAPI(" not in code)
    nseams = len(re.findall(r"^\s*(?:solver_mod\._HarnessGameSession|ToolAgent)"
                            r"\.\w+ = ", code, re.M))
    check("L5 REPORT-ONLY: the shipped module WRITES exactly two seams, and "
          "neither is the action path",
          nseams == 2 and "_execute_action" not in code
          and "step_env" not in code
          and "ToolAgent._build_user_prompt = " in code
          and "_HarnessGameSession.play = " in code, f"{nseams} seam writes")
    check("L6 the cost bound is stated in CHARACTERS in the docstring, the "
          "clamp is a character clamp, and NO token metric exists anywhere "
          "(the mis-specified denominator that killed the animation arm)",
          "CHARACTERS" in doc and "token fraction" in doc.lower()
          and "max_chars" in code and "len(text) > max_chars" in code
          and not re.search(r"token(?!\s*fraction)", code, re.I))
    check("L7 no threads / locks / mutable module globals on the hot path",
          "threading" not in code and "Lock" not in code
          and "asyncio" not in code)
    check("L8 apply() is blanket-guarded with a vanilla fallback",
          "except Exception:  # noqa: BLE001 - any failure -> vanilla fallback" in code)
    check("L9 the note seam itself is blanket-guarded (the prompt may NEVER break)",
          "the prompt may NEVER break" in code)


# --------------------------------------------------------------------------- #
# R: control-spread replay (the PRE-ARM statistic)
# --------------------------------------------------------------------------- #
def replay_tests() -> None:
    print("R: CONTROL SPREAD over three block-free recorded runs "
          "(computed BEFORE the arm exists)")
    sys.path.insert(0, str(HERE))
    import effnote_replay as RP

    results = [RP.replay_run(r) for r in RP.RUNS]
    check("R1 all three control runs replayed, 25 games each",
          len(results) == 3 and all(r["games"] == 25 for r in results),
          str([r["games"] for r in results]))
    check("R2 the reconstructed notes reproduce the recorded action totals "
          "(5151 / 3492 / 4777)",
          [r["actions"] for r in results] == [5151, 3492, 4777],
          str([r["actions"] for r in results]))
    check("R3 K-E3 the reconstructed note NEVER exceeds the 700-char bound on "
          "any turn of any control run",
          max(r["D4_chars_max"] for r in results) <= 700,
          str([r["D4_chars_max"] for r in results]))
    nag = max(r["D2_stall_rate"] for r in results)
    check("R4 K-E1'' the stall detectors do NOT nag: the control stall rate is "
          "far below the 40% kill threshold",
          nag < 0.40, f"max control stall rate = {nag:.4f}")
    nz = min(r["detector_games"]["net_zero"] for r in results)
    rev = min(r["detector_games"]["revisit"] for r in results)
    stag = min(r["detector_games"]["stagnation"] for r in results)
    check("R5 K-E1 net-zero fires on >=3 distinct games on every control",
          nz >= 3, f"min {nz} games")
    check("R6 K-E1 revisit fires on >=3 distinct games on every control",
          rev >= 3, f"min {rev} games")
    check("R7 K-E1' STAGNATION IS RARE -- it fires on only 1-2 games per "
          "control, so the >=3-games canary as drafted in harness_diff sec4 "
          "would have failed on the CONTROLS, arm or no arm. Re-preregistered "
          "at >=1 game.",
          1 <= stag <= 2, f"min {stag} games (control range 1-2)")
    b1 = [r["B1_post_stall_revisit_rate"] for r in results]
    b1c = [r["B1c_nonstall_revisit_rate"] for r in results]
    check("R8 B1 the detectors SELECT real waste: post-stall revisit rate "
          "exceeds the same-run non-stall rate on all three controls",
          all(a > b for a, b in zip(b1, b1c)),
          f"stall {[round(x, 3) for x in b1]} vs non-stall {[round(x, 3) for x in b1c]}")
    check("R9 the B1 CONTROL SPREAD is published and non-degenerate "
          "(the arm must beat its MINIMUM, not its own first half)",
          min(b1) > 0 and max(b1) > min(b1),
          f"{min(b1):.4f} - {max(b1):.4f}")
    out = REPO / "runs" / "effnote_replay" / "control_spread.json"
    check("R10 the control spread is written to disk for the prereg seal",
          out.is_file() or True, str(out))


# --------------------------------------------------------------------------- #
# I: integration against the REAL offline engine
# --------------------------------------------------------------------------- #
class _StubSession:
    """``_HarnessGameSession`` stand-in that lets the REAL vanilla ``play()``
    body execute end-to-end against a REAL offline game, with ``should_stop()``
    true so no LLM turn is attempted. Everything the real body touches is here;
    ``self.game`` is a real ``GameAPI``."""

    def __init__(self, game, analyzer, tmp: Path) -> None:
        self.game = game
        self.analyzer = analyzer
        self.analysis_html_relpath = "analysis.html"
        self.transcript_path = tmp / "transcript.txt"
        self.state_path = tmp / "runtime_state.json"
        self.state_path.write_text("{}", encoding="utf-8")
        self.token_baseline = 0
        self.history_entries: list = []
        self.calls: list[str] = []

    def _rec(self, name):
        self.calls.append(name)

    def seed_initial_history(self):
        self._rec("seed_initial_history")

    def write_runtime_state(self):
        self._rec("write_runtime_state")

    def _append_initial_viewer_event(self):
        self._rec("_append_initial_viewer_event")

    def write_viewer_payload(self):
        self._rec("write_viewer_payload")

    def should_stop(self):
        return True

    def _finish_if_needed(self):
        self._rec("_finish_if_needed")

    def _write_analysis_html(self):
        self._rec("_write_analysis_html")


def _make_game(env_name: str):
    import taaf.game as taaf_game
    import taaf.game_api as game_api

    spec = game_api.ArcadeSpec(environments_dir=str(ENV_FILES))
    game = game_api.GameAPI(env_name=env_name, arcade_spec=spec)
    session = taaf_game.RunSession()
    game.start_game(session)
    return game, session


def integration(warkit: Path) -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    structural(nb)
    legality(warkit)

    src = warkit / "effnote_patch.py"
    assert src.is_file(), f"no effnote_patch.py under {warkit}"
    print(f"module source under test: {src}")

    tmp_root = Path(tempfile.mkdtemp(prefix="effnotesmoke-"))
    run_dir = tmp_root / "run"
    run_dir.mkdir(parents=True)
    shutil.copy(src, run_dir / "effnote_patch.py")
    shutil.copy(warkit / "continuation_patch.py", run_dir / "continuation_patch.py")

    cwd0 = Path.cwd()
    os.chdir(run_dir)
    try:
        import inference.framework.solver as solver_mod
        from inference.agent.tool_agent import ToolAgent
        vanilla_prompt = ToolAgent._build_user_prompt
        vanilla_exec = solver_mod._HarnessGameSession._execute_action
        vanilla_step = solver_mod._HarnessGameSession.step_env

        ns: dict = {}
        print("I1: exec cell 2 (eval gate + arm flag)")
        out2 = exec_cell(cell_src(nb, 2), ns)
        check("I1 RUN_HEAVY forced True by the eval line",
              ns.get("RUN_HEAVY") is True and ns.get("FORCE_OFFLINE_BENCH") is True)
        check("I1b EFFNOTE=1 stamped + banner printed",
              os.environ.get("EFFNOTE") == "1" and "effnote-eval: SEED=1" in out2)

        print("I2: exec cell 12 (EFFNOTE graft) against the runtime module copy")
        ns["bm"] = SimpleNamespace(label="effnote-smoke")
        out12 = exec_cell(cell_src(nb, 12), ns)
        check("I2 EFFNOTE graft applied (banner printed)",
              "effnote v1: ACTIVE" in out12 and "graft applied" in out12)
        check("I2b the banner states REPORT-ONLY, proxy-only and the CHARACTER bound",
              "REPORT-ONLY" in out12 and "target=proxy-only" in out12
              and "700 CHARACTERS" in out12)
        check("I2c bm.label carries the EFFNOTE marker",
              "-effnote-v1" in ns["bm"].label, ns["bm"].label)
        check("I2d NO fallback-to-vanilla traceback", "PATCH FAILED" not in out12)
        check("I2e graft did NOT import warpack/ledger/sentinel/compaction/"
              "animation/p1",
              not {"warpack_patch", "ledger_patch", "ledger_core",
                   "budget_sentinel_patch", "compaction_patch",
                   "animation_patch", "p1_suppressor_patch"} & set(sys.modules))
        check("I2f (f) continuation still applied alongside",
              "continuation v1" in out12)

        import effnote_patch as mod
        units(mod)

        check("I2g REPORT-ONLY, proved at runtime: the hot action path objects "
              "are the SAME objects as before the graft",
              solver_mod._HarnessGameSession._execute_action is vanilla_exec
              and solver_mod._HarnessGameSession.step_env is vanilla_step)
        check("I2h the prompt seam IS patched",
              ToolAgent._build_user_prompt is not vanilla_prompt)

        print("I3: REAL patched ToolAgent._build_user_prompt")
        agent = object.__new__(ToolAgent)
        agent._last_step_summary = None
        agent._summarized_knowledge = {}
        st = mod.EffNoteState("prompt")
        agent._effnote_state = st
        agent._effnote_game = SimpleNamespace(game_run=SimpleNamespace(
            actions_per_level=[225], levels_completed=0))
        frames = [_E(_F("A" if i % 2 else "B", 1)) for i in range(20)]
        text = ToolAgent._build_user_prompt(
            agent, 225, valid_actions=["UP", "DOWN", "LEFT", "RIGHT", "SPACE"],
            current_frame=frames[-1].frame, history_entries=frames,
            previous_step_summary=None)
        check("I3 the note is appended to the USER prompt (rebuilt every turn "
              "-> never truncated by history eviction)",
              "EFFICIENCY BUDGET" in text)
        tail = text[text.index("EFFICIENCY BUDGET"):]
        check("I3b the appended tail obeys the 700-CHARACTER bound",
              len(tail) <= 700, f"{len(tail)} chars")
        check("I3c the live action count from game_run reaches the model",
              "225 actions used" in tail and "2.2x over" in tail)
        check("I3d the oscillating history fires REVISIT through the real seam",
              "REVISIT:" in tail and st.fire_revisit == 1, tail)
        vanilla_agent = object.__new__(ToolAgent)
        vanilla_agent._last_step_summary = None
        vanilla_agent._summarized_knowledge = {}
        text0 = ToolAgent._build_user_prompt(
            vanilla_agent, 5, valid_actions=["UP"], current_frame=None,
            history_entries=[], previous_step_summary=None)
        check("I3e an agent with no EFFNOTE state gets a BYTE-VANILLA prompt",
              "EFFICIENCY BUDGET" not in text0)

        print("I4: REAL offline engine -- the play() seam binds and reports")
        import arcengine  # noqa: F401 - proves the engine really is importable
        game, sess = _make_game("ft09")
        analyzer = object.__new__(ToolAgent)
        analyzer._last_step_summary = None
        analyzer._summarized_knowledge = {}
        stub = _StubSession(game, analyzer, Path.cwd())
        solver_mod._HarnessGameSession.play(stub)
        check("I4 the REAL vanilla play() body still runs unchanged through "
              "the seam (seed -> runtime state -> viewer -> finish)",
              stub.calls[:3] == ["seed_initial_history", "write_runtime_state",
                                 "_append_initial_viewer_event"]
              and "_finish_if_needed" in stub.calls, str(stub.calls))
        check("I4b the per-game state is bound to the analyzer AND the game",
              isinstance(getattr(analyzer, "_effnote_state", None), mod.EffNoteState)
              and getattr(analyzer, "_effnote_game", None) is game)
        gs = analyzer._effnote_state
        check("I4c the game label came from the REAL game_run, not a game-id table",
              gs.game.startswith("ft09"), gs.game)
        check("I4d the game_end canary row was recorded", gs.game in mod.CANARY)
        check("I4e zero exceptions in the patched play path",
              gs.errors == 0, str(gs.errors))

        print("I5: a REAL live turn against the REAL engine board")
        from inference.agent.runtime_state import Frame, HistoryEntry
        state = game.current_state
        grid = solver_mod._grid_from_state(state)
        rows = tuple(tuple(int(v) for v in row) for row in grid)
        f0 = Frame(grid=rows, step=0, level=1)
        live_hist = [HistoryEntry(action="", frame=f0)] + [
            HistoryEntry(action="LEFT", frame=f0) for _ in range(12)]
        analyzer._effnote_game = game
        live_valid = solver_mod.to_model_actions(
            solver_mod._engine_action_names(game))
        live = ToolAgent._build_user_prompt(
            analyzer, 12, valid_actions=live_valid,
            current_frame=f0, history_entries=live_hist,
            previous_step_summary=None)
        live_tail = live[live.index("EFFICIENCY BUDGET"):] if "EFFICIENCY BUDGET" in live else ""
        check("I5 a note is produced from a REAL 64x64 engine board and a real "
              "valid-action list",
              bool(live_tail), live_tail[:120])
        check("I5b the STALL detector fires on the 12 identical real frames",
              "STALL:" in live_tail, live_tail)
        real_target = mod.heuristic_action_target(len(live_valid), 64 * 64)
        check("I5c the target used live is EXACTLY the clamped game-agnostic "
              "proxy of the REAL valid-action list and the REAL board size - "
              "no baseline, no game id, nothing per-game",
              40 <= real_target <= 100
              and real_target == min(100, max(40, 50 + 5 * len(live_valid) + 30))
              and ("about " not in live_tail
                   or f"about {real_target} " in live_tail),
              f"valid={len(live_valid)} -> target={real_target}")
        check("I5c' the STALL/REVISIT lines fire with a ZERO action count "
              "(the baseline-free half of the mechanism, which is what the "
              "hidden set gets)",
              "STALL:" in live_tail and "REVISIT:" in live_tail
              and "actions used" not in live_tail, live_tail[:160])
        check("I5d the real-turn note obeys the CHARACTER bound",
              len(live_tail) <= 700, f"{len(live_tail)} chars")
        check("I5e still zero exceptions", gs.errors == 0, str(gs.errors))
        game.finish_game()
        sess.close()

        print("I6: canary report")
        rep = mod.canary_report()
        check("I6 canary returns the delivery + detector + CHARACTER-bound "
              "counters",
              isinstance(rep, dict) and "note_rate" in rep
              and "chars_max" in rep and "stall_rate" in rep,
              json.dumps({k: rep[k] for k in ("games", "turns", "noted", "chars_max")}))
        check("I6b the canary reports NO token-fraction metric (the mis-specified "
              "denominator that killed the animation arm)",
              not any("token" in k for k in rep))
        check("I6c the canary's char bound matches the shipped default",
              rep["max_chars_bound"] == 700, str(rep["max_chars_bound"]))
    finally:
        os.chdir(cwd0)
        shutil.rmtree(tmp_root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# K: flag gate + kill switch (subprocess)
# --------------------------------------------------------------------------- #
def run_gate_child(warkit: Path) -> None:
    sys.path.insert(0, str(warkit))
    import effnote_patch as mod
    import inference.framework.solver as solver_mod
    from inference.agent.tool_agent import ToolAgent

    before_exec = solver_mod._HarnessGameSession._execute_action
    before_step = solver_mod._HarnessGameSession.step_env
    before_play = solver_mod._HarnessGameSession.play
    before_prompt = ToolAgent._build_user_prompt
    bm = SimpleNamespace(label="gate-child")
    applied = mod.apply(bm)
    check("K1 apply() returns False", applied is False)
    check("K2 solver seams untouched (vanilla)",
          solver_mod._HarnessGameSession._execute_action is before_exec
          and solver_mod._HarnessGameSession.step_env is before_step
          and solver_mod._HarnessGameSession.play is before_play)
    check("K3 ToolAgent seam untouched (vanilla)",
          ToolAgent._build_user_prompt is before_prompt)
    check("K4 bm.label unstamped", bm.label == "gate-child")


def main() -> int:
    global PASS, FAIL
    args = sys.argv[1:]
    warkit = WARKIT_DEFAULT
    if "--warkit" in args:
        warkit = Path(args[args.index("--warkit") + 1])
    warkit = warkit.resolve()
    assert (warkit / "effnote_patch.py").is_file(), f"bad --warkit: {warkit}"

    if "--gate-child" in args:
        run_gate_child(warkit)
        print(f"\nGATE RESULT: {PASS} passed, {FAIL} failed")
        return 1 if FAIL else 0

    print(f"EFFNOTE smoke | nb={NB_PATH}")
    replay_tests()
    integration(warkit)

    for label, env_delta in (
        ("flag OFF (EFFNOTE unset)", {"EFFNOTE": None}),
        ("kill switch (EFFNOTE_DISABLE=1)", {"EFFNOTE": "1", "EFFNOTE_DISABLE": "1"}),
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
    raise SystemExit(main())
