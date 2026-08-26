#!/usr/bin/env python
"""LOCAL PRE-PUSH GATE for the ARC-AGI-3 campaign  (lane: local-rail)

    uv run python scripts/local_gate.py --arm <name> [--notebook <path>]

ONE command, run BEFORE any `kaggle kernels push`. Catches everything that is
catchable without a GPU. It NEVER seals a verdict: Kaggle remains the only
certification rail (env-mismatch confirmed 5x). See docs/local_gate.md.

WHY IT EXISTS
-------------
Three times in two days an incoming/campaign scorer was validated against
SELF-WRITTEN FIXTURES rather than real artifacts, and each time the fixture
keys did not match the keys the REAL taaf harness emits:

  D1 (08-21)  benchmark reader keyed on `games`            real: `game_runs`
              -> n_games 0 -> FALSE INFRA DEATH on a healthy arm
  D2 (08-21)  actions reader keyed on `total_actions`      real: `actions_per_level`
              -> actions 0 -> the sealed wallclock-KILL line fired on ratio 0.00
  D3 (08-22)  score reader keyed on `score`                real: `final_score`
              -> mean_score 0

"Internal consistency is not correctness." The top-priority check group here
executes every reader against REAL artifacts already on disk and cross-checks
what it extracts against an INDEPENDENT direct computation from the raw JSON.

CHECK GROUPS
------------
  R  real-artifact reader certification   (priority 1; the recurring defect)
  N  notebook static gate                 (compile / tokens / flags / determinism
                                           / diff-vs-base / metadata; composes
                                           scripts/preflight.py, offline)
  H  harness smoke without the 27B        (fake OpenAI-shaped server + the REAL
                                           competition simulator; CPU, seconds)
  A  arm matrix                           (own-arm markers present, every other
                                           arm's REAL artifact refused)
  X  wrapped existing suites              (scorer selftests, private_smoke, ...)

SELF-NEGATIVE-CONTROLS (campaign doctrine: prove the instrument can refuse)
    uv run python scripts/local_gate.py --self-test
deliberately breaks a notebook and a reader (including verbatim reconstructions
of D1/D2/D3) and asserts this gate FAILS on each.

SAFETY
------
Writes NOTHING outside a temp dir. Builders are re-run with their OUT_* paths
redirected into a scratch tree, and check X4 hashes notebooks/ before and after
to prove nothing in any lane's staging moved. No kernel pushes, no submissions,
no queue edits.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
DUCK = REPO / "duck_eval"
RUNS = REPO / "runs"
NOTEBOOKS = REPO / "notebooks"
PY = sys.executable

# Scorer modules live in per-lane dirs and import each other BY NAME
# (private_score does `from graft_score import ...`), so every lane dir goes on
# sys.path once, up front.
#
# scripts/ is deliberately NOT added: it contains queue.py, which would shadow
# the stdlib `queue` and break every downstream import of the harness
# (threading/concurrent.futures pull it in). preflight is loaded by file path.
#
# Running `python scripts/local_gate.py` puts scripts/ on sys.path[0] by
# default, so it is stripped here before anything else imports.
_SELF_DIR = str(Path(__file__).resolve().parent)
sys.path[:] = [p for p in sys.path
               if p and Path(p).resolve() != Path(_SELF_DIR).resolve()]
for _d in (DUCK / "private", DUCK / "graft", DUCK / "q38", DUCK / "budget", DUCK / "warpack",
           DUCK / "a17", DUCK / "execwm", DUCK / "p2"):
    if _d.is_dir() and str(_d) not in sys.path:
        sys.path.insert(0, str(_d))


# ===========================================================================
# 0. RESULT TABLE
# ===========================================================================
PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass
class Check:
    group: str
    code: str
    status: str
    message: str
    detail: str = ""


class Report:
    def __init__(self, arm: str, mode: str) -> None:
        self.arm = arm
        self.mode = mode
        self.checks: list[Check] = []
        self.t0 = time.time()

    def add(self, group: str, code: str, status: str, message: str, detail: str = "") -> Check:
        c = Check(group, code, status, message, detail)
        self.checks.append(c)
        return c

    def ok(self, g, c, m, d=""):   return self.add(g, c, PASS, m, d)
    def bad(self, g, c, m, d=""):  return self.add(g, c, FAIL, m, d)
    def warn(self, g, c, m, d=""): return self.add(g, c, WARN, m, d)
    def skip(self, g, c, m, d=""): return self.add(g, c, SKIP, m, d)

    @property
    def n_fail(self) -> int:  return sum(1 for c in self.checks if c.status == FAIL)
    @property
    def n_warn(self) -> int:  return sum(1 for c in self.checks if c.status == WARN)
    @property
    def n_pass(self) -> int:  return sum(1 for c in self.checks if c.status == PASS)
    @property
    def n_skip(self) -> int:  return sum(1 for c in self.checks if c.status == SKIP)
    @property
    def verdict(self) -> str: return "FAIL" if self.n_fail else "PASS"

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "mode": self.mode,
            "verdict": self.verdict,
            "n_pass": self.n_pass, "n_fail": self.n_fail,
            "n_warn": self.n_warn, "n_skip": self.n_skip,
            "elapsed_s": round(time.time() - self.t0, 2),
            "checks": [c.__dict__ for c in self.checks],
        }

    def render(self) -> str:
        w = max([len(c.code) for c in self.checks] + [8])
        lines = ["", "=" * 100,
                 f"LOCAL GATE  arm={self.arm}  mode={self.mode}",
                 "=" * 100]
        cur = None
        for c in self.checks:
            if c.group != cur:
                cur = c.group
                lines.append(f"-- {GROUP_TITLES.get(cur, cur)}")
            mark = {PASS: "ok", FAIL: "XX", WARN: "!!", SKIP: "--"}[c.status]
            lines.append(f"  [{mark}] {c.code:<{w}}  {c.message}")
            if c.detail and c.status in (FAIL, WARN):
                for ln in str(c.detail).splitlines()[:12]:
                    lines.append(f"         | {ln}")
        lines += ["=" * 100,
                  f"  {self.verdict}   pass={self.n_pass} fail={self.n_fail} "
                  f"warn={self.n_warn} skip={self.n_skip}  "
                  f"({round(time.time() - self.t0, 1)}s)",
                  "=" * 100]
        if self.n_fail:
            lines += ["  DO NOT PUSH. Every FAIL above is reproducible on CPU in seconds.",
                      "  This gate NEVER certifies: a PASS licenses a Kaggle BUILD, not a verdict."]
        else:
            lines += ["  Local gate clear -> a Kaggle BUILD is licensed. Certification is still",
                      "  the kernel's job (env-mismatch confirmed 5x); see docs/local_gate.md."]
        return "\n".join(lines)


GROUP_TITLES = {
    "R": "R  REAL-ARTIFACT READER CERTIFICATION (priority 1)",
    "N": "N  NOTEBOOK STATIC GATE",
    "H": "H  HARNESS SMOKE WITHOUT THE 27B (fake LLM + real simulator)",
    "A": "A  ARM MATRIX / CROSS-ARM NEGATIVE CONTROLS",
    "P": "P  P0 PERMANENT INSTRUMENTS (guard behaviour + cadence)",
    "X": "X  WRAPPED EXISTING SUITES + DO-NO-HARM",
    "S": "S  SELF-TEST (negative controls on this gate)",
}


# ===========================================================================
# 1. ARM REGISTRY
# ===========================================================================
GRAFT_TOKENS = ("[goalkeep] armed", "[hudmask] armed", "[clickmap] armed",
                "[banking] armed", "[searchmap] armed", "[transfer] armed")


@dataclass
class Arm:
    name: str
    scorer_module: str                       # importable module name
    scorer_path: Path
    scorer_arm_flag: str | None = None       # --arm value, for multi-arm scorers
    artifact: Path | None = None             # arm-matched REAL pull directory
    notebook: Path | None = None
    kernel: str = ""
    builder_path: Path | None = None
    builder_call: str = "build"
    builder_args: tuple = ()
    base_notebook: Path | None = None
    expect_diff_cells: tuple[int, ...] | None = None      # MODIFIED cells vs base
    expect_inserted_cells: tuple[int, ...] = ()           # cells with no base peer
    expect_n_cells: int | None = None
    forbidden_tokens: tuple[str, ...] = ()
    required_literals: tuple[tuple[str, str], ...] = ()   # (NAME, "True"/"False")
    extra_suites: tuple[tuple[str, tuple], ...] = ()      # (label, argv)
    # Arms that run the SAME vehicle bytes (e.g. a replication seed). Their
    # artifacts MUST certify under each other, so they are excluded from the
    # cross-arm negative controls -- a control that cannot help but fire is
    # noise, and noise is how a gate gets ignored.
    sibling_arms: tuple[str, ...] = ()
    note: str = ""


_PRIV = DUCK / "private" / "private_score.py"
_PRIV_NB = NOTEBOOKS / "q38-private-eval" / "arc3-q38-private-eval.ipynb"
_FIELD_NB = NOTEBOOKS / "q38-field-eval" / "arc3-q38-field-eval.ipynb"
_PRIV_BUILD = DUCK / "private" / "build_private_eval.py"
_PRIV_FORBIDDEN = ("taaf_grafts", "install(bm", "reasoning_effort", "banking",
                   "searchmap", "clickmap", "litellm")
_PRIV_SUITES = (("private_smoke", (str(DUCK / "private" / "private_smoke.py"),)),)


def _priv(name: str, flag: str, artifact: str | None, e1: bool, e2: bool) -> Arm:
    return Arm(
        name=name, scorer_module="private_score", scorer_path=_PRIV,
        scorer_arm_flag=flag,
        artifact=(RUNS / "kernel_pulls" / artifact) if artifact else None,
        notebook=_PRIV_NB, kernel="canivel/arc3-q38-private-eval",
        builder_path=_PRIV_BUILD, builder_args=(e1, e2),
        base_notebook=_FIELD_NB,
        # prereg: "diff confined to cells [0, 3, 5] + inserted [8] (12 cells total)"
        expect_diff_cells=(0, 3, 5), expect_inserted_cells=(8,), expect_n_cells=12,
        forbidden_tokens=_PRIV_FORBIDDEN,
        required_literals=(("PRIVATE_EDGE1_CTX_RAISE", str(e1)),
                           ("PRIVATE_EDGE2_VISIBLE_CONTRACT", str(e2))),
        extra_suites=_PRIV_SUITES,
    )


ARMS: dict[str, Arm] = {
    "private-base":   _priv("private-base",   "base",   "private_base_v1",  False, False),
    "private-edge1":  _priv("private-edge1",  "edge1",  "private_edge1_v2", True,  False),
    "private-edge2":  _priv("private-edge2",  "edge2",  None,               False, True),
    "private-edge12": _priv("private-edge12", "edge12", None,               True,  True),

    "budget-t05": Arm(
        name="budget-t05", scorer_module="budget_score",
        scorer_path=DUCK / "budget" / "budget_score.py",
        artifact=None,
        notebook=NOTEBOOKS / "budget-t05-eval" / "arc3-budget-t05-eval.ipynb",
        kernel="canivel/arc3-budget-t05-eval",
        builder_path=DUCK / "budget" / "build_budget_eval.py", builder_args=("t05",),
        expect_n_cells=11,
        forbidden_tokens=("taaf_grafts", "install(bm", "reasoning_effort", "litellm",
                          "= 7920.0", "= 23760.0", "EDGE1", "EDGE2"),
        note="ARM1 budget T0.5: one-constant change of the FOYSAL vehicle (3960.0)",
    ),
    "budget-t3": Arm(
        name="budget-t3", scorer_module="budget_score",
        scorer_path=DUCK / "budget" / "budget_score.py",
        artifact=None,
        notebook=NOTEBOOKS / "budget-t3-eval" / "arc3-budget-t3-eval.ipynb",
        kernel="canivel/arc3-budget-t3-eval",
        builder_path=DUCK / "budget" / "build_budget_eval.py", builder_args=("t3",),
        expect_n_cells=11,
        forbidden_tokens=("taaf_grafts", "install(bm", "reasoning_effort", "litellm",
                          "= 7920.0", "= 3960.0", "EDGE1", "EDGE2"),
        note="ARM1 budget T3: one-constant change of the FOYSAL vehicle (23760.0)",
    ),
    "q38-field": Arm(
        name="q38-field", scorer_module="q38field_score",
        scorer_path=DUCK / "q38" / "q38field_score.py",
        artifact=RUNS / "kernel_pulls" / "q38_field_v1",
        notebook=_FIELD_NB, kernel="canivel/arc3-q38-field-eval",
        expect_n_cells=11,
        forbidden_tokens=("taaf_grafts", "install(bm", "reasoning_effort", "litellm"),
        note="hand-rebased FOYSAL vehicle: no builder, so N4 (determinism) is N/A",
    ),
    "graft-floor": Arm(
        name="graft-floor", scorer_module="graft_score",
        scorer_path=DUCK / "graft" / "graft_score.py",
        artifact=RUNS / "kernel_pulls" / "graft_floor_v1",
        notebook=NOTEBOOKS / "graft-floor-eval" / "arc3-graft-floor-eval.ipynb",
        kernel="canivel/arc3-graft-floor-eval",
        builder_path=DUCK / "graft" / "build_graft_eval.py",
        base_notebook=NOTEBOOKS / "duckfork"
                      / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb",
        expect_diff_cells=(2, 4, 6, 12, 14), expect_n_cells=17,
        forbidden_tokens=("litellm", "[banking] armed", "[clickmap] armed"),
        sibling_arms=("graft-confirm",),
        extra_suites=(("graft_smoke", (str(DUCK / "graft" / "graft_smoke.py"),)),),
    ),
    "graft-confirm": Arm(
        name="graft-confirm", scorer_module="graft_confirm_score",
        scorer_path=DUCK / "graft" / "graft_confirm_score.py",
        artifact=RUNS / "kernel_pulls" / "graft_confirm_v1",
        notebook=NOTEBOOKS / "graft-floor-eval" / "arc3-graft-floor-eval.ipynb",
        kernel="canivel/arc3-graft-floor-eval",
        base_notebook=NOTEBOOKS / "duckfork"
                      / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb",
        expect_diff_cells=(2, 4, 6, 12, 14), expect_n_cells=17,
        forbidden_tokens=("litellm", "[banking] armed", "[clickmap] armed"),
        sibling_arms=("graft-floor",),
        note="replication seed of graft-floor: byte-identical treatment "
             "(sha b0316275f53c6c85), so the two artifacts certify under each other",
    ),
    "q38-graft": Arm(
        name="q38-graft", scorer_module="q38graft_score",
        scorer_path=DUCK / "graft" / "q38graft_score.py",
        artifact=RUNS / "kernel_pulls" / "q38graft_v1",
        notebook=NOTEBOOKS / "q38-graft-eval" / "arc3-q38-graft-eval.ipynb",
        kernel="canivel/arc3-q38-graft-eval",
        builder_path=DUCK / "graft" / "build_q38graft_eval.py",
        base_notebook=NOTEBOOKS / "duckfork"
                      / "tufa-labs-duck-harness-june-30-milestone-winner.ipynb",
        expect_n_cells=17,
        forbidden_tokens=("litellm", "[banking] armed", "[searchmap] armed"),
        extra_suites=(("graft_bundle_check",
                       (str(DUCK / "graft" / "graft_bundle_check.py"),)),),
    ),
    "execwm": Arm(
        name="execwm", scorer_module="execwm_score",
        scorer_path=DUCK / "execwm" / "execwm_score.py",
        artifact=None,   # no pull yet; first artifact registers here after v1
        notebook=NOTEBOOKS / "execwm-eval" / "arc3-execwm-eval.ipynb",
        kernel="canivel/arc3-execwm-eval",
        builder_path=DUCK / "execwm" / "build_execwm_eval.py",
        base_notebook=_FIELD_NB,
        # prereg execwm_prereg_2026-08-25.md: vehicle bytes untouched; ONE
        # inserted patch cell at 6 (12 cells total), no modified cells.
        expect_inserted_cells=(6,), expect_n_cells=12,
        forbidden_tokens=("taaf_grafts", "install(bm", "reasoning_effort",
                          "litellm", "[notes] persistent-namespace armed",
                          "[p2] reset semantics OK", "[banking] armed",
                          "[clickmap] armed"),
        extra_suites=(("ewm_smoke_fast",
                       (str(DUCK / "execwm" / "ewm_smoke.py"), "--fast")),
                      ("execwm_score_selftest",
                       (str(DUCK / "execwm" / "execwm_score.py"), "--selftest"))),
        note="exec-WM arm: executable world model (mine/verify/plan/fallback) "
             "as ONE inserted patch cell on the certified field-floor vehicle",
    ),
    "p2": Arm(
        name="p2", scorer_module="p2_score",
        scorer_path=DUCK / "p2" / "p2_score.py",
        artifact=None,   # no pull yet; the first artifact registers here after v1
        notebook=NOTEBOOKS / "p2-retry-eval" / "arc3-p2-retry-eval.ipynb",
        kernel="canivel/arc3-p2-retry-eval",
        builder_path=DUCK / "p2" / "build_p2_eval.py",
        base_notebook=_FIELD_NB,
        # prereg p2_reset_retry_prereg_2026-08-22.md: vehicle bytes untouched;
        # ONE inserted patch cell at 6 (12 cells total), no modified cells.
        expect_inserted_cells=(6,), expect_n_cells=12,
        forbidden_tokens=("taaf_grafts", "install(bm", "reasoning_effort",
                          "litellm", "[notes] persistent-namespace armed",
                          "[execwm] armed", "[banking] armed", "[clickmap] armed",
                          "[cadence] effort pin armed", "[cadence] max_output armed"),
        extra_suites=(("p2_episode_smoke",
                       (str(DUCK / "p2" / "p2_smoke.py"),)),
                      ("p2_trigger_smoke",
                       (str(DUCK / "p2" / "p2_trigger_smoke.py"),)),
                      ("p2_score_selftest",
                       (str(DUCK / "p2" / "p2_score.py"), "--selftest")),
                      ("p2_cell_smoke",
                       (str(DUCK / "p2" / "p2_cell_smoke.py"),))),
        note="P2 arm: reset-anchored episodic retry -- attempt() in the sandbox "
             "plus an H=4 stuck trigger, as ONE inserted patch cell on the "
             "certified field-floor vehicle. Fireability pre-measured at 19/25 "
             "on this vehicle (p2_trigger_fireability_2026-08-26.md).",
    ),
}

# Which REAL pull directory belongs to which arm. Used by the cross-arm matrix:
# every arm's certification must REFUSE every other arm's real artifact.
ARM_ARTIFACTS = {n: a.artifact for n, a in ARMS.items() if a.artifact and a.artifact.is_dir()}


# ===========================================================================
# 2. REAL-ARTIFACT CORPUS  (referenced by path, never copied)
# ===========================================================================
CORPUS_ROOTS = (RUNS, DUCK)
# Directories that are working scratch for a live lane, or vendored upstream
# source trees that merely happen to contain a file called benchmark.json.
CORPUS_EXCLUDE = ("__pycache__", "/.venv/", "/site-packages/", "/node_modules/",
                  "_fixtures", "_test_fixtures", "/harness_diff_", "/fork_audit/")


def _looks_real(bench: Any) -> bool:
    """A REAL taaf benchmark artifact: object with a non-empty game_runs list
    whose entries carry the keys the harness actually emits."""
    if not isinstance(bench, dict):
        return False
    runs = bench.get("game_runs")
    if not isinstance(runs, list) or not runs:
        return False
    r0 = runs[0]
    return isinstance(r0, dict) and "levels_completed" in r0 and "final_score" in r0


@dataclass
class Artifact:
    path: Path
    run_dir: Path
    producer: str          # which arm/kernel produced it, best effort
    n_games: int
    bytes: int
    oracle_error: str = ""   # non-empty => DEGENERATE (aborted/legacy run)

    @property
    def degenerate(self) -> bool:
        return bool(self.oracle_error)

    @property
    def rel(self) -> str:
        try:
            return self.path.relative_to(REPO).as_posix()
        except ValueError:
            return self.path.as_posix()


_PRODUCER_HINTS = {
    "private_base_v1": "private-base", "private_edge1_v2": "private-edge1",
    "q38_field_v1": "q38-field", "graft_floor_v1": "graft-floor",
    "graft_confirm_v1": "graft-confirm", "q38graft_v1": "q38-graft",
}


def build_corpus(limit: int | None = None) -> list[Artifact]:
    """Index every REAL benchmark artifact on disk. Never copies: the index
    references files in place, so a 400 KB pull costs nothing."""
    seen: set[Path] = set()
    out: list[Artifact] = []
    for root in CORPUS_ROOTS:
        if not root.is_dir():
            continue
        for p in root.rglob("benchmark.json"):
            sp = p.as_posix()
            if any(x in sp for x in CORPUS_EXCLUDE) or p in seen:
                continue
            seen.add(p)
            try:
                bench = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not _looks_real(bench):
                continue
            run_dir = p.parent
            producer = _PRODUCER_HINTS.get(run_dir.name, "")
            if not producer:
                producer = str(bench.get("label") or run_dir.name)
            try:
                oracle(bench)
                err = ""
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
            out.append(Artifact(p, run_dir, producer,
                                len(bench["game_runs"]), p.stat().st_size, err))
    out.sort(key=lambda a: a.rel)
    return out[:limit] if limit else out


# ===========================================================================
# 3. THE ORACLE — independent direct computation from the raw JSON
# ===========================================================================
# Deliberately shares NO code with any scorer and accesses every key STRICTLY
# (missing key -> ArtifactError, never a silent 0). This is the second opinion
# that D1/D2/D3 would each have failed against on day one.
class ArtifactError(Exception):
    pass


ORACLE_METRICS = ("n_games", "lc_total", "total_actions", "mean_score", "games_won")


def oracle(bench: Any) -> dict:
    if not isinstance(bench, dict):
        raise ArtifactError(f"benchmark root is {type(bench).__name__}, expected object")
    if "game_runs" not in bench:
        raise ArtifactError("no 'game_runs' key at the benchmark root "
                            "(defect D1: fixtures used 'games')")
    runs = bench["game_runs"]
    if not isinstance(runs, list):
        raise ArtifactError(f"'game_runs' is {type(runs).__name__}, expected list")
    if not runs:
        raise ArtifactError("'game_runs' is empty")
    lc = 0
    actions = 0
    scores: list[float] = []
    won = 0
    for i, r in enumerate(runs):
        if not isinstance(r, dict):
            raise ArtifactError(f"game_runs[{i}] is {type(r).__name__}, expected object")
        for key in ("levels_completed", "actions_per_level", "final_score"):
            if key not in r:
                raise ArtifactError(f"game_runs[{i}] has no {key!r}")
        try:
            lc += int(r["levels_completed"])
        except (TypeError, ValueError):
            raise ArtifactError(f"game_runs[{i}].levels_completed is "
                                f"{r['levels_completed']!r}, not an int") from None
        apl = r["actions_per_level"]
        if not isinstance(apl, list):
            raise ArtifactError(f"game_runs[{i}].actions_per_level is "
                                f"{type(apl).__name__}, expected list "
                                f"(defect D2: fixtures used scalar 'total_actions')")
        try:
            actions += sum(int(x) for x in apl)
            scores.append(float(r["final_score"]))
        except (TypeError, ValueError):
            raise ArtifactError(f"game_runs[{i}] carries a null/non-numeric "
                                f"final_score or actions_per_level entry "
                                f"(state={r.get('state')!r}) -- this run never "
                                f"finished; the artifact is DEGENERATE") from None
        if str(r.get("state", "")) == "won":
            won += 1
    return {"n_games": len(runs), "lc_total": lc, "total_actions": actions,
            "mean_score": sum(scores) / len(scores), "games_won": won}


# ===========================================================================
# 4. READER REGISTRY
# ===========================================================================
# Two tiers:
#   direct    a reader function callable on a parsed benchmark dict. Exercised
#             against EVERY artifact in the corpus -> the widest net.
#   pipeline  a scorer's full score(run_dir). Exercised against its OWN arm's
#             REAL pull directory (a foreign log would legitimately refuse).
@dataclass
class Reader:
    name: str
    kind: str                      # "direct" | "pipeline"
    fn: Callable
    provides: tuple[str, ...]      # subset of ORACLE_METRICS it returns
    arm: str = ""                  # pipeline readers: which arm's artifact


_MOD_CACHE: dict[str, Any] = {}


def load_module(name: str):
    """Import a lane module by name. `preflight` is loaded by FILE PATH so that
    scripts/ never has to go on sys.path (scripts/queue.py shadows stdlib
    `queue`, which silently breaks every harness import downstream)."""
    if name in _MOD_CACHE:
        return _MOD_CACHE[name]
    if name == "preflight":
        spec = importlib.util.spec_from_file_location(
            "_localgate_preflight", REPO / "scripts" / "preflight.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_localgate_preflight"] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(name)
    _MOD_CACHE[name] = mod
    return mod


def _quiet(fn, *a, **kw):
    """Scorers print freely and some raise SystemExit. Swallow both, and
    normalise every refusal into ArtifactError so 'refused' is one thing."""
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        try:
            return fn(*a, **kw), buf.getvalue()
        except SystemExit as e:
            raise ArtifactError(f"SystemExit({e.code}): "
                                f"{buf.getvalue().strip()[-300:]}") from e
        except ArtifactError:
            raise
        except Exception as e:
            raise ArtifactError(f"{type(e).__name__}: {e}") from e


def _direct_private(bench: dict) -> dict:
    m = load_module("private_score")
    (lc, actions, mean_score), _ = _quiet(m._lc_actions_score, bench)
    return {"lc_total": lc, "total_actions": actions, "mean_score": mean_score}


def _pipeline(module: str, arm_flag: str | None):
    def run(run_dir: Path) -> dict:
        m = load_module(module)
        if module == "private_score":
            bench, _ = _quiet(m._load_benchmark, run_dir)
            (lc, actions, ms), _ = _quiet(m._lc_actions_score, bench)
            cert, _ = _quiet(m.certify, run_dir, arm_flag)
            return {"lc_total": lc, "total_actions": actions, "mean_score": ms,
                    "n_games": len(bench.get("game_runs") or []),
                    "_certified": bool(cert)}
        res, _ = _quiet(m.score, run_dir)
        if not isinstance(res, dict):
            raise ArtifactError(f"{module}.score returned {type(res).__name__}")
        out = {}
        for k_out, k_in in (("lc_total", "lc_total"), ("total_actions", "total_actions"),
                            ("mean_score", "mean_score"), ("n_games", "n_games"),
                            ("games_won", "won")):
            if k_in in res:
                out[k_out] = res[k_in]
        if "games_won" not in out and "games_won" in res:
            out["games_won"] = res["games_won"]
        out["_verdict"] = res.get("verdict")
        out["_reason"] = res.get("reason", "")
        return out
    return run


def reader_registry() -> list[Reader]:
    rs = [Reader("private_score._lc_actions_score", "direct", _direct_private,
                 ("lc_total", "total_actions", "mean_score"))]
    for arm_name, arm in ARMS.items():
        if not (arm.artifact and arm.artifact.is_dir()):
            continue
        rs.append(Reader(f"{arm.scorer_module}.score[{arm_name}]", "pipeline",
                         _pipeline(arm.scorer_module, arm.scorer_arm_flag),
                         ("lc_total", "total_actions", "mean_score", "n_games"),
                         arm=arm_name))
    return rs


# ---- malformed artifacts: a reader must FAIL LOUDLY, never return zeros ----
def malformed_cases() -> dict[str, Any]:
    good_run = {"game_id": "g0", "levels_completed": 3, "final_score": 9.5,
                "actions_per_level": [11, 12, 13], "state": "gave_up"}
    return {
        "root_is_list": [good_run],
        "no_game_runs": {"label": "x", "n_passes": 1},
        "fixture_key_games": {"label": "x", "games": [good_run]},          # D1 shape
        "empty_game_runs": {"label": "x", "game_runs": []},
        "missing_levels_completed": {"game_runs": [
            {k: v for k, v in good_run.items() if k != "levels_completed"}]},
        "missing_final_score": {"game_runs": [                             # D3 shape
            {**{k: v for k, v in good_run.items() if k != "final_score"}, "score": 9.5}]},
        "scalar_total_actions": {"game_runs": [                            # D2 shape
            {**{k: v for k, v in good_run.items() if k != "actions_per_level"},
             "total_actions": 36}]},
        "run_is_string": {"game_runs": ["not-an-object"]},
    }


# ---- the three historical defects, reconstructed verbatim -----------------
# git history does not carry the broken versions (duck_eval/private/ is
# untracked), so they are reconstructed from the ITERATION_LOG post-mortems:
#   D1 ITERATION_LOG.md:1513   D2 :1517   D3 :1535
def defect_readers() -> dict[str, Callable[[dict], dict]]:
    def d1_games_key(bench: dict) -> dict:
        games = bench.get("games") or []                      # THE BUG
        return {"n_games": len(games),
                "lc_total": sum(int(g.get("levels_completed") or 0) for g in games)}

    def d2_total_actions(bench: dict) -> dict:
        games = bench.get("game_runs") or []
        return {"total_actions": sum(int(g.get("total_actions") or 0) for g in games)}

    def d3_score_key(bench: dict) -> dict:
        games = bench.get("game_runs") or []
        scores = [float(g.get("score", 0.0)) for g in games]   # THE BUG
        return {"mean_score": (sum(scores) / len(scores)) if scores else 0.0}

    return {"D1_games_key": d1_games_key,
            "D2_total_actions_key": d2_total_actions,
            "D3_score_key": d3_score_key}


# ===========================================================================
# 5. CHECK GROUP R — reader certification
# ===========================================================================
def _pipeline_reject_test(arm: Arm, art_dir: Path, cases: dict) -> list[str]:
    """Drop each malformed benchmark.json beside the arm's OWN real log and
    require the scorer to refuse. Returns the cases it scored anyway."""
    scored: list[str] = []
    tmp = Path(tempfile.mkdtemp(prefix="localgate-rej-"))
    try:
        for p in art_dir.iterdir():
            if p.is_file() and p.suffix in (".log", ".txt"):
                shutil.copy2(p, tmp / p.name)
        for label, bad_bench in cases.items():
            (tmp / "benchmark.json").write_text(json.dumps(bad_bench), encoding="utf-8")
            m = load_module(arm.scorer_module)
            try:
                if arm.scorer_arm_flag is not None:
                    bench, _ = _quiet(m._load_benchmark, tmp)
                    _quiet(m.certify, tmp, arm.scorer_arm_flag)
                    _quiet(m._lc_actions_score, bench)
                else:
                    res, _ = _quiet(m.score, tmp)
                    if str(res.get("verdict")) != "INFRA DEATH":
                        scored.append(f"{label}: verdict={res.get('verdict')!r} "
                                      f"lc={res.get('lc_total')!r}")
                        continue
                    else:
                        continue
            except ArtifactError:
                continue                       # loud refusal: correct
            except Exception:
                continue
            scored.append(f"{label}: certification PASSED on a malformed artifact")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return scored


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    return a == b


def check_readers(rep: Report, arm: Arm, corpus: list[Artifact], fast: bool) -> None:
    # R0 corpus
    if not corpus:
        rep.bad("R", "R0.corpus", "no REAL benchmark artifacts found on disk -- "
                                  "reader certification is impossible")
        return
    healthy = [a for a in corpus if not a.degenerate]
    degenerate = [a for a in corpus if a.degenerate]
    n_pull = sum(1 for a in corpus if "kernel_pulls" in a.path.as_posix())
    rep.ok("R", "R0.corpus",
           f"{len(corpus)} real artifacts indexed ({n_pull} pulled kernel outputs, "
           f"{len(healthy)} contract-clean, {len(degenerate)} degenerate/aborted), "
           f"{sum(a.n_games for a in corpus)} game_runs total")

    readers = reader_registry()

    # R1 direct readers x EVERY contract-clean real artifact, vs the oracle
    direct = [r for r in readers if r.kind == "direct"]
    sample = healthy if not fast else healthy[:max(10, len(ARM_ARTIFACTS))]
    for r in direct:
        bad: list[str] = []
        n = 0
        for art in sample:
            bench = json.loads(art.path.read_text(encoding="utf-8"))
            truth = oracle(bench)
            try:
                got = r.fn(bench)
            except ArtifactError as e:
                bad.append(f"{art.rel}: reader REFUSED a healthy artifact -- {e}")
                continue
            n += 1
            for k in r.provides:
                if k not in got:
                    bad.append(f"{art.rel}: reader omitted {k}")
                elif not _close(got[k], truth[k]):
                    bad.append(f"{art.rel}: {k} reader={got[k]!r} oracle={truth[k]!r}")
                elif truth[k] and not got[k]:
                    bad.append(f"{art.rel}: {k} SILENT ZERO on a non-zero artifact")
        if bad:
            rep.bad("R", f"R1.{r.name}",
                    f"reader disagrees with the direct computation on "
                    f"{len(set(x.split(':')[0] for x in bad))} real artifact(s)",
                    "\n".join(bad[:12]))
        else:
            rep.ok("R", f"R1.{r.name}",
                   f"matches the independent computation on {n}/{len(sample)} real "
                   f"artifacts ({', '.join(r.provides)})")

    # R1b degenerate real artifacts: a reader must NOT report confident numbers
    # from a run that never finished (null final_score, empty levels, ...).
    if degenerate:
        for r in direct:
            confident = []
            for art in degenerate:
                bench = json.loads(art.path.read_text(encoding="utf-8"))
                try:
                    got = r.fn(bench)
                except ArtifactError:
                    continue                    # refused: correct
                if any(got.get(k) for k in r.provides):
                    confident.append(f"{art.rel}: returned {got} "
                                     f"(oracle: {art.oracle_error})")
            if confident:
                rep.bad("R", f"R1b.{r.name}",
                        f"{len(confident)} DEGENERATE artifact(s) produced confident "
                        f"numbers instead of a refusal", "\n".join(confident[:6]))
            else:
                rep.ok("R", f"R1b.{r.name}",
                       f"refuses all {len(degenerate)} degenerate real artifacts "
                       f"({', '.join(a.run_dir.name for a in degenerate[:3])}...)")

    # R2 pipeline readers on their OWN arm's real pull directory.
    # A reader belonging to ANOTHER lane is still exercised (it is nearly free,
    # and a broken campaign scorer should be visible from anywhere), but it is
    # reported as a WARN: lane X must not be blocked from pushing because lane
    # Y's instrument is broken. Only the arm under test can FAIL this gate.
    for r in [x for x in readers if x.kind == "pipeline"]:
        mine = (r.arm == arm.name) or (r.arm in arm.sibling_arms)
        note = "" if mine else "OTHER LANE -- advisory, does not block this arm: "
        bad = rep.bad if mine else rep.warn
        art_dir = ARMS[r.arm].artifact
        bench_p = next(iter(art_dir.rglob("benchmark.json")), None)
        if bench_p is None:
            rep.skip("R", f"R2.{r.arm}", f"no benchmark.json under {art_dir.name}")
            continue
        truth = oracle(json.loads(bench_p.read_text(encoding="utf-8")))
        try:
            got = r.fn(art_dir)
        except ArtifactError as e:
            bad("R", f"R2.{r.arm}",
                f"{note}{r.name} could not read its OWN arm's real artifact", str(e))
            continue
        except Exception as e:
            bad("R", f"R2.{r.arm}", f"{note}{r.name} raised {type(e).__name__}",
                traceback.format_exc(limit=3))
            continue
        got_metrics = {k: v for k, v in got.items() if not k.startswith("_")}
        if not got_metrics:
            bad("R", f"R2.{r.arm}",
                f"{note}{r.name} returned NO metrics for its own arm "
                f"(verdict={got.get('_verdict')!r})",
                f"reason: {got.get('_reason','')}\n"
                f"oracle: {truth}")
            continue
        mism = [f"{k}: scorer={v!r} oracle={truth[k]!r}"
                for k, v in got_metrics.items()
                if k in truth and not _close(v, truth[k])]
        zero = [k for k, v in got_metrics.items()
                if k in truth and truth[k] and not v]
        if mism or zero:
            bad("R", f"R2.{r.arm}",
                f"{note}{r.name} disagrees with the direct computation on its own artifact",
                "\n".join(mism + [f"{k}: SILENT ZERO" for k in zero]))
        else:
            rep.ok("R", f"R2.{r.arm}",
                   f"{r.name} on {art_dir.name}: "
                   + ", ".join(f"{k}={got_metrics[k]}" for k in
                               ("n_games", "lc_total", "total_actions", "mean_score")
                               if k in got_metrics))

    # R3 reject-tests at the PIPELINE level: a malformed benchmark.json dropped
    # beside the arm's OWN real log must be REFUSED, never scored. This is the
    # level at which a bad artifact would actually reach a verdict.
    cases = malformed_cases()
    for arm_name, art_dir in sorted(ARM_ARTIFACTS.items()):
        mine = (arm_name == arm.name) or (arm_name in arm.sibling_arms)
        rej = rep.bad if mine else rep.warn
        scored = _pipeline_reject_test(ARMS[arm_name], art_dir, cases)
        if scored:
            rej("R", f"R3.{arm_name}",
                f"{'' if mine else 'OTHER LANE -- advisory: '}"
                f"{len(scored)} malformed artifact(s) were SCORED instead of refused "
                f"-- a corrupt pull could reach a verdict",
                "\n".join(scored[:8]))
        else:
            rep.ok("R", f"R3.{arm_name}",
                   f"refuses all {len(cases)} malformed benchmark shapes placed beside "
                   f"its own real log (incl. the D1/D2/D3 key shapes)")

    # R3r advisory: the same cases against the bare reader function. Tolerant
    # key-fallbacks are recorded, not failed -- the pipeline (R3) is the gate.
    for r in direct:
        silent, tolerant = [], []
        for label, bad_bench in cases.items():
            try:
                oracle(bad_bench)
                silent.append(f"ORACLE accepted {label} -- the oracle is too lax")
                continue
            except Exception:
                pass
            try:
                got = r.fn(bad_bench)
            except Exception:
                continue                        # loud refusal: correct
            vals = {k: v for k, v in got.items() if k in ORACLE_METRICS}
            if vals and all(not v for v in vals.values()):
                silent.append(f"{label}: all-zero {vals}")
            elif any(not v for v in vals.values()):
                silent.append(f"{label}: PARTIAL silent zero {vals} "
                              f"(the D1/D2/D3 signature)")
            elif vals:
                tolerant.append(f"{label}: tolerated via key fallback -> {vals}")
        if silent:
            rep.warn("R", f"R3r.{r.name}",
                     f"{len(silent)} malformed shape(s) silently zero this reader "
                     f"(guarded upstream by the pipeline's n_games gate -- see R3)",
                     "\n".join(silent[:8] + tolerant[:4]))
        else:
            rep.ok("R", f"R3r.{r.name}",
                   f"no silent zeros across {len(cases)} malformed shapes"
                   + (f"; {len(tolerant)} tolerated by declared key fallbacks"
                      if tolerant else ""))

    # R4 historical-defect regression: the known-bad readers must be CAUGHT
    ref = next((a for a in corpus if a.producer in ARM_ARTIFACTS), corpus[0])
    bench = json.loads(ref.path.read_text(encoding="utf-8"))
    truth = oracle(bench)
    escaped: list[str] = []
    for label, fn in defect_readers().items():
        got = fn(bench)
        caught = any((k in truth) and not _close(v, truth[k]) for k, v in got.items())
        if not caught:
            escaped.append(f"{label} agreed with the oracle on {ref.rel} -- "
                           f"the R1 cross-check would NOT have caught it")
    if escaped:
        rep.bad("R", "R4.historical", "known-bad readers slipped past the cross-check",
                "\n".join(escaped))
    else:
        rep.ok("R", "R4.historical",
               f"D1(games)/D2(total_actions)/D3(score) all caught on {ref.rel} "
               f"(each returns 0 where the oracle returns non-zero)")


# ===========================================================================
# 6. CHECK GROUP N — notebook static gate
# ===========================================================================
def _cells(nb: dict) -> list[dict]:
    return nb.get("cells", [])


def _src(cell: dict) -> str:
    s = cell.get("source", "")
    return s if isinstance(s, str) else "".join(s)


def check_notebook(rep: Report, arm: Arm, nb_path: Path | None, fast: bool) -> None:
    nb_path = nb_path or arm.notebook
    if nb_path is None or not nb_path.is_file():
        rep.skip("N", "N0.notebook", f"no notebook for arm {arm.name}")
        return
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cells = _cells(nb)
    code = [(i, _src(c)) for i, c in enumerate(cells) if c.get("cell_type") == "code"]
    try:
        shown = nb_path.relative_to(REPO).as_posix()
    except ValueError:
        shown = nb_path.as_posix()
    rep.ok("N", "N0.notebook", f"{shown}  cells={len(cells)} code={len(code)}")

    # N1 every code cell compiles
    errs = []
    for i, s in code:
        try:
            ast.parse(s)
        except SyntaxError as e:
            errs.append(f"cell {i}: line {e.lineno}: {e.msg}")
    if errs:
        rep.bad("N", "N1.compile", f"{len(errs)} code cell(s) do not parse",
                "\n".join(errs[:8]))
    else:
        rep.ok("N", "N1.compile", f"all {len(code)} code cells ast.parse clean")

    if arm.expect_n_cells is not None:
        if len(cells) == arm.expect_n_cells:
            rep.ok("N", "N1b.cellcount", f"{len(cells)} cells as declared")
        else:
            rep.bad("N", "N1b.cellcount",
                    f"{len(cells)} cells, prereg declares {arm.expect_n_cells} (STRUCTURAL DRIFT)")

    # N2 forbidden tokens
    whole = "\n".join(s for _, s in code)
    hits = [t for t in arm.forbidden_tokens if t in whole]
    if hits:
        rep.bad("N", "N2.tokens", f"forbidden token(s) present for arm {arm.name}: {hits}")
    elif arm.forbidden_tokens:
        rep.ok("N", "N2.tokens",
               f"0 occurrences of {len(arm.forbidden_tokens)} forbidden tokens")
    else:
        rep.skip("N", "N2.tokens", "no forbidden-token list declared for this arm")

    # N3 flag literals match the intended arm
    if arm.required_literals:
        bad = []
        for name, want in arm.required_literals:
            m = re.search(rf"^{re.escape(name)}\s*=\s*(True|False)\b", whole, re.M)
            if m is None:
                bad.append(f"{name}: literal assignment not found")
            elif m.group(1) != want:
                bad.append(f"{name} = {m.group(1)}, arm {arm.name} requires {want}")
        if bad:
            rep.bad("N", "N3.flags",
                    "staged notebook is NOT the arm you asked for", "\n".join(bad))
        else:
            rep.ok("N", "N3.flags",
                   ", ".join(f"{n}={v}" for n, v in arm.required_literals))
    else:
        rep.skip("N", "N3.flags", "no flag literals declared for this arm")

    # N4 builder determinism (redirected: nothing under notebooks/ is touched)
    if arm.builder_path is None or not arm.builder_path.is_file():
        rep.skip("N", "N4.determinism", arm.note or "no builder registered for this arm")
    elif fast:
        rep.skip("N", "N4.determinism", "fast path (use --full)")
    else:
        check_builder_determinism(rep, arm)

    # N5/N6/N7/N8 -- composed from scripts/preflight.py, offline
    check_preflight(rep, arm, nb, nb_path)


_DRIVER = r'''
import json, sys, importlib.util
from pathlib import Path
builder, outroot, call, argsjson = sys.argv[1], Path(sys.argv[2]), sys.argv[3], sys.argv[4]
spec = importlib.util.spec_from_file_location("_bld", builder)
m = importlib.util.module_from_spec(spec)
sys.modules["_bld"] = m
sys.argv = [builder]
spec.loader.exec_module(m)      # safe: every builder guards build() behind __main__
# Redirect ONLY the OUT_* path constants -- inputs (BASE_NB_PATH, SRC_NB) are
# left alone so the builder still reads the real base notebook.
redirected = []
for name in dir(m):
    if not name.startswith("OUT"):
        continue
    v = getattr(m, name)
    if isinstance(v, Path):
        setattr(m, name, outroot / v.name)
        redirected.append(name)
outroot.mkdir(parents=True, exist_ok=True)
getattr(m, call)(*json.loads(argsjson))
print("REDIRECTED " + ",".join(sorted(redirected)))
'''


def _hash_tree(root: Path) -> dict[str, str]:
    out = {}
    if not root.is_dir():
        return out
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in p.as_posix():
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def check_builder_determinism(rep: Report, arm: Arm) -> None:
    tmp = Path(tempfile.mkdtemp(prefix="localgate-det-"))
    try:
        drv = tmp / "_driver.py"
        drv.write_text(_DRIVER, encoding="utf-8")
        digests = []
        for run in ("a", "b"):
            out = tmp / run
            r = subprocess.run(
                [PY, str(drv), str(arm.builder_path), str(out),
                 arm.builder_call, json.dumps(list(arm.builder_args))],
                capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(REPO), timeout=300,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONHASHSEED": "0"})
            if r.returncode != 0:
                rep.bad("N", "N4.determinism",
                        f"builder {arm.builder_path.name} failed (run {run})",
                        (r.stdout + r.stderr)[-1500:])
                return
            digests.append(_hash_tree(out))
        if not digests[0]:
            rep.bad("N", "N4.determinism",
                    "builder produced no output under the redirected OUT_* paths "
                    "(redirection may have missed the real write target)")
            return
        if digests[0] == digests[1]:
            rep.ok("N", "N4.determinism",
                   f"{arm.builder_path.name}{list(arm.builder_args)} rebuilt twice -> "
                   f"byte-identical ({len(digests[0])} file(s))")
        else:
            diff = sorted(set(digests[0]) ^ set(digests[1])) or \
                   [k for k in digests[0] if digests[0][k] != digests[1].get(k)]
            rep.bad("N", "N4.determinism",
                    "builder is NOT deterministic -- a rebuild would change pushed bytes",
                    "\n".join(diff[:10]))
    except subprocess.TimeoutExpired:
        rep.bad("N", "N4.determinism", "builder timed out (300 s)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_preflight(rep: Report, arm: Arm, nb: dict, nb_path: Path) -> None:
    """Compose scripts/preflight.py at the FUNCTION level so it runs OFFLINE.

    preflight's CLI is pull-based (it is the AT-PUSH gate). Its structural
    checks, its duck-family diff and its host gates are pure functions over a
    parsed notebook, so the pre-push gate calls those directly against the
    LOCAL staged bytes. Nothing is reimplemented here.
    """
    try:
        pf = load_module("preflight")
    except Exception as e:
        rep.warn("N", "N5.preflight", f"could not import scripts/preflight.py: {e}")
        return

    # N5 structural checks (K-series), duck-harness family
    try:
        checks = pf.structural_checks(nb, pf.FAMILY_DUCK)
    except Exception as e:
        rep.warn("N", "N5.preflight", f"structural_checks raised {type(e).__name__}: {e}")
        checks = []
    fails = [c for c in checks if c.get("status") == "FAIL"]
    warns = [c for c in checks if c.get("status") == "WARN"]
    if fails:
        rep.bad("N", "N5.preflight",
                f"preflight structural_checks: {len(fails)} FAIL",
                "\n".join(f"{c['check']}: {c['message']}" for c in fails[:8]))
    else:
        rep.ok("N", "N5.preflight",
               f"preflight structural_checks (duck-harness family): "
               f"{len(checks) - len(warns)} ok, {len(warns)} warn")

    # N6 diff vs base with the prereg's declared cell indices.
    # preflight's D3 demands identical cell SHAPE, so an arm that legitimately
    # INSERTS a cell is first reduced to its base shape by removing exactly the
    # declared insertions; what remains must then differ from base only at the
    # declared modified indices. Nothing about preflight is reimplemented.
    if arm.base_notebook is None or not arm.base_notebook.is_file():
        rep.skip("N", "N6.diff", "no local base notebook registered for this arm")
    else:
        tmpd = Path(tempfile.mkdtemp(prefix="localgate-pf-"))
        try:
            nb_cmp = nb
            if arm.expect_inserted_cells:
                nb_cmp = json.loads(json.dumps(nb))
                for i in sorted(arm.expect_inserted_cells, reverse=True):
                    if i < len(nb_cmp["cells"]):
                        nb_cmp["cells"].pop(i)
                rep.ok("N", "N6a.inserted",
                       f"declared insertions {list(arm.expect_inserted_cells)} removed "
                       f"-> {len(nb_cmp['cells'])} cells compared against base "
                       f"{arm.base_notebook.name}")
            base_nb = pf.load_baseline_notebook(str(arm.base_notebook), tmpd)
            # Translate the prereg's declared indices into preflight's D4
            # vocabulary: indices AFTER the declared insertions are removed,
            # and CODE cells only (D4 diffs code cells; a retitled markdown
            # cell is verified separately below).
            expect_code = None
            if arm.expect_diff_cells:
                ins = sorted(arm.expect_inserted_cells)
                shifted = [i - sum(1 for j in ins if j < i)
                           for i in sorted(arm.expect_diff_cells)]
                expect_code, expect_md = [], []
                for i in shifted:
                    if 0 <= i < len(nb_cmp["cells"]):
                        (expect_code if nb_cmp["cells"][i].get("cell_type") == "code"
                         else expect_md).append(i)
                if expect_md:
                    same = [i for i in expect_md
                            if i < len(base_nb.get("cells", []))
                            and _src(nb_cmp["cells"][i]) == _src(base_nb["cells"][i])]
                    if same:
                        rep.bad("N", "N6b.markdown",
                                f"declared-to-differ non-code cell(s) {same} are "
                                f"byte-identical to base -- the builder did not stamp them")
                    else:
                        rep.ok("N", "N6b.markdown",
                               f"declared non-code diff cell(s) {expect_md} do differ "
                               f"from base (D4 covers code cells only)")
            dchecks = pf.duck_diff_checks(
                arm.kernel or arm.name, nb_cmp, str(arm.base_notebook), base_nb,
                expect_code)
            dfails = [c for c in dchecks if c.get("status") == "FAIL"]
            msg = "; ".join(f"{c['check']}={c['status']}" for c in dchecks)
            if dfails:
                # D2 (metadata byte-identity vs base) is informational for arms
                # that intentionally retarget metadata; D3/D4 are hard.
                hard = [c for c in dfails if c["check"] in ("D1", "D3", "D4")]
                detail = "\n".join(f"{c['check']}: {c['message']}" for c in dfails)
                if hard:
                    rep.bad("N", "N6.diff",
                            f"diff-vs-base: {len(hard)} hard FAIL "
                            f"(expected diff cells {arm.expect_diff_cells})", detail)
                else:
                    rep.warn("N", "N6.diff",
                             "diff-vs-base: metadata differs from base (expected for a "
                             "retargeted arm; D3/D4 clean)", detail)
            else:
                rep.ok("N", "N6.diff", f"diff-vs-base clean -- {msg}")
        except Exception as e:
            rep.warn("N", "N6.diff", f"duck_diff_checks raised {type(e).__name__}: {e}")
        finally:
            shutil.rmtree(tmpd, ignore_errors=True)

    # N7 metadata.kaggle sanity + the pushed kernel-metadata.json
    km = (nb.get("metadata") or {}).get("kaggle") or {}
    meta_path = nb_path.parent / "kernel-metadata.json"
    problems, facts, soft = [], [], []
    if km:
        if km.get("accelerator"):
            facts.append(f"accelerator={km['accelerator']}")
        else:
            problems.append("metadata.kaggle.accelerator missing")
        # isGpuEnabled inside the notebook is the web-UI mirror; the field that
        # actually drives the push is kernel-metadata.json enable_gpu (checked
        # below) and preflight's H1. The certified field-eval vehicle carries
        # isGpuEnabled=False with accelerator set and ran on the RTX PRO 6000,
        # so a mismatch here is advisory, never a block.
        if km.get("isGpuEnabled") is not True and km.get("accelerator"):
            soft.append(f"nb metadata.kaggle.isGpuEnabled={km.get('isGpuEnabled')!r} "
                        f"while accelerator={km['accelerator']!r} "
                        f"(matches the certified base; enable_gpu is authoritative)")
        elif km.get("isGpuEnabled") is True:
            facts.append("isGpuEnabled=True")
        ds = km.get("dataSources") or []
        if not ds:
            problems.append("metadata.kaggle.dataSources is empty")
        else:
            facts.append(f"dataSources={len(ds)}")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("id", "kernel_type", "language", "enable_gpu"):
            if key not in meta:
                problems.append(f"kernel-metadata.json has no {key!r}")
        if meta.get("enable_gpu") is False:
            problems.append("kernel-metadata.json enable_gpu=false on a GPU arm")
        if not (meta.get("dataset_sources") or meta.get("model_sources")):
            problems.append("kernel-metadata.json has neither dataset_sources nor model_sources")
        else:
            facts.append(f"dataset_sources={len(meta.get('dataset_sources') or [])}"
                         f" model_sources={len(meta.get('model_sources') or [])}")
        if meta.get("enable_internet") is True:
            problems.append("kernel-metadata.json enable_internet=true "
                            "(the eval rail runs internet OFF)")
        if arm.kernel and meta.get("id") and meta["id"] != arm.kernel:
            problems.append(f"kernel-metadata.json id={meta['id']!r} "
                            f"but arm registry says {arm.kernel!r}")
    else:
        problems.append(f"no kernel-metadata.json beside {nb_path.name}")
    if problems:
        rep.bad("N", "N7.metadata", f"{len(problems)} metadata problem(s)",
                "\n".join(problems + soft))
    elif soft:
        rep.warn("N", "N7.metadata",
                 "; ".join(facts) + f"  [{len(soft)} advisory]", "\n".join(soft))
    else:
        rep.ok("N", "N7.metadata", "; ".join(facts) or "metadata present")

    # N8 host gates (warn-only, exactly as preflight runs them)
    try:
        kmeta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None
        hchecks = pf.host_gates(arm.kernel or arm.name, nb, kmeta, "warn")
        hfail = [c for c in hchecks if c.get("status") == "FAIL"]
        hwarn = [c for c in hchecks if c.get("status") == "WARN"]
        if hfail:
            rep.bad("N", "N8.hostgates", f"{len(hfail)} host gate FAIL",
                    "\n".join(f"{c['check']}: {c['message']}" for c in hfail))
        elif hwarn:
            rep.warn("N", "N8.hostgates", f"{len(hwarn)} host gate warn",
                     "\n".join(f"{c['check']}: {c['message']}" for c in hwarn[:6]))
        else:
            rep.ok("N", "N8.hostgates", f"H1-H4 clean ({len(hchecks)} checks)")
    except Exception as e:
        rep.warn("N", "N8.hostgates", f"host_gates raised {type(e).__name__}: {e}")


# ===========================================================================
# 7. CHECK GROUP H — harness smoke without the 27B
# ===========================================================================
# A stdlib HTTP server speaks the OpenAI-shaped /chat/completions the harness's
# `requests.post` expects (no litellm, no openai SDK -- the harness itself only
# ever uses raw `requests`). The game is the REAL taaf competition simulator
# (taaf.game_examples.ExampleGame) driven by the REAL HarnessSolver + ToolAgent
# out of the vendored 08-15 bundle. Everything is CPU and finishes in seconds.
HARNESS_SCENARIOS = ("native_tool_calls", "fenced_markup", "capture_contract",
                     "flags_off", "noop_guard")

_HARNESS_CHILD = r'''
import asyncio, json, os, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

cfg = json.loads(sys.argv[1])
BUNDLE = Path(cfg["bundle"])
sys.path.insert(0, str(BUNDLE))

scenario = cfg["scenario"]
LVL1 = 'action(["ACTION1","ACTION1","ACTION1"])\nprint("LEVEL1-SEQUENCE-SENT")'
LVL2 = 'action(["ACTION2","ACTION2","ACTION2"])\nprint("LEVEL2-SEQUENCE-SENT")'
NOOP = 'r1 = action("ACTION2")\nr2 = action("ACTION2")\nprint("NOOP-PROBE", r1, r2)'

if scenario == "noop_guard":
    SCRIPT = [NOOP, LVL1, LVL2]
elif scenario in ("capture_contract", "capture_hidden_only"):
    # turn 0 is content-only (no tool call); turn 1 must NOT complete a level,
    # because a level transition deliberately CLEARS the carried world model
    # (_update_summarized_knowledge_from_step_summary) -- the carry has to be
    # observed on a turn where the state block is rebuilt and the model stands.
    SCRIPT = ['action("ACTION1")',
              'action(["ACTION1","ACTION1"])',
              LVL2]
else:
    SCRIPT = [LVL1, LVL2]

# The capture contract (exp 17): the harness builds its carried world model
# ONLY from VISIBLE assistant `content`; `reasoning_content` is the hidden
# channel that 97.6% of a reasoning model's output routes into. Two distinct
# markers make the split measurable.
VISIBLE_MARK = "VISIBLECAPTUREMARKER7Q2"
HIDDEN_MARK = "HIDDENCHANNELMARKER9X4"
VISIBLE = "World model: " + VISIBLE_MARK + " -- each level is a fixed 3-press sequence."
HIDDEN = "World model: " + HIDDEN_MARK + " -- this arrived on the hidden channel."
CARRY_HEADER = "Working world model carried from earlier turns:"
seen = {"n": 0, "payloads": []}
lock = threading.Lock()

# NEGATIVE CONTROL for the H3 instrument: in this scenario EVERY turn sends
# the world-model line only on the hidden channel and a neutral visible text.
# If H3 still reports "captured", H3 is measuring the wrong thing.
HIDDEN_ONLY = scenario == "capture_hidden_only"
VIS_TEXT = "ok." if HIDDEN_ONLY else VISIBLE
REA_TEXT = VISIBLE if HIDDEN_ONLY else HIDDEN

def make_message(i):
    if scenario in ("capture_contract", "capture_hidden_only"):
        if i == 0:
            # No tool call at all: forces the harness down the
            # _update_summarized_knowledge_from_assistant path, then a followup.
            return {"role": "assistant", "content": VIS_TEXT,
                    "reasoning_content": REA_TEXT}
        code = SCRIPT[i - 1] if (i - 1) < len(SCRIPT) else 'print("IDLE")'
    else:
        code = SCRIPT[i] if i < len(SCRIPT) else 'print("IDLE")'
    if scenario == "fenced_markup":
        # No native tool_calls: the model emits the qwen3-coder markup the
        # harness must recover with _recover_tool_calls_from_markup.
        body = ("<tool_call><function=python><parameter=code>"
                + code + "</parameter></function></tool_call>")
        return {"role": "assistant", "content": VIS_TEXT + "\n" + body,
                "reasoning_content": REA_TEXT}
    return {"role": "assistant",
            "content": VIS_TEXT,
            "reasoning_content": REA_TEXT,
            "tool_calls": [{"id": "call_%d" % i, "type": "function",
                            "function": {"name": "python",
                                         "arguments": json.dumps({"code": code})}}]}

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        raw = self.rfile.read(n) or b"{}"
        with lock:
            i = seen["n"]; seen["n"] += 1
            try: seen["payloads"].append(json.loads(raw))
            except Exception: pass
        out = {"id": "x", "object": "chat.completion", "created": 0, "model": "stub",
               "choices": [{"index": 0, "message": make_message(i),
                            "finish_reason": "tool_calls"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:%d/v1" % srv.server_address[1]
os.environ["OPENAI_API_KEY"] = "stub"
os.environ["MULTIMODAL_CONTEXT"] = "0"

import taaf.benchmark, taaf.game_examples
from inference.framework.solver import HarnessSolver

anim = scenario != "flags_off"
noop = scenario != "flags_off"
job = Path(tempfile.mkdtemp(prefix="localgate-h-"))
solver = HarnessSolver(label="local-gate-stub", model="stub-model", concurrency=1,
                       max_actions_per_game=40, kaggle_enable_vllm=False,
                       start_local_server=False, analyzer_timeout=30.0,
                       animation_awareness=anim, animation_retrieval=False,
                       hard_noop_guard=noop, save_request_logs=True)
game = taaf.game_examples.ExampleGame(label="localgate_scripted")
bm = taaf.benchmark.Benchmark(label="local-gate-smoke", games=[game], solver=solver,
                              n_passes=1, job_dir=job, periodic_save_interval_s=1e9)
asyncio.run(bm.run())
srv.shutdown()

bench = json.loads((job / "benchmark.json").read_text(encoding="utf-8"))
run = bench["game_runs"][0]
flags = solver.effective_flags()
sent = seen["payloads"][0] if seen["payloads"] else {}

# Did the VISIBLE content reach the harness's carried world model, and did the
# HIDDEN channel stay out of it? Read the CARRIED BLOCK of later requests --
# the transcript is a diagnostic file and is the wrong instrument for this.
carry_blocks = []
for payload in seen["payloads"][1:]:
    for msg in (payload.get("messages") or []):
        c = msg.get("content")
        if not isinstance(c, str) or CARRY_HEADER not in c:
            continue
        start = c.index(CARRY_HEADER)
        end = c.find("- Revise any item above", start)
        carry_blocks.append(c[start:end if end > 0 else start + 2000])
result = {
    "scenario": scenario,
    "job_dir": str(job),
    "benchmark_path": str(job / "benchmark.json"),
    "n_requests": seen["n"],
    "state": run.get("state"),
    "levels_completed": run.get("levels_completed"),
    "actions_per_level": run.get("actions_per_level"),
    "final_score": run.get("final_score"),
    "bench_keys": sorted(bench.keys()),
    "run_keys": sorted(run.keys()),
    "effective_flags": flags,
    "tools_offered": [t.get("function", {}).get("name") for t in (sent.get("tools") or [])],
    "system_prompt_len": len((sent.get("messages") or [{}])[0].get("content") or ""),
    "carry_blocks": len(carry_blocks),
    "visible_captured": any(VISIBLE_MARK in b for b in carry_blocks),
    "hidden_leaked": any(HIDDEN_MARK in b for b in carry_blocks),
}
print("LOCALGATE_RESULT " + json.dumps(result))
'''


def _run_harness_scenario(scenario: str, timeout: int = 240) -> dict:
    bundle = DUCK / "private" / "bundle_20260815" / "src" / "ARC3-Inference"
    child_dir = Path(tempfile.mkdtemp(prefix="localgate-hchild-"))
    child = child_dir / "child.py"
    child.write_text(_HARNESS_CHILD, encoding="utf-8")
    cfg = json.dumps({"bundle": str(bundle), "scenario": scenario})
    try:
        r = subprocess.run([PY, str(child), cfg], capture_output=True, text=True, encoding="utf-8", errors="replace",
                           cwd=str(REPO), timeout=timeout,
                           env={**os.environ, "PYTHONUTF8": "1",
                                "PYTHONIOENCODING": "utf-8"})
    finally:
        shutil.rmtree(child_dir, ignore_errors=True)
    for line in (r.stdout or "").splitlines():
        if line.startswith("LOCALGATE_RESULT "):
            return json.loads(line[len("LOCALGATE_RESULT "):])
    raise RuntimeError(f"harness child produced no result (rc={r.returncode})\n"
                       f"{(r.stdout + r.stderr)[-2000:]}")


def check_harness(rep: Report, arm: Arm) -> None:
    bundle = DUCK / "private" / "bundle_20260815" / "src" / "ARC3-Inference"
    if not (bundle / "inference" / "framework" / "solver.py").is_file():
        rep.skip("H", "H0.bundle", f"vendored bundle not found at {bundle}")
        return
    try:
        import taaf.game_examples  # noqa: F401
        import arcengine           # noqa: F401
    except Exception as e:
        rep.bad("H", "H0.bundle",
                f"competition wheels not importable in this interpreter: {e}",
                "run through the repo venv: uv run python scripts/local_gate.py ...")
        return
    rep.ok("H", "H0.bundle", "vendored 08-15 bundle + real arc_agi/arcengine/taaf wheels present")

    results: dict[str, dict] = {}
    for sc in HARNESS_SCENARIOS:
        try:
            results[sc] = _run_harness_scenario(sc)
        except Exception as e:
            rep.bad("H", f"H*.{sc}", f"scenario crashed: {type(e).__name__}", str(e)[:1500])
    if "native_tool_calls" not in results:
        return

    base = results["native_tool_calls"]

    # H1 the agent can actually complete a scripted game
    if base["state"] == "won" and base["levels_completed"] == 2 \
            and sum(base["actions_per_level"]) == 6:
        rep.ok("H", "H1.agent_loop",
               f"scripted game WON via native tool_calls: levels=2/2 actions=6 "
               f"score={base['final_score']:.1f} in {base['n_requests']} LLM round-trips")
    else:
        rep.bad("H", "H1.agent_loop",
                "agent did not complete the scripted game",
                json.dumps({k: base[k] for k in
                            ("state", "levels_completed", "actions_per_level",
                             "n_requests")}, indent=1))

    # H1b the request the harness actually sends is well-formed
    if base["tools_offered"] == ["python"] and base["system_prompt_len"] > 500:
        rep.ok("H", "H1b.request",
               f"outbound payload carries the python tool and a "
               f"{base['system_prompt_len']}-char system prompt")
    else:
        rep.bad("H", "H1b.request", "outbound chat payload is not the expected shape",
                json.dumps({k: base[k] for k in ("tools_offered", "system_prompt_len")}))

    # H2 the text tool-call parser (no native tool_calls field)
    fm = results.get("fenced_markup")
    if fm is None:
        pass
    elif fm["state"] == "won" and fm["levels_completed"] == 2:
        rep.ok("H", "H2.toolcall_parser",
               "fenced <tool_call><function=python> markup recovered and executed "
               "(the parser path, exercised with zero native tool_calls)")
    else:
        rep.bad("H", "H2.toolcall_parser",
                "the harness could not recover a fenced markup tool call",
                json.dumps({k: fm[k] for k in ("state", "levels_completed", "n_requests")}))

    # H3 visible-capture contract: the mechanism edge-2 exists to exploit.
    # A 'World model:' line on the VISIBLE channel must reappear in the carried
    # world-model block of a later request; the identical line delivered on
    # reasoning_content must NOT.
    cc = results.get("capture_contract")
    if cc is None:
        pass
    elif cc["carry_blocks"] == 0:
        rep.bad("H", "H3.capture_contract",
                "the harness never carried a world-model block forward -- the capture "
                "path did not run at all, so a capture-side arm would measure nothing",
                json.dumps({k: cc[k] for k in ("n_requests", "state", "carry_blocks")}))
    elif cc["visible_captured"] and not cc["hidden_leaked"]:
        rep.ok("H", "H3.capture_contract",
               f"visible-only capture CONFIRMED on the real agent: the visible "
               f"'World model:' line is carried into {cc['carry_blocks']} later "
               f"request(s), the identical line sent on reasoning_content is NOT "
               f"(the exp-17 mechanism edge-2 targets)")
    elif not cc["visible_captured"]:
        rep.bad("H", "H3.capture_contract",
                "visible assistant text was NOT captured into the carried world model "
                "-- the capture path is broken; every capture-side arm measures nothing")
    else:
        rep.bad("H", "H3.capture_contract",
                "hidden reasoning_content LEAKED into the carried world model -- the "
                "premise of the capture-contract arm does not hold on these bytes")

    # H4 flag paths change behaviour in the expected direction
    off = results.get("flags_off")
    if off is None:
        pass
    else:
        on_f, off_f = base["effective_flags"], off["effective_flags"]
        changed = [k for k in on_f if on_f[k] != off_f.get(k)]
        want = {"ARC3_ANIMATION_AWARENESS", "ARC3_HARD_NOOP_GUARD"}
        if want.issubset(set(changed)):
            rep.ok("H", "H4.flag_paths",
                   f"flags propagate end-to-end: {sorted(changed)} differ between the "
                   f"ON and OFF runs, and both runs still complete the game "
                   f"(off: state={off['state']} lc={off['levels_completed']})")
        else:
            rep.bad("H", "H4.flag_paths",
                    "a behaviour flag did NOT change the resolved configuration "
                    "(a flag that cannot fire is an arm that cannot be measured)",
                    f"changed={sorted(changed)} expected superset of {sorted(want)}")

    # H5 the locally produced artifact is REAL-SHAPED and readable by the arm's reader
    #    (this closes the loop: ground truth generated by the REAL library, not by us)
    try:
        bench = json.loads(Path(base["benchmark_path"]).read_text(encoding="utf-8"))
        truth = oracle(bench)
        got = _direct_private(bench)
        if all(_close(got[k], truth[k]) for k in got):
            rep.ok("H", "H5.loop_closed",
                   f"artifact emitted by the REAL taaf Benchmark "
                   f"(keys: {', '.join(base['run_keys'][:6])}...) reads correctly: "
                   f"lc={truth['lc_total']} actions={truth['total_actions']} "
                   f"score={truth['mean_score']:.1f}")
        else:
            rep.bad("H", "H5.loop_closed",
                    "reader disagrees with the oracle on a library-generated artifact",
                    f"reader={got} oracle={truth}")
    except Exception as e:
        rep.bad("H", "H5.loop_closed", f"library-generated artifact unreadable: {e}")

    # H6 noop guard direction
    ng = results.get("noop_guard")
    if ng is not None:
        if ng["state"] == "won":
            rep.ok("H", "H6.noop_guard",
                   "repeat-of-a-known-noop probe ran with the guard armed and the game "
                   "still completed (guard blocks the repeat, not the run)")
        else:
            rep.warn("H", "H6.noop_guard",
                     f"noop probe run ended {ng['state']} lc={ng['levels_completed']} "
                     f"-- inspect if the guard is expected to be transparent here")

    for r in results.values():
        shutil.rmtree(r.get("job_dir", ""), ignore_errors=True)


# ===========================================================================
# 8. CHECK GROUP A — arm matrix / cross-arm negative controls
# ===========================================================================
def _certifies(arm: Arm, run_dir: Path) -> tuple[bool, str]:
    """Does this arm's certifier accept that run directory?"""
    m = load_module(arm.scorer_module)
    try:
        if arm.scorer_arm_flag is not None:
            _quiet(m.certify, run_dir, arm.scorer_arm_flag)
            return True, "certified"
        res, _ = _quiet(m.score, run_dir)
        v = str(res.get("verdict", ""))
        return (v != "INFRA DEATH"), f"verdict={v}: {res.get('reason','')[:160]}"
    except ArtifactError as e:
        return False, str(e)[:200]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"[:200]


def check_arm_matrix(rep: Report, arm: Arm) -> None:
    if not ARM_ARTIFACTS:
        rep.skip("A", "A0.matrix", "no real per-arm artifacts on disk")
        return

    own = arm.artifact if (arm.artifact and arm.artifact.is_dir()) else None
    # A1 own-arm markers present on its OWN real artifact
    if own is None:
        rep.skip("A", f"A1.{arm.name}",
                 "this arm has no pulled artifact yet -- own-arm certification "
                 "cannot be exercised against real data (it will be at pull time)")
    else:
        ok, why = _certifies(arm, own)
        if ok:
            rep.ok("A", f"A1.{arm.name}",
                   f"own certification ACCEPTS its real artifact {own.name}")
        else:
            rep.bad("A", f"A1.{arm.name}",
                    f"own certification REFUSES its own real artifact {own.name} "
                    f"-- the instrument would void a healthy arm", why)

    # A2 cross-arm negative controls against every OTHER arm's real artifact
    accepted, refused = [], []
    for other, other_dir in sorted(ARM_ARTIFACTS.items()):
        if own is not None and other_dir == own:
            continue
        if other == arm.name or other in arm.sibling_arms:
            continue
        ok, why = _certifies(arm, other_dir)
        (accepted if ok else refused).append(f"{other}({other_dir.name}): {why}")
    if arm.sibling_arms:
        rep.ok("A", f"A2s.{arm.name}",
               f"sibling arm(s) {list(arm.sibling_arms)} excluded from the negative "
               f"controls: same vehicle bytes, so mutual acceptance is correct")
    if not (accepted or refused):
        rep.skip("A", f"A2.{arm.name}", "no foreign artifacts to test against")
    elif accepted:
        rep.bad("A", f"A2.{arm.name}",
                f"{len(accepted)} foreign artifact(s) ACCEPTED by this arm's certification "
                f"-- the arm markers do not discriminate",
                "\n".join(accepted))
    else:
        rep.ok("A", f"A2.{arm.name}",
               f"all {len(refused)} foreign real artifacts REFUSED "
               f"({', '.join(x.split('(')[0] for x in refused)})")


# ===========================================================================
# 8b. CHECK GROUP P — P0 PERMANENT INSTRUMENTS
#
# ARM P0 (perturn_program_2026-08-22.md §5.3) left one item open: "delivery
# instruments compile into local_gate as gate-group checks". This group is
# that compile. It is a REGRESSION harness for two instruments the campaign
# now depends on:
#
#   * the no-op guard's *behaviour* (P0.3): the shipped guard cannot fire
#     under a ticking HUD, through two independent defeat paths, and the
#     interior re-key arms it with zero false blocks. If a future bundle
#     changes `board_signature`/`observe`, these checks move.
#   * the CADENCE instrument (duck_eval/cadence/cadence_instrument.py): the
#     tokens-per-acting-turn / acting-turns-per-game reader that the cadence
#     arm's delivery verdict is read from. It is validated here against the
#     REAL artifacts on disk, before the arm's data lands
#     (feedback_audit_the_instrument).
#
# Everything runs on synthetic grids in-process plus the sealed P0 evidence
# files; no GPU, no network, no writes.
# ===========================================================================
P0_DIR = DUCK / "p0"
CADENCE_DIR = DUCK / "cadence"


def _p0_modules():
    """(noop_guard module, board_signature_fix module) from the vendored bundle."""
    bundle = DUCK / "private" / "bundle_20260815" / "src" / "ARC3-Inference"
    for d in (str(bundle), str(P0_DIR)):
        if d not in sys.path:
            sys.path.insert(0, d)
    ng = importlib.import_module("inference.agent.noop_guard")
    fx = importlib.import_module("board_signature_fix")
    return ng, fx


def _hud_grid(n: int, *, interior_fill: int = 0, tick: int = 0, size: int = 8):
    """A grid whose LAST row is a ticking HUD strip and whose interior is
    constant unless ``interior_fill`` says otherwise."""
    g = [[interior_fill for _ in range(size)] for _ in range(size - 1)]
    # a real HUD/timer strip ticks MONOTONICALLY -- it does not cycle back to a
    # value it already had inside a run, which is exactly why the full-grid key
    # never recurs.  No modulo here: a wrapping strip would let the shipped key
    # recur by accident and would make P2 a test of the fixture, not the guard.
    g.append([tick * 64 + c for c in range(size)])   # row size-1 ticks
    if n:
        g[0][0] = n % 7                                # optional interior change
    return g


def check_p0_instruments(rep: Report) -> None:
    try:
        ng, fx = _p0_modules()
    except Exception as e:
        rep.bad("P", "P0.import",
                f"cannot import the shipped guard / the P0 fix: {type(e).__name__}: {e}",
                "expected duck_eval/private/bundle_20260815/.../noop_guard.py and "
                "duck_eval/p0/board_signature_fix.py")
        return
    rep.ok("P", "P0.import",
           "shipped NoopGuard + duck_eval/p0/board_signature_fix loaded")

    ACT = "ACTION1"
    HUD_ROWS, HUD_COLS = {7}, set()

    # ---- P1  defeat path 1: the RECORD path -------------------------------
    # A true interior no-op with a ticking HUD looks "changed" to a full-grid
    # compare, so the guard never records it and can never block it.
    guard = ng.NoopGuard()
    blocked_record = 0
    for t in range(12):
        before, after = _hud_grid(0, tick=t), _hud_grid(0, tick=t + 1)
        sig = ng.board_signature(before)
        if guard.is_known_noop(0, sig, ACT):
            blocked_record += 1
        guard.observe(level=0, board_before_sig=sig, action_sig=ACT,
                      board_changed=(ng.board_signature(before)
                                     != ng.board_signature(after)))
    if blocked_record == 0:
        rep.ok("P", "P1.guard_record_path",
               "shipped guard blocks 0/12 identical interior no-ops under a ticking "
               "HUD -- full-grid board_changed hides every no-op (P0.3 defeat path 1, "
               "reproduces the field's 0-in-1630)")
    else:
        rep.bad("P", "P1.guard_record_path",
                f"the shipped guard blocked {blocked_record}/12 -- P0.3's record-path "
                f"finding no longer reproduces; the bundle's guard changed and every "
                f"downstream no-op number is stale")

    # ---- P2  defeat path 2: the MATCH path --------------------------------
    # Even when recording is forced (board_changed=False), the full-grid key
    # is renewed by the HUD tick, so the memo can never match.
    guard = ng.NoopGuard()
    blocked_match = 0
    for t in range(12):
        before = _hud_grid(0, tick=t)
        sig = ng.board_signature(before)
        if guard.is_known_noop(0, sig, ACT):
            blocked_match += 1
        guard.observe(level=0, board_before_sig=sig, action_sig=ACT,
                      board_changed=False)
    if blocked_match == 0:
        rep.ok("P", "P2.guard_match_path",
               "with recording FORCED, the shipped guard still matches 0/12 -- the "
               "full-grid key never recurs (P0.3 defeat path 2: recorded 74, matched 0 "
               "on the real field stream)")
    else:
        rep.bad("P", "P2.guard_match_path",
                f"the full-grid key recurred and blocked {blocked_match}/12 -- P0.3's "
                f"match-path finding no longer reproduces")

    # ---- P3  the interior re-key ARMS the guard ---------------------------
    guard = ng.NoopGuard()
    blocked_interior = 0
    for t in range(12):
        before, after = _hud_grid(0, tick=t), _hud_grid(0, tick=t + 1)
        sig = fx.interior_signature(before, HUD_ROWS, HUD_COLS)
        if guard.is_known_noop(0, sig, ACT):
            blocked_interior += 1
        guard.observe(level=0, board_before_sig=sig, action_sig=ACT,
                      board_changed=fx.interior_changed(before, after,
                                                        HUD_ROWS, HUD_COLS))
    if blocked_interior >= 11:
        rep.ok("P", "P3.interior_key_fires",
               f"interior-keyed guard blocks {blocked_interior}/12 repeats of the same "
               f"interior no-op (22 blocks on the real 1,630-action field stream)")
    else:
        rep.bad("P", "P3.interior_key_fires",
                f"the interior re-key fired only {blocked_interior}/12 -- the P0.4 fix "
                f"no longer arms the guard, so the P3 rider is unmeasurable")

    # ---- P4  ...with ZERO false blocks on a genuinely changing interior ----
    guard = ng.NoopGuard()
    false_blocks = 0
    for t in range(12):
        before, after = _hud_grid(t, tick=t), _hud_grid(t + 1, tick=t + 1)
        sig = fx.interior_signature(before, HUD_ROWS, HUD_COLS)
        if guard.is_known_noop(0, sig, ACT):
            false_blocks += 1
        guard.observe(level=0, board_before_sig=sig, action_sig=ACT,
                      board_changed=fx.interior_changed(before, after,
                                                        HUD_ROWS, HUD_COLS))
    if false_blocks == 0:
        rep.ok("P", "P4.no_false_blocks",
               "interior-keyed guard blocks 0/12 actions that really do change the "
               "interior (0 false blocks measured on all 1,630 real field actions)")
    else:
        rep.bad("P", "P4.no_false_blocks",
                f"{false_blocks}/12 real interior changes were blocked -- the repaired "
                f"guard would suppress working actions")

    # ---- P5  the animation exemption survives the re-key ------------------
    guard = ng.NoopGuard()
    before = _hud_grid(0, tick=0)
    sig = fx.interior_signature(before, HUD_ROWS, HUD_COLS)
    guard.observe(level=0, board_before_sig=sig, action_sig=ACT,
                  board_changed=False, animated=True)
    if not guard.is_known_noop(0, sig, ACT):
        rep.ok("P", "P5.animation_exempt",
               "an ANIMATED action with an identical before/after interior is still "
               "not recorded as a no-op (the ft09/sb26 regression the docstring warns "
               "about stays fixed)")
    else:
        rep.bad("P", "P5.animation_exempt",
                "an animated action was recorded as a no-op -- this is the exact "
                "regression that hard-blocked working actions on the animation games")

    # ---- P6  HUD detection: finds the strip, ignores mid-board activity ----
    try:
        mask = fx.HudMask()
        for t in range(14):
            mask.observe(_hud_grid(0, tick=t), _hud_grid(0, tick=t + 1))
        found_border = mask.exclude_rows == {7}
        mid = fx.HudMask()
        for t in range(14):
            a = [[0] * 8 for _ in range(8)]
            b = [[0] * 8 for _ in range(8)]
            b[4][4] = t % 5                       # interior, not border-flush
            mid.observe(a, b)
        ignores_mid = not mid.exclude_rows and not mid.exclude_cols
        cold = fx.HudMask()
        cold.observe(_hud_grid(0, tick=0), _hud_grid(0, tick=1))
        degrades = (cold.signature(_hud_grid(0, tick=3))
                    == ng.board_signature(_hud_grid(0, tick=3)))
        if found_border and ignores_mid and degrades:
            rep.ok("P", "P6.hudmask",
                   "HudMask finds the ticking border strip (row 7), ignores mid-board "
                   "activity, and before convergence returns signatures byte-identical "
                   "to the shipped board_signature (safe default)")
        else:
            rep.bad("P", "P6.hudmask",
                    "HudMask behaviour regressed",
                    f"found_border={found_border} ignores_mid={ignores_mid} "
                    f"degrades_to_shipped={degrades}")
    except Exception as e:
        rep.bad("P", "P6.hudmask", f"HudMask raised: {type(e).__name__}: {e}")

    # ---- P7  the sealed P0 evidence is present and self-consistent --------
    ev = P0_DIR / "p0_noop_results.json"
    rs = P0_DIR / "p0_reset_results.json"
    try:
        agg = json.loads(ev.read_text(encoding="utf-8"))["aggregate"]
        per = json.loads(ev.read_text(encoding="utf-8"))["per_game"]
        want = {"actions": 1630, "blocked_original": 0, "blocked_interior": 22,
                "interior_noop": 143, "hud_masked_noop": 69, "full_noop": 74}
        bad = {k: (agg.get(k), v) for k, v in want.items() if agg.get(k) != v}
        sums_ok = sum(g["blocked_interior"] for g in per) == agg["blocked_interior"] \
            and sum(g["n_actions"] for g in per) == agg["actions"]
        if not bad and sums_ok and len(per) == 25:
            rep.ok("P", "P7.p0_evidence",
                   f"sealed P0.2/P0.3 evidence intact: {agg['actions']} real actions, "
                   f"shipped guard {agg['blocked_original']} blocks, interior "
                   f"{agg['blocked_interior']}, interior no-op rate "
                   f"{100*agg['interior_noop']/agg['actions']:.1f}%, per-game sums "
                   f"reconcile over 25 games")
        else:
            rep.bad("P", "P7.p0_evidence",
                    "the sealed P0 evidence file no longer matches P0_FINDINGS",
                    f"mismatches={bad} per_game_sums_ok={sums_ok} n_games={len(per)}")
    except Exception as e:
        rep.bad("P", "P7.p0_evidence", f"{ev.name} unreadable: {type(e).__name__}: {e}")

    # ---- P8  RESET semantics evidence (the gate P2 is built on) -----------
    try:
        r = json.loads(rs.read_text(encoding="utf-8"))
        a, b_, d = r["case_A"], r["case_B"], r["case_D"]
        ok = (a["level_preserved"] and a["score_preserved"]
              and a["frame_equals_level1_start"] and not a["full_reset_flag"]
              and d["frame_equals_level1_start"] and b_["full_reset_flag"])
        if ok:
            rep.ok("P", "P8.reset_semantics",
                   "P0.1 evidence intact: with ONLY_RESET_LEVELS=true a mid-level RESET "
                   "returns the byte-equal level-start frame and keeps level+score; "
                   "with the flag unset it full-resets (the footgun) -- P2's premise "
                   "holds and its negative control fires")
        else:
            rep.bad("P", "P8.reset_semantics",
                    "the sealed RESET evidence no longer supports P2's premise",
                    json.dumps({"A": a, "B": b_, "D": d})[:600])
    except Exception as e:
        rep.bad("P", "P8.reset_semantics", f"{rs.name} unreadable: {type(e).__name__}: {e}")

    # ---- P9  the CADENCE instrument reproduces the BP35 diagnostic --------
    ci = CADENCE_DIR / "cadence_instrument.py"
    if not ci.is_file():
        rep.bad("P", "P9.cadence_instrument", f"{ci} missing -- the cadence arm's "
                                              f"delivery verdict has no reader")
    else:
        try:
            r = _run_suite((str(ci), "--validate", "--quiet"), timeout=300)
            tail = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
            last = tail[-1] if tail else ""
            if r.returncode == 0:
                rep.ok("P", "P9.cadence_instrument",
                       f"tokens-per-acting-turn / acting-turns-per-game re-derive the "
                       f"08-22 BP35 diagnostic from the real artifacts on disk -- {last[:90]}")
            else:
                rep.bad("P", "P9.cadence_instrument",
                        f"the cadence instrument no longer reproduces the diagnostic "
                        f"(rc={r.returncode})", "\n".join(tail[-12:]))
        except subprocess.TimeoutExpired:
            rep.bad("P", "P9.cadence_instrument", "validation timed out")


# ===========================================================================
# 9. CHECK GROUP X — wrapped suites + do-no-harm
# ===========================================================================
def _run_suite(argv: tuple, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run([PY, *argv], capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=str(REPO), timeout=timeout,
                          env={**os.environ, "PYTHONUTF8": "1"})


def check_suites(rep: Report, arm: Arm, fast: bool) -> None:
    # X1 every scorer's own selftest -- WRAPPED, not reimplemented
    scorers = sorted({(a.scorer_module, a.scorer_path) for a in ARMS.values()})
    for mod, path in scorers:
        if not path.is_file():
            continue
        try:
            r = _run_suite((str(path), "--selftest"), timeout=300)
        except subprocess.TimeoutExpired:
            rep.bad("X", f"X1.{mod}", "selftest timed out")
            continue
        tail = (r.stdout + r.stderr).strip().splitlines()
        summary = tail[-1] if tail else ""
        if r.returncode == 0:
            rep.ok("X", f"X1.{mod}", f"selftest OK -- {summary[:110]}")
        else:
            rep.bad("X", f"X1.{mod}", f"selftest FAILED (rc={r.returncode})",
                    "\n".join(tail[-12:]))

    # X2 arm-specific existing deep suites (private_smoke 37/37, graft smoke, ...)
    for label, argv in arm.extra_suites:
        if not Path(argv[0]).is_file():
            rep.skip("X", f"X2.{label}", f"{argv[0]} not present")
            continue
        if fast:
            rep.skip("X", f"X2.{label}", "fast path (use --full)")
            continue
        try:
            r = _run_suite(argv)
        except subprocess.TimeoutExpired:
            rep.bad("X", f"X2.{label}", "suite timed out (900 s)")
            continue
        tail = (r.stdout + r.stderr).strip().splitlines()
        if r.returncode == 0:
            rep.ok("X", f"X2.{label}", f"OK -- {(tail[-1] if tail else '')[:110]}")
        else:
            failing = [ln.strip() for ln in tail
                       if "FAIL" in ln or "Error" in ln or "Traceback" in ln]
            rep.bad("X", f"X2.{label}", f"FAILED (rc={r.returncode})",
                    "\n".join((failing or tail[-15:])[:12]))

    # X3 STANDING suites -- not arm-specific; they guard shared instruments that
    #    every arm's verdict now leans on (ARM P0 §5.3 item 4).
    standing = (("p0_noop_repro", ("-m", "pytest", "-q",
                                   str(P0_DIR / "test_noop_guard_repro.py"))),)
    for label, argv in standing:
        target = Path(argv[-1])
        if not target.is_file():
            rep.skip("X", f"X3.{label}", f"{target} not present")
            continue
        if fast:
            rep.skip("X", f"X3.{label}", "fast path (use --full)")
            continue
        try:
            r = _run_suite(argv, timeout=300)
        except subprocess.TimeoutExpired:
            rep.bad("X", f"X3.{label}", "suite timed out")
            continue
        tail = [ln for ln in (r.stdout + r.stderr).strip().splitlines() if ln.strip()]
        if r.returncode == 5:
            rep.warn("X", f"X3.{label}", "pytest collected no tests")
        elif r.returncode == 0:
            rep.ok("X", f"X3.{label}", f"OK -- {(tail[-1] if tail else '')[:110]}")
        elif r.returncode == 4 or any("No module named pytest" in ln for ln in tail):
            rep.skip("X", f"X3.{label}", "pytest not installed in this interpreter")
        else:
            rep.bad("X", f"X3.{label}", f"FAILED (rc={r.returncode})",
                    "\n".join(tail[-12:]))


def snapshot_protected() -> dict[str, str]:
    """Hashes of everything a lane could be harmed by. The gate must not move
    a single byte of it (edge-2 is mid-run; the graft lane owns the queue)."""
    snap: dict[str, str] = {}
    for rel in ("notebooks",):
        for k, v in _hash_tree(REPO / rel).items():
            snap[f"{rel}/{k}"] = v
    for extra in ("submission_queue.json", "runs/lane_locks.json"):
        p = REPO / extra
        if p.is_file():
            snap[extra] = hashlib.sha256(p.read_bytes()).hexdigest()
    return snap


def check_no_harm(rep: Report, before: dict[str, str]) -> None:
    after = snapshot_protected()
    moved = sorted(set(before) ^ set(after)) + \
            sorted(k for k in before if k in after and before[k] != after[k])
    if moved:
        rep.bad("X", "X4.no_harm",
                f"THE GATE MODIFIED {len(moved)} protected file(s) -- this is a bug in "
                f"the gate itself and may have disturbed a live lane",
                "\n".join(moved[:15]))
    else:
        rep.ok("X", "X4.no_harm",
               f"{len(before)} protected files under notebooks/ + queue + lane_locks "
               f"byte-unchanged (edge-2 staging untouched)")


# ===========================================================================
# 10. SELF-TEST — negative controls on this gate
# ===========================================================================
def _tmp_notebook_variant(src: Path, mutate: Callable[[dict], None]) -> Path:
    nb = json.loads(src.read_text(encoding="utf-8"))
    mutate(nb)
    d = Path(tempfile.mkdtemp(prefix="localgate-neg-"))
    shutil.copy2(src.parent / "kernel-metadata.json", d / "kernel-metadata.json")
    out = d / src.name
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


def _sub_report(arm: Arm, nb: Path) -> Report:
    r = Report(arm.name, "negative-control")
    check_notebook(r, arm, nb, fast=True)
    return r


def self_test() -> Report:
    rep = Report("SELF-TEST", "negative-controls")
    corpus = build_corpus()
    arm = ARMS["private-edge2"]
    nb_src = arm.notebook

    def expect_fail(code_prefix: str, sub: Report, label: str, why: str) -> None:
        hit = [c for c in sub.checks
               if c.code.startswith(code_prefix) and c.status == FAIL]
        if hit:
            rep.ok("S", label, f"{why} -> {hit[0].code} FAILs (as it must)")
        else:
            got = ", ".join(f"{c.code}={c.status}" for c in sub.checks)
            rep.bad("S", label,
                    f"THE GATE DID NOT CATCH IT: {why} did not produce a "
                    f"{code_prefix}* FAIL", got)

    # S1 syntax error injected into a code cell
    def break_syntax(nb: dict) -> None:
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                c["source"] = "def broken(:\n    pass\n"
                return
    expect_fail("N1.compile", _sub_report(arm, _tmp_notebook_variant(nb_src, break_syntax)),
                "S1.broken_notebook_syntax", "a code cell that does not parse")

    # S2 forbidden token injected
    def add_token(nb: dict) -> None:
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                c["source"] = "import litellm  # banned\n" + _src(c)
                return
    expect_fail("N2.tokens", _sub_report(arm, _tmp_notebook_variant(nb_src, add_token)),
                "S2.forbidden_token", "a banned import (litellm) in a code cell")

    # S3 wrong arm flag literal
    def flip_flag(nb: dict) -> None:
        for c in nb["cells"]:
            s = _src(c)
            if "PRIVATE_EDGE2_VISIBLE_CONTRACT" in s:
                c["source"] = re.sub(r"PRIVATE_EDGE2_VISIBLE_CONTRACT\s*=\s*True",
                                     "PRIVATE_EDGE2_VISIBLE_CONTRACT = False", s)
                return
    expect_fail("N3.flags", _sub_report(arm, _tmp_notebook_variant(nb_src, flip_flag)),
                "S3.wrong_arm_flag", "an artifact stamped for a different arm")

    # S4 cell dropped (structural drift)
    def drop_cell(nb: dict) -> None:
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code":
                nb["cells"].pop(i)
                return
    expect_fail("N1b.cellcount", _sub_report(arm, _tmp_notebook_variant(nb_src, drop_cell)),
                "S4.structural_drift", "a deleted code cell")

    # S5-S7 the three historical defects, run through the REAL R1 machinery
    ref = next((a for a in corpus if a.producer == "private-base"), corpus[0])
    bench = json.loads(ref.path.read_text(encoding="utf-8"))
    truth = oracle(bench)
    for label, fn in defect_readers().items():
        got = fn(bench)
        wrong = {k: (v, truth[k]) for k, v in got.items()
                 if k in truth and not _close(v, truth[k])}
        if wrong:
            shown = "; ".join(f"{k}: bad-reader={v[0]!r} truth={v[1]!r}"
                              for k, v in wrong.items())
            rep.ok("S", f"S5.{label}",
                   f"historical defect reproduced on the REAL artifact {ref.rel} "
                   f"and CAUGHT -- {shown}")
        else:
            rep.bad("S", f"S5.{label}",
                    f"the reconstructed defect agreed with the oracle on {ref.rel} "
                    f"-- the cross-check would not have caught it", json.dumps(got))

    # S8 a reader that silently returns zeros on malformed input must be caught
    def silent_zero_reader(bench: dict) -> dict:
        games = bench.get("game_runs") or []
        return {"lc_total": sum(int(g.get("levels_completed") or 0) for g in games),
                "total_actions": 0, "mean_score": 0.0}
    r = Reader("SYNTHETIC.silent_zero", "direct", silent_zero_reader,
               ("lc_total", "total_actions", "mean_score"))
    sub = Report("synthetic", "negative-control")
    _orig = reader_registry
    try:
        globals()["reader_registry"] = lambda: [r]
        check_readers(sub, arm, corpus[:6], fast=True)
    finally:
        globals()["reader_registry"] = _orig
    if any(c.code.startswith("R1.SYNTHETIC") and c.status == FAIL for c in sub.checks):
        rep.ok("S", "S8.silent_zero_reader",
               "a reader returning silent zeros for actions/score is FAILED by R1")
    else:
        rep.bad("S", "S8.silent_zero_reader",
                "THE GATE DID NOT CATCH a silently-zeroing reader",
                "; ".join(f"{c.code}={c.status}" for c in sub.checks))

    # S9 a non-deterministic builder must be caught by N4
    d = Path(tempfile.mkdtemp(prefix="localgate-nd-"))
    (d / "nd_builder.py").write_text(
        "import time\nfrom pathlib import Path\n"
        "OUT_DIR = Path('notebooks/does-not-exist')\n"
        "OUT_NB = OUT_DIR / 'x.ipynb'\n"
        "def build():\n"
        "    OUT_NB.parent.mkdir(parents=True, exist_ok=True)\n"
        "    OUT_NB.write_text(str(time.time_ns()))\n"
        "if __name__ == '__main__':\n    build()\n", encoding="utf-8")
    nd_arm = Arm(name="synthetic-nd", scorer_module="private_score", scorer_path=_PRIV,
                 builder_path=d / "nd_builder.py")
    sub = Report("synthetic-nd", "negative-control")
    check_builder_determinism(sub, nd_arm)
    if any(c.code == "N4.determinism" and c.status == FAIL for c in sub.checks):
        rep.ok("S", "S9.nondeterministic_builder",
               "a builder whose output changes between runs is FAILED by N4")
    else:
        rep.bad("S", "S9.nondeterministic_builder",
                "THE GATE DID NOT CATCH a non-deterministic builder",
                "; ".join(f"{c.code}={c.status}" for c in sub.checks))
    shutil.rmtree(d, ignore_errors=True)

    # S11 the H3 capture instrument must be able to report "not captured"
    try:
        neg = _run_harness_scenario("capture_hidden_only")
        if neg["visible_captured"]:
            rep.bad("S", "S11.capture_instrument",
                    "H3 reported the world model CAPTURED when it was delivered only "
                    "on the hidden reasoning channel -- H3 cannot refuse",
                    json.dumps({k: neg[k] for k in
                                ("carry_blocks", "visible_captured", "n_requests")}))
        else:
            rep.ok("S", "S11.capture_instrument",
                   f"a world model sent ONLY on reasoning_content is reported NOT "
                   f"captured ({neg['carry_blocks']} carry block(s) seen, marker absent) "
                   f"-- H3 measures the visible channel, not the transcript")
    except Exception as e:
        rep.warn("S", "S11.capture_instrument", f"could not run the control: {e}")

    # S12 the P-group guard instrument must be able to REFUSE.
    #     Undo the P0.4 fix (interior_signature := the shipped full-grid one) and
    #     P3 must FAIL: an instrument that reports "armed" whatever it is fed is
    #     feedback_guard_never_fired all over again.
    try:
        ng, fx = _p0_modules()
        orig_sig, orig_chg = fx.interior_signature, fx.interior_changed
        try:
            fx.interior_signature = lambda g, xr=(), xc=(): ng.board_signature(g)
            fx.interior_changed = (lambda a, b, xr, xc:
                                   ng.board_signature(a) != ng.board_signature(b))
            sub = Report("synthetic-p0", "negative-control")
            check_p0_instruments(sub)
        finally:
            fx.interior_signature, fx.interior_changed = orig_sig, orig_chg
        if any(c.code == "P3.interior_key_fires" and c.status == FAIL
               for c in sub.checks):
            rep.ok("S", "S12.guard_instrument_can_refuse",
                   "with the P0.4 interior re-key reverted to the shipped full-grid "
                   "signature, P3 FAILs -- the guard-behaviour instrument can refuse")
        else:
            rep.bad("S", "S12.guard_instrument_can_refuse",
                    "P3 still PASSED with the fix reverted -- the P-group cannot "
                    "detect a broken guard repair",
                    "; ".join(f"{c.code}={c.status}" for c in sub.checks
                              if c.group == "P"))
    except Exception as e:
        rep.warn("S", "S12.guard_instrument_can_refuse", f"control not runnable: {e}")

    # S13 the CADENCE instrument must be able to refuse: feed it an expectation
    #     it cannot meet and require validate() to report failures.
    try:
        if str(CADENCE_DIR) not in sys.path:
            sys.path.insert(0, str(CADENCE_DIR))
        ci = importlib.import_module("cadence_instrument")
        clean = _quiet(ci.validate, verbose=False)[0]   # (retval, captured stdout)
        orig = ci._EXPECTED
        try:
            ci._EXPECTED = {
                "runs/tufa_example_run/benchmark.json": {
                    "bp35": {"median_tokens_per_turn": 9999},
                },
            }
            poisoned = _quiet(ci.validate, verbose=False)[0]
        finally:
            ci._EXPECTED = orig
        if clean == 0 and poisoned > 0:
            rep.ok("S", "S13.cadence_instrument_can_refuse",
                   "cadence validate() returns 0 failures on the real artifacts and "
                   ">0 against a poisoned expectation -- it measures the artifact, not "
                   "its own constants")
        else:
            rep.bad("S", "S13.cadence_instrument_can_refuse",
                    f"cadence validate() cannot discriminate "
                    f"(clean={clean} failures, poisoned={poisoned} failures)")
    except Exception as e:
        rep.warn("S", "S13.cadence_instrument_can_refuse", f"control not runnable: {e}")

    # S10 cross-arm control must actually be able to refuse
    a_priv, a_graft = ARMS["private-base"], ARMS["graft-floor"]
    if a_graft.artifact and a_graft.artifact.is_dir():
        ok, why = _certifies(a_priv, a_graft.artifact)
        if ok:
            rep.bad("S", "S10.cross_arm_can_refuse",
                    "the private certifier ACCEPTED a graft artifact -- "
                    "the negative control cannot fire", why)
        else:
            rep.ok("S", "S10.cross_arm_can_refuse",
                   f"the private certifier refuses a real graft artifact ({why[:90]})")
    else:
        rep.skip("S", "S10.cross_arm_can_refuse", "no graft artifact on disk")

    return rep


# ===========================================================================
# 11. CLI
# ===========================================================================
def run_gate(arm_name: str, notebook: Path | None, fast: bool,
             skip_harness: bool) -> Report:
    arm = ARMS[arm_name]
    rep = Report(arm_name, "fast" if fast else "full")
    before = snapshot_protected()
    corpus = build_corpus()

    check_readers(rep, arm, corpus, fast)
    check_notebook(rep, arm, notebook, fast)
    if fast or skip_harness:
        rep.skip("H", "H*.harness",
                 "fast path -- the CPU harness smoke runs on --full (~1-2 min)")
    else:
        check_harness(rep, arm)
    check_arm_matrix(rep, arm)
    check_p0_instruments(rep)
    check_suites(rep, arm, fast)
    check_no_harm(rep, before)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="local_gate.py",
        description="LOCAL pre-push gate for the ARC-AGI-3 campaign (lane: local-rail)")
    ap.add_argument("--arm", choices=sorted(ARMS), help="which arm is about to be pushed")
    ap.add_argument("--notebook", type=Path, default=None,
                    help="override the arm's staged notebook path")
    ap.add_argument("--fast", action="store_true",
                    help="fast path (<2 min): skips the harness smoke, builder "
                         "determinism and the heavy per-arm suites")
    ap.add_argument("--full", action="store_true", help="full path (default)")
    ap.add_argument("--no-harness", action="store_true",
                    help="full path minus the CPU harness smoke")
    ap.add_argument("--self-test", action="store_true",
                    help="negative controls on this gate: deliberately break a "
                         "notebook and a reader, prove the gate FAILS")
    ap.add_argument("--corpus", action="store_true",
                    help="print the real-artifact fixture index and exit")
    ap.add_argument("--all-arms", action="store_true",
                    help="run every registered arm (reader/matrix coverage sweep)")
    ap.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    ap.add_argument("--quiet", action="store_true", help="JSON only, no table")
    args = ap.parse_args()

    if args.corpus:
        corpus = build_corpus()
        print(f"{'producer':<22} {'n':>4} {'bytes':>9}  path")
        for a in corpus:
            print(f"{a.producer[:22]:<22} {a.n_games:>4} {a.bytes:>9}  {a.rel}")
        print(f"\n{len(corpus)} real artifacts, "
              f"{sum(a.n_games for a in corpus)} game_runs, "
              f"{sum(a.bytes for a in corpus)/1e6:.1f} MB (referenced in place)")
        return 0

    if args.self_test:
        rep = self_test()
    elif args.all_arms:
        rep = Report("ALL-ARMS", "fast" if args.fast else "full")
        for name in sorted(ARMS):
            sub = run_gate(name, None, args.fast, args.no_harness)
            for c in sub.checks:
                c.code = f"{name}/{c.code}"
                rep.checks.append(c)
    elif args.arm:
        rep = run_gate(args.arm, args.notebook, args.fast, args.no_harness)
    else:
        ap.error("one of --arm / --all-arms / --self-test / --corpus is required")
        return 2

    if not args.quiet:
        print(rep.render())
    if args.json:
        args.json.write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
        if not args.quiet:
            print(f"  json -> {args.json}")
    elif args.quiet:
        print(json.dumps(rep.to_dict(), indent=2))
    return 1 if rep.n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
