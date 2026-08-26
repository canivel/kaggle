"""Unit tests for the preflight FAMILY PROFILES (added 2026-08-12).

Companion to test_host_gates.py, same house style: pure functions exercised
directly, plus two end-to-end run_preflight() tests with the `kaggle` pull
stubbed out (no network, no Kaggle round-trip anywhere in this file).

WHAT IS BEING PROTECTED
  1. The `arc3-baseline` agent-swarm family behaves EXACTLY as before. The
     golden check lists below are the verbatim pre-change output for a healthy
     swarm notebook and for a drifted one (the 5-FAIL shape that
     feedback_arc_kernel_structural_drift exists to catch).
  2. The duck-harness family (taaf + vLLM) reports K2/K4/K5/K6/K8 as N/A with
     an explicit reason -- never silently dropped, never counted in a verdict.
  3. D1-D4, the gate that actually applies to that family, does gate: a
     mutated metadata.kaggle is a hard BLOCK.

Run either way:
  uv run python -m pytest scripts/test_family_profiles.py -q
  uv run python scripts/test_family_profiles.py            # no pytest needed
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight  # noqa: E402
from preflight import (FAMILY_BASELINE, FAMILY_DUCK, detect_family,  # noqa: E402
                       duck_diff_checks, run_preflight, structural_checks,
                       summarize)


# --- fixtures --------------------------------------------------------------

def _cells(*specs) -> list[dict]:
    """specs = (cell_type, source) pairs."""
    return [{"cell_type": t, "source": [s]} for t, s in specs]


SWARM_RERUN_SRC = """
if os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
    open("agents/__init__.py", "w").write('''
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random
from dotenv import load_dotenv
load_dotenv()
AVAILABLE_AGENTS = {"random": Random, "myagent": MyAgent}
''')
    open(".env", "w").write('''
SCHEME=http
HOST=gateway
PORT=8001
ARC_API_KEY=test-key-123
ARC_BASE_URL=http://gateway:8001/api/games
OPERATION_MODE=swarm
RECORDINGS_DIR=/kaggle/working/recordings
''')
    os.chdir("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents")
    os.system("uv run main.py --agent myagent")
"""

SWARM_NB = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"kaggle": {"dataSources": [
        {"sourceType": "competition", "sourceId": "arc-prize-2026-arc-agi-3"}]}},
    "cells": _cells(
        ("markdown", "# baseline"),
        ("code", "%%writefile /kaggle/working/my_agent.py\nclass MyAgent: pass\n"),
        ("code", SWARM_RERUN_SRC),
    ),
}

# The war-eval baseline shape: 7 cells, code at absolute indices 2, 4, 6.
DUCK_BASE_NB = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"kernelspec": {"name": "python3"}},
    "cells": _cells(
        ("markdown", "# Tufa Labs ARC3 submission"),
        ("markdown", "## 1. Environment"),
        ("code", 'import os; os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"\n'),
        ("markdown", "## 2. Bundle"),
        ("code", 'DATASET_SOURCES = ["jeroencottaar/taaf-kaggle-source-share"]\n'
                 'with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as f: pass\n'),
        ("markdown", "## 3. Hook"),
        # The real war-eval run cell DOES mention KAGGLE_IS_COMPETITION_RERUN
        # (it branches on live gateway vs bundled offline files) -- keep it, so
        # the fixture reproduces the true old 5-FAIL shape under the strict
        # profile rather than a shorter "no rerun cell" one.
        ("code", "# customization hook: vanilla duck\n"
                 'if os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):\n'
                 "    bm.solver.setup()\n"),
    ),
}

# An arm graft: cells 2 and 6 carry the arm flag + the patch graft.
DUCK_ARM_NB = copy.deepcopy(DUCK_BASE_NB)
DUCK_ARM_NB["cells"][2]["source"] = [
    'import os; os.environ["WARPACK_FORCE_OFFLINE_BENCH"] = "1"\n'
    'os.environ["P1_SUPPRESS"] = "1"  # THE arm flag\n']
DUCK_ARM_NB["cells"][6]["source"] = [
    "# customization hook: P1 graft\nimport p1_suppressor_patch\n"
    "p1_suppressor_patch.apply(bm)\n"
    'if os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):\n'
    "    bm.solver.setup()\n"]


def _status(checks: list, code: str) -> str | None:
    for c in checks:
        if c["check"] == code:
            return c["status"]
    return None


def _msg(checks: list, code: str) -> str:
    for c in checks:
        if c["check"] == code:
            return c["message"]
    return ""


def _pairs(checks: list) -> list[tuple[str, str]]:
    return [(c["check"], c["status"]) for c in checks]


# ---------------------------------------------------------------------------
# 1. detect_family -- explicit, content-based, one-way
# ---------------------------------------------------------------------------

def test_detect_swarm_notebook_is_baseline_family():
    det = detect_family("canivel/arc3-baseline", SWARM_NB)
    assert det["family"] == FAMILY_BASELINE, det
    assert det["swarm_markers"], det


def test_detect_duck_notebook_is_duck_family():
    det = detect_family("canivel/arc3-duck-p1-eval", DUCK_ARM_NB)
    assert det["family"] == FAMILY_DUCK, det
    assert len(det["duck_markers"]) >= preflight.DUCK_MIN_MARKERS, det


def test_detect_is_one_way_swarm_marker_wins():
    """A notebook with BOTH duck markers and a swarm marker must fall back to
    the STRICT profile. Detection may tighten, never relax -- this is what
    keeps feedback_arc_kernel_structural_drift protection intact."""
    hybrid = copy.deepcopy(DUCK_ARM_NB)
    hybrid["cells"].append({"cell_type": "code",
                            "source": ["os.system('uv run main.py --agent myagent')\n"]})
    det = detect_family("canivel/arc3-duck-p1-eval", hybrid)
    assert det["family"] == FAMILY_BASELINE, det


def test_detect_slug_alone_cannot_relax():
    """A 'duck'-slugged kernel whose body has too few duck markers stays on
    the strict profile: the slug is provenance, not evidence."""
    thin = {"nbformat": 4, "nbformat_minor": 4, "metadata": {},
            "cells": _cells(("code", "print('hello')\n"))}
    det = detect_family("canivel/arc3-duck-whatever", thin)
    assert det["family"] == FAMILY_BASELINE, det


def test_detect_unknown_notebook_defaults_strict():
    det = detect_family("canivel/mystery", {"cells": []})
    assert det["family"] == FAMILY_BASELINE, det


# ---------------------------------------------------------------------------
# 2. arc3-baseline family COMPLETELY unchanged (golden lists)
# ---------------------------------------------------------------------------

def test_baseline_family_healthy_notebook_golden():
    checks = structural_checks(SWARM_NB, FAMILY_BASELINE)
    assert _pairs(checks) == [("K2", "OK"), ("K3", "OK"), ("K4", "OK"),
                              ("K5", "OK"), ("K5b", "OK"), ("K6", "OK"),
                              ("K8", "OK")], _pairs(checks)
    assert summarize("k", None, checks)["verdict"] == "ALLOW"


def test_baseline_family_drifted_notebook_still_blocks():
    """The 5-FAIL shape that ERRORed v45/v62/v63/v64/v65. If this ever stops
    BLOCKing, the gate has been weakened."""
    drifted = copy.deepcopy(SWARM_NB)
    drifted["metadata"].pop("kaggle")
    drifted["cells"][1]["source"] = ["print('no writefile here')\n"]
    drifted["cells"][2]["source"] = [
        'if os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):\n'
        '    open("agents/__init__.py","w").write("from .agent import Agent")\n']
    checks = structural_checks(drifted, FAMILY_BASELINE)
    assert _pairs(checks) == [("K2", "FAIL"), ("K3", "OK"), ("K4", "FAIL"),
                              ("K5", "FAIL"), ("K6", "FAIL"),
                              ("K8", "FAIL")], _pairs(checks)
    assert summarize("k", None, checks)["verdict"] == "BLOCK"


def test_baseline_family_is_the_default_argument():
    assert structural_checks(SWARM_NB) == structural_checks(SWARM_NB, FAMILY_BASELINE)


def test_baseline_family_never_emits_na():
    for nb in (SWARM_NB, DUCK_ARM_NB):
        checks = structural_checks(nb, FAMILY_BASELINE)
        assert all(c["status"] != "N/A" for c in checks), checks


def test_duck_notebook_under_baseline_profile_still_blocks():
    """Forcing --family arc3-baseline on a duck notebook reproduces the OLD
    behaviour exactly: the same 5 FAILs, hence BLOCK. The escape hatch is the
    profile, not a softened check."""
    checks = structural_checks(DUCK_ARM_NB, FAMILY_BASELINE)
    fails = [c["check"] for c in checks if c["status"] == "FAIL"]
    assert fails == ["K2", "K4", "K5", "K6", "K8"], fails


def test_summarize_without_extra_is_unchanged():
    """summarize() with no `extra` must produce exactly the original 6 keys --
    trusted-fork mode and daily_submit.py depend on this."""
    rep = summarize("k", 3, [{"check": "T3", "status": "OK", "message": "x"}])
    assert set(rep) == {"kernel", "version", "checks", "n_fail", "n_warn",
                        "verdict"}, sorted(rep)


# ---------------------------------------------------------------------------
# 3. duck-harness family: N/A is explicit, reasoned, and verdict-neutral
# ---------------------------------------------------------------------------

def test_duck_family_five_checks_are_na():
    checks = structural_checks(DUCK_ARM_NB, FAMILY_DUCK)
    assert _pairs(checks) == [("K2", "N/A"), ("K3", "OK"), ("K4", "N/A"),
                              ("K5", "N/A"), ("K6", "N/A"),
                              ("K8", "N/A")], _pairs(checks)


def test_duck_family_na_checks_carry_a_reason():
    checks = structural_checks(DUCK_ARM_NB, FAMILY_DUCK)
    for code in ("K2", "K4", "K5", "K6", "K8"):
        m = _msg(checks, code)
        assert m.startswith(f"inapplicable to family '{FAMILY_DUCK}' — "), (code, m)
        # a reason, not just a label
        assert len(m) > 80, (code, m)


def test_duck_family_k5b_absence_is_explained_not_silent():
    """K5b is a sub-leg of K5; it is not emitted for this family, so the K5
    N/A message must name it so the omission cannot look like a pass."""
    assert "K5b" in _msg(structural_checks(DUCK_ARM_NB, FAMILY_DUCK), "K5")


def test_na_does_not_affect_verdict():
    checks = structural_checks(DUCK_ARM_NB, FAMILY_DUCK)
    rep = summarize("k", None, checks)
    assert rep["verdict"] == "ALLOW", rep
    assert rep["n_fail"] == 0 and rep["n_warn"] == 0
    assert rep["n_na"] == 5, rep


def test_na_is_not_counted_as_ok():
    """The five skips must be distinguishable from passes in the report."""
    checks = structural_checks(DUCK_ARM_NB, FAMILY_DUCK)
    assert sum(1 for c in checks if c["status"] == "OK") == 1  # K3 only


# ---------------------------------------------------------------------------
# 4. K3 still gates in the duck family
# ---------------------------------------------------------------------------

def test_duck_family_k3_still_fires_on_nbformat_drift():
    """K3 is WARN-severity by original design (unchanged); the point is that
    it is NOT skipped for this family -- it still fires and still degrades the
    verdict away from a clean ALLOW."""
    bad = copy.deepcopy(DUCK_ARM_NB)
    bad["nbformat_minor"] = 5
    checks = structural_checks(bad, FAMILY_DUCK)
    assert _status(checks, "K3") == "WARN", checks
    assert summarize("k", None, checks)["verdict"] == "WARN"


# ---------------------------------------------------------------------------
# 5. D1-D4 -- the gate that actually applies
# ---------------------------------------------------------------------------

def test_D_happy_path_all_ok():
    checks = duck_diff_checks("canivel/arc3-duck-p1-eval", DUCK_ARM_NB,
                              "canivel/arc3-duck-war-eval", DUCK_BASE_NB)
    assert _pairs(checks) == [("D1", "OK"), ("D2", "OK"), ("D3", "OK"),
                              ("D4", "OK")], _pairs(checks)
    assert "[2, 6]" in _msg(checks, "D4"), _msg(checks, "D4")


def test_D2_mutated_metadata_kaggle_blocks():
    """A metadata.kaggle that differs from the baseline's is a HARD BLOCK --
    precisely the drift that ERRORed 5 kernels."""
    mutated = copy.deepcopy(DUCK_ARM_NB)
    mutated["metadata"]["kaggle"] = {"dataSources": [{"sourceType": "competition"}]}
    checks = duck_diff_checks("canivel/arc3-duck-p1-eval", mutated,
                              "canivel/arc3-duck-war-eval", DUCK_BASE_NB)
    assert _status(checks, "D2") == "FAIL", checks
    assert summarize("k", None, checks)["verdict"] == "BLOCK"


def test_D2_dropped_metadata_kaggle_blocks_the_other_direction():
    base_with_meta = copy.deepcopy(DUCK_BASE_NB)
    base_with_meta["metadata"]["kaggle"] = {"dataSources": [{"sourceType": "competition"}]}
    checks = duck_diff_checks("canivel/arc3-duck-p1-eval", DUCK_ARM_NB,
                              "canivel/arc3-duck-war-eval", base_with_meta)
    assert _status(checks, "D2") == "FAIL", checks


def test_D2_value_drift_inside_the_block_blocks():
    base = copy.deepcopy(DUCK_BASE_NB)
    base["metadata"]["kaggle"] = {"dockerImageVersionId": 1111}
    cand = copy.deepcopy(DUCK_ARM_NB)
    cand["metadata"]["kaggle"] = {"dockerImageVersionId": 2222}
    checks = duck_diff_checks("k", cand, "b", base)
    assert _status(checks, "D2") == "FAIL", checks


def test_D2_key_order_is_not_a_difference():
    base = copy.deepcopy(DUCK_BASE_NB)
    base["metadata"]["kaggle"] = {"a": 1, "b": 2}
    cand = copy.deepcopy(DUCK_ARM_NB)
    cand["metadata"]["kaggle"] = {"b": 2, "a": 1}
    checks = duck_diff_checks("k", cand, "b", base)
    assert _status(checks, "D2") == "OK", checks


def test_D3_cell_shape_drift_blocks_and_D4_declines():
    reauthored = copy.deepcopy(DUCK_ARM_NB)
    reauthored["cells"].append({"cell_type": "code", "source": ["print('extra')\n"]})
    checks = duck_diff_checks("k", reauthored, "b", DUCK_BASE_NB)
    assert _status(checks, "D3") == "FAIL", checks
    assert _status(checks, "D4") == "WARN", checks
    assert summarize("k", None, checks)["verdict"] == "BLOCK"


def test_D4_expect_diff_cells_match():
    checks = duck_diff_checks("k", DUCK_ARM_NB, "b", DUCK_BASE_NB,
                              expect_diff_cells=[2, 6])
    assert _status(checks, "D4") == "OK", checks


def test_D4_expect_diff_cells_mismatch_blocks():
    checks = duck_diff_checks("k", DUCK_ARM_NB, "b", DUCK_BASE_NB,
                              expect_diff_cells=[2, 4, 6])
    assert _status(checks, "D4") == "FAIL", checks
    assert summarize("k", None, checks)["verdict"] == "BLOCK"


def test_D4_unexpected_extra_edited_cell_blocks_under_expectation():
    """An arm that quietly edits the bundle-locating cell too."""
    sneaky = copy.deepcopy(DUCK_ARM_NB)
    sneaky["cells"][4]["source"] = ["DATASET_SOURCES = ['someone-else/bundle']\n"
                                    "with open(BUNDLE_DIR / 'deploy_target.pkl','rb') as f: pass\n"]
    checks = duck_diff_checks("k", sneaky, "b", DUCK_BASE_NB,
                              expect_diff_cells=[2, 6])
    assert _status(checks, "D4") == "FAIL", checks
    assert "[2, 4, 6]" in _msg(checks, "D4")


def test_D4_zero_diff_arm_kernel_warns():
    checks = duck_diff_checks("canivel/arc3-duck-p1-eval", DUCK_BASE_NB,
                              "canivel/arc3-duck-war-eval", DUCK_BASE_NB)
    assert _status(checks, "D4") == "WARN", checks


def test_D4_zero_diff_against_itself_is_ok():
    checks = duck_diff_checks("canivel/arc3-duck-war-eval", DUCK_BASE_NB,
                              "canivel/arc3-duck-war-eval", DUCK_BASE_NB)
    assert _status(checks, "D4") == "OK", checks


def test_D4_non_ascii_roundtrip_mangling_is_not_a_diff():
    """A Kaggle pull mangles non-ASCII; a cell differing ONLY by that must not
    be reported as an arm-defining diff (same policy as trusted-fork T3)."""
    base = copy.deepcopy(DUCK_BASE_NB)
    base["cells"][4]["source"] = ["# setup commands — wheels\nbm = 1\n"]
    cand = copy.deepcopy(DUCK_BASE_NB)
    cand["cells"][4]["source"] = ["# setup commands � wheels\nbm = 1\n"]
    checks = duck_diff_checks("canivel/arc3-duck-war-eval", cand,
                              "canivel/arc3-duck-war-eval", base)
    assert _status(checks, "D4") == "OK", checks
    assert "mangling" in _msg(checks, "D4")


def test_D1_missing_baseline_blocks():
    checks = duck_diff_checks("k", DUCK_ARM_NB, "canivel/nope", None)
    assert _pairs(checks) == [("D1", "FAIL")], checks
    assert summarize("k", None, checks)["verdict"] == "BLOCK"


# ---------------------------------------------------------------------------
# 6. end-to-end run_preflight with the kaggle pull stubbed (no network)
# ---------------------------------------------------------------------------

class _FakeCompleted:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _stub_pull(mapping: dict, fail_for: set | None = None):
    """Return a subprocess.run replacement that materialises `mapping`'s
    notebooks into the requested -p directory, mimicking `kaggle kernels pull`."""
    fail_for = fail_for or set()

    def _run(cmd, *a, **kw):
        ref = cmd[3]
        dest = Path(cmd[cmd.index("-p") + 1])
        if ref in fail_for or ref not in mapping:
            return _FakeCompleted(1, stderr=f"404 - {ref} not found")
        dest.mkdir(parents=True, exist_ok=True)
        slug = ref.split("/")[-1]
        (dest / f"{slug}.ipynb").write_text(json.dumps(mapping[ref]),
                                            encoding="utf-8")
        return _FakeCompleted(0, stdout="Source code downloaded")
    return _run


def _with_stub(stub, fn):
    orig = preflight.subprocess.run
    preflight.subprocess.run = stub
    try:
        return fn()
    finally:
        preflight.subprocess.run = orig


DUCK_MAP = {"canivel/arc3-duck-war-eval": DUCK_BASE_NB,
            "canivel/arc3-duck-p1-eval": DUCK_ARM_NB,
            "canivel/arc3-baseline": SWARM_NB}


def test_e2e_duck_kernel_allows_and_names_its_profile():
    rep = _with_stub(_stub_pull(DUCK_MAP),
                     lambda: run_preflight("canivel/arc3-duck-p1-eval", None,
                                           expect_diff_cells=[2, 6]))
    assert rep["verdict"] == "ALLOW", rep
    assert rep["family"] == FAMILY_DUCK, rep
    assert rep["family_detection"]["mode"] == "auto"
    assert rep["n_na"] == 5
    assert _pairs(rep["checks"]) == [
        ("K1", "OK"), ("K2", "N/A"), ("K3", "OK"), ("K4", "N/A"),
        ("K5", "N/A"), ("K6", "N/A"), ("K8", "N/A"),
        ("D1", "OK"), ("D2", "OK"), ("D3", "OK"), ("D4", "OK")], _pairs(rep["checks"])


def test_e2e_baseline_kernel_verdict_and_checks_unchanged():
    rep = _with_stub(_stub_pull(DUCK_MAP),
                     lambda: run_preflight("canivel/arc3-baseline", None))
    assert rep["verdict"] == "ALLOW", rep
    assert rep["family"] == FAMILY_BASELINE
    assert _pairs(rep["checks"]) == [
        ("K1", "OK"), ("K2", "OK"), ("K3", "OK"), ("K4", "OK"),
        ("K5", "OK"), ("K5b", "OK"), ("K6", "OK"), ("K8", "OK")], _pairs(rep["checks"])
    assert "n_na" not in rep


def test_e2e_duck_kernel_blocks_when_baseline_unreachable():
    rep = _with_stub(_stub_pull(DUCK_MAP, fail_for={"canivel/arc3-duck-war-eval"}),
                     lambda: run_preflight("canivel/arc3-duck-p1-eval", None))
    assert rep["verdict"] == "BLOCK", rep
    assert _status(rep["checks"], "D1") == "FAIL"


def test_e2e_k1_pull_failure_still_blocks_in_duck_family():
    """A genuine K1 failure short-circuits to BLOCK -- the family profile
    cannot rescue an unpullable kernel."""
    rep = _with_stub(_stub_pull(DUCK_MAP, fail_for={"canivel/arc3-duck-p1-eval"}),
                     lambda: run_preflight("canivel/arc3-duck-p1-eval", None))
    assert rep["verdict"] == "BLOCK", rep
    assert _status(rep["checks"], "K1") == "FAIL"


def test_e2e_explicit_family_flag_overrides_detection():
    rep = _with_stub(_stub_pull(DUCK_MAP),
                     lambda: run_preflight("canivel/arc3-duck-p1-eval", None,
                                           family=FAMILY_BASELINE))
    assert rep["family"] == FAMILY_BASELINE
    assert rep["family_detection"]["mode"] == "explicit"
    assert rep["verdict"] == "BLOCK", rep  # the old 5 FAILs, on demand


def test_e2e_local_baseline_path_is_accepted(tmp_path=None):
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="pf-test-"))
    p = d / "war.ipynb"
    p.write_text(json.dumps(DUCK_BASE_NB), encoding="utf-8")
    rep = _with_stub(_stub_pull({"canivel/arc3-duck-p1-eval": DUCK_ARM_NB}),
                     lambda: run_preflight("canivel/arc3-duck-p1-eval", None,
                                           baseline_ref=str(p)))
    assert rep["verdict"] == "ALLOW", rep
    assert str(p) in _msg(rep["checks"], "D1")


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _run_all():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"PASS {name}")
        except AssertionError as e:
            failed.append((name, e))
            print(f"FAIL {name}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
