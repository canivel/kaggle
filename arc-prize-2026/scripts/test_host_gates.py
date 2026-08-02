"""Unit tests for the additive host common-error gates H1-H4 in preflight.py.

Derived from the Kaggle host "500 Submissions Analyzed - Common Errors" post
(learnings/sweeps/discussions_2026-08-02.md). These gates are ADDITIVE and
OPT-IN; these tests exercise the pure `host_gates()` function directly (no
kaggle round-trip) with one positive and one negative case per gate, plus the
severity contract (WARN by default, FAIL under strict, WARN-on-missing-metadata
even under strict), and the "off = no gates" default.

Run either way:
  uv run python -m pytest scripts/test_host_gates.py -q
  uv run python scripts/test_host_gates.py            # no pytest needed
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preflight import host_gates, _family_of  # noqa: E402


# --- helpers ---------------------------------------------------------------

def _nb(*code_cells: str) -> dict:
    return {"cells": [{"cell_type": "code", "source": [c]} for c in code_cells]}


def _status(checks: list, code: str) -> str | None:
    for c in checks:
        if c["check"] == code:
            return c["status"]
    return None


GOOD_META = {
    "enable_gpu": True,
    "dataset_sources": ["driessmit1/arc3-vllm-h100-wheelhouse-v3"],
    "model_sources": [],
}
NO_GPU_META = {**GOOD_META, "enable_gpu": False}
NO_DS_META = {"enable_gpu": True, "dataset_sources": [], "model_sources": []}

CLEAN_NB = _nb(
    "import os\n",
    "resp = requests.get('http://gateway:8001/api/games')\n",
    "data = open('/kaggle/input/foo/bar.json').read()\n",  # READ is fine
)


# --- family classifier -----------------------------------------------------

def test_family_duck():
    assert _family_of("canivel/arc3-duck-repro") == "duck"
    assert _family_of("canivel/arc3-duck-gate") == "duck"


def test_family_nonduck():
    assert _family_of("canivel/arc3-baseline") == "arc3-baseline"


# --- H1: GPU accelerator ---------------------------------------------------

def test_H1_positive_gpu_enabled():
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, GOOD_META)
    assert _status(checks, "H1") == "OK"


def test_H1_negative_gpu_disabled_warns():
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, NO_GPU_META)
    assert _status(checks, "H1") == "WARN"


def test_H1_negative_gpu_disabled_strict_fails():
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, NO_GPU_META, strict=True)
    assert _status(checks, "H1") == "FAIL"


def test_H1_missing_metadata_warns_even_strict():
    # No kernel-metadata.json available: cannot verify -> WARN, never FAIL.
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, None, strict=True)
    assert _status(checks, "H1") == "WARN"


def test_H1_nonduck_family_na():
    checks = host_gates("canivel/arc3-baseline", CLEAN_NB, NO_GPU_META, strict=True)
    # baseline family is not GPU-required by this gate -> n/a OK, never FAIL.
    assert _status(checks, "H1") == "OK"


# --- H2: no three.arcprize.org --------------------------------------------

def test_H2_positive_clean():
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, GOOD_META)
    assert _status(checks, "H2") == "OK"


def test_H2_negative_forbidden_endpoint_warns():
    bad = _nb("r = requests.get('https://three.arcprize.org/api/games')\n")
    checks = host_gates("canivel/arc3-duck-repro", bad, GOOD_META)
    assert _status(checks, "H2") == "WARN"


def test_H2_negative_forbidden_endpoint_strict_fails():
    bad = _nb("r = requests.get('https://three.arcprize.org/api/games')\n")
    checks = host_gates("canivel/arc3-duck-repro", bad, GOOD_META, strict=True)
    assert _status(checks, "H2") == "FAIL"


# --- H3: no writes to /kaggle/input ---------------------------------------

def test_H3_positive_read_only_ok():
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, GOOD_META)
    assert _status(checks, "H3") == "OK"


def test_H3_negative_open_write_warns():
    bad = _nb("f = open('/kaggle/input/foo/out.txt', 'w')\n")
    checks = host_gates("canivel/arc3-duck-repro", bad, GOOD_META)
    assert _status(checks, "H3") == "WARN"


def test_H3_negative_writefile_strict_fails():
    bad = _nb("%%writefile /kaggle/input/foo.py\nprint('x')\n")
    checks = host_gates("canivel/arc3-duck-repro", bad, GOOD_META, strict=True)
    assert _status(checks, "H3") == "FAIL"


def test_H3_negative_to_parquet_warns():
    bad = _nb("df.to_parquet('/kaggle/input/comp/sub.parquet')\n")
    checks = host_gates("canivel/arc3-duck-repro", bad, GOOD_META)
    assert _status(checks, "H3") == "WARN"


# --- H4: dataset/model sources attached -----------------------------------

def test_H4_positive_dataset_attached():
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, GOOD_META)
    assert _status(checks, "H4") == "OK"


def test_H4_positive_model_source_counts():
    meta = {"enable_gpu": True, "dataset_sources": [], "model_sources": ["some/model"]}
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, meta)
    assert _status(checks, "H4") == "OK"


def test_H4_negative_none_attached_warns():
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, NO_DS_META)
    assert _status(checks, "H4") == "WARN"


def test_H4_negative_none_attached_strict_fails():
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, NO_DS_META, strict=True)
    assert _status(checks, "H4") == "FAIL"


def test_H4_missing_metadata_warns_even_strict():
    checks = host_gates("canivel/arc3-duck-repro", CLEAN_NB, None, strict=True)
    assert _status(checks, "H4") == "WARN"


# --- contract: default (non-strict) never emits FAIL ----------------------

def test_default_mode_never_fails():
    """All-bad notebook + all-bad metadata in non-strict mode: WARN only, no FAIL.
    This is the invariant that keeps an ALLOW from ever flipping to BLOCK."""
    bad = _nb(
        "requests.get('https://three.arcprize.org/api')\n",
        "open('/kaggle/input/x.txt','w')\n",
    )
    checks = host_gates("canivel/arc3-duck-repro", bad, NO_GPU_META, strict=False)
    assert all(c["status"] != "FAIL" for c in checks), checks


def test_strict_mode_can_fail():
    bad = _nb("requests.get('https://three.arcprize.org/api')\n")
    checks = host_gates("canivel/arc3-duck-repro", bad, NO_GPU_META, strict=True)
    assert any(c["status"] == "FAIL" for c in checks)


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
