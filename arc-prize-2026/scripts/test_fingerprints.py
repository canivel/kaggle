"""Unit tests for the two-tier failure-fingerprint system.

Run either way:
  uv run python -m pytest scripts/test_fingerprints.py -q
  uv run python scripts/test_fingerprints.py            # no pytest needed
"""
from __future__ import annotations

import datetime as dt
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprints import (  # noqa: E402
    STORE_PATH, candidate_families, elapsed_bucket, empty_store,
    extract_failure_signal, family_index, format_staleness_banner,
    iter_log_sources, load_store, normalize_error, recurrence_check,
    staleness_report, tier1_fingerprint, tier1_root_fingerprint,
    tier2_families, tier2_fingerprint,
)

# ---------------------------------------------------------------------------
# 1. Normalization
# ---------------------------------------------------------------------------

NASTY = [
    'FileNotFoundError: [Errno 2] No such file: /kaggle/working/agents/__init__.py',
    'IndexError: list index out of range at line 417 in /opt/conda/lib/x.py',
    'run 8f14e45f-ceea-4a2b-9d3c-1a2b3c4d5e6f finished 2026-07-18T04:00:12Z',
    'game ft09-0a0ad940deadbee1 crashed at 12:33:44 after 4096 steps',
    'RuntimeError: CUDA out of memory. Tried to allocate 9.90 GiB at 0x7fa3b2c40000',
    'worker died, heartbeat 2026-06-28 00:07:11 pid 31337',
    'C:\\Users\\dcani\\AppData\\Local\\Temp\\preflight-abc123\\kernel.ipynb missing',
    '', '   ', 'plain message with no volatile tokens',
]


def test_normalization_idempotent():
    for s in NASTY:
        once = normalize_error(s)
        twice = normalize_error(once)
        assert once == twice, f"not idempotent: {s!r} -> {once!r} -> {twice!r}"


def test_normalization_collapses_volatiles():
    a = normalize_error("IndexError: index 5 out of range, line 417, "
                        "run 8f14e45f-ceea-4a2b-9d3c-1a2b3c4d5e6f")
    b = normalize_error("IndexError: index 9 out of range, line 23, "
                        "run 00000000-1111-2222-3333-444444444444")
    assert a == b
    assert "<uuid>" in a and "line <n>" in a


def test_normalization_preserves_error_identity():
    a = normalize_error("IndexError: list index out of range")
    b = normalize_error("KeyError: 'ft09'")
    assert a != b
    assert a.startswith("IndexError")


# ---------------------------------------------------------------------------
# 2. Tier-1 fingerprints: distinct errors don't collide; volatile-token
#    variants of the SAME error do collide (that's the point)
# ---------------------------------------------------------------------------

def test_distinct_errors_do_not_collide():
    f1 = tier1_fingerprint("stage: swarm boot", "IndexError: list index out of range")
    f2 = tier1_fingerprint("stage: swarm boot", "KeyError: 'SCHEME'")
    f3 = tier1_fingerprint("stage: model load", "IndexError: list index out of range")
    assert f1["fingerprint"] != f2["fingerprint"]      # different error
    assert f1["fingerprint"] != f3["fingerprint"]      # different stage
    assert len({f1["fingerprint"], f2["fingerprint"], f3["fingerprint"]}) == 3


def test_same_error_different_volatiles_collide():
    f1 = tier1_fingerprint(
        "game ab12cd34ef567890 step 100",
        "FileNotFoundError: /kaggle/tmp/run-11112222-3333-4444-5555-666677778888/x.pkl")
    f2 = tier1_fingerprint(
        "game 99887766deadbeef step 4096",
        "FileNotFoundError: /kaggle/tmp/run-aaaabbbb-cccc-dddd-eeee-ffff00001111/y.pkl")
    assert f1["fingerprint"] == f2["fingerprint"]


# ---------------------------------------------------------------------------
# 3. Silent-death keying: last-progress marker + elapsed bucket IS the material
# ---------------------------------------------------------------------------

def test_silent_death_keying():
    s1 = tier1_fingerprint("phase1 v2: budget=6 game 12/25", None, elapsed=2.5 * 3600)
    s2 = tier1_fingerprint("phase1 v2: budget=6 game 19/25", None, elapsed=3.9 * 3600)
    s3 = tier1_fingerprint("vllm server: waiting for weights", None, elapsed=2.5 * 3600)
    s4 = tier1_fingerprint("phase1 v2: budget=6 game 12/25", None, elapsed=9 * 3600)
    assert s1["mode"] == "silent" and s1["error"] is None
    assert s1["fingerprint"] == s2["fingerprint"]   # same stage skeleton+bucket
    assert s1["fingerprint"] != s3["fingerprint"]   # different last-progress marker
    assert s1["fingerprint"] != s4["fingerprint"]   # different elapsed bucket
    assert s1["material"].startswith("t1-silent|")


def test_elapsed_buckets():
    assert elapsed_bucket(30) == "<10m"
    assert elapsed_bucket(1800) == "10-60m"
    assert elapsed_bucket(2 * 3600) == "1-4h"
    assert elapsed_bucket(5 * 3600) == "4-8h"
    assert elapsed_bucket(12 * 3600) == ">8h"


def test_silent_signal_extraction_from_log():
    recs = [
        {"stream_name": "stdout", "time": 10.0, "data": "boot ok\n"},
        {"stream_name": "stdout", "time": 60.0, "data": "game 3/25 level 1\n"},
        {"stream_name": "stderr", "time": 61.0, "data": "SyntaxWarning: blah\n"},
    ]
    sig = extract_failure_signal(recs)
    assert sig["error"] is None
    assert sig["stage"] == "game 3/25 level 1"
    assert sig["elapsed"] == 61.0


def test_error_signal_extraction_ignores_code_echoes():
    recs = [
        {"stream_name": "stdout", "time": 1.0, "data": "def assert_expected_cuda_gpu() -> None:\n"},
        {"stream_name": "stdout", "time": 2.0, "data": "stage: swarm boot\n"},
        {"stream_name": "stderr", "time": 3.0, "data": "Traceback (most recent call last):\n"},
        {"stream_name": "stderr", "time": 3.1, "data": '  File "/x/main.py", line 4, in <module>\n'},
        {"stream_name": "stderr", "time": 3.2, "data": "ImportError: cannot import name 'Swarm'\n"},
    ]
    sig = extract_failure_signal(recs)
    assert sig["error"] == "ImportError: cannot import name 'Swarm'"
    assert sig["stage"] == "stage: swarm boot"


# ---------------------------------------------------------------------------
# 4. Tier-2 + recurrence WARN logic
# ---------------------------------------------------------------------------

def _mk_store(incidents):
    store = empty_store()
    store["incidents"] = incidents
    return store


def _t2_incident(iid, date, kernel, version, status, score_class,
                 provenance=None, confidence="high"):
    fp = tier2_fingerprint(kernel, version, status, score_class)
    return {"id": iid, "tier": 2, "date": date, "kernel": kernel,
            "version": version, "status_class": status,
            "score_class": score_class,
            "fingerprint": fp["fingerprint"],
            "families": tier2_families(kernel, status, score_class, provenance),
            "confidence": confidence, "source": "test"}


def test_recurrence_no_warn_below_two_deaths():
    store = _mk_store([
        _t2_incident("i1", "2026-06-20", "canivel/x", 1, "ERROR", "none"),
    ])
    rec = recurrence_check(store, candidate_families("canivel/x"))
    assert rec["warn"] is False and rec["matches"] == []


def test_recurrence_warn_at_two_deaths_with_refs():
    store = _mk_store([
        _t2_incident("i1", "2026-06-20", "canivel/x", 1, "ERROR", "none"),
        _t2_incident("i2", "2026-06-22", "canivel/x", 2, "ERROR", "none"),
    ])
    rec = recurrence_check(store, candidate_families("canivel/x"))
    assert rec["warn"] is True
    m = rec["matches"][0]
    assert m["family"] == "slug:canivel/x"
    assert m["n_prior_deaths"] == 2
    assert m["incidents"] == ["i1", "i2"]
    assert m["first_seen"] == "2026-06-20" and m["last_seen"] == "2026-06-22"


def test_recurrence_family_isolation():
    """Deaths on another slug must not warn this candidate."""
    store = _mk_store([
        _t2_incident("i1", "2026-06-20", "canivel/other", 1, "ERROR", "none"),
        _t2_incident("i2", "2026-06-22", "canivel/other", 2, "ERROR", "none"),
    ])
    rec = recurrence_check(store, candidate_families("canivel/x"))
    assert rec["warn"] is False


def test_recurrence_provenance_family_crosses_slugs():
    """The structural-drift signature: scratch-built deaths on DIFFERENT
    slugs still warn a scratch-built candidate."""
    store = _mk_store([
        _t2_incident("v45", "2026-05-26", "canivel/arc3-final", 26,
                     "ERROR", "none", provenance="scratch-built"),
        _t2_incident("v62", "2026-06-20", "canivel/arc3-forge62", 4,
                     "ERROR", "none", provenance="scratch-built"),
    ])
    fams = candidate_families("canivel/arc3-jepa-v2", "scratch-built")
    rec = recurrence_check(store, fams)
    assert rec["warn"] is True
    assert rec["matches"][0]["family"] == "provenance:scratch-built"
    # ...but a baseline-derived candidate is NOT warned by that family
    fams2 = candidate_families("canivel/arc3-baseline", "baseline-derived")
    assert recurrence_check(store, fams2)["warn"] is False


def test_recurrence_before_date_cutoff():
    """Chronological replay: deaths on/after the cutoff don't count."""
    store = _mk_store([
        _t2_incident("i1", "2026-06-20", "canivel/x", 1, "ERROR", "none"),
        _t2_incident("i2", "2026-06-22", "canivel/x", 2, "ERROR", "none"),
    ])
    rec = recurrence_check(store, candidate_families("canivel/x"),
                           before_date="2026-06-22")
    assert rec["warn"] is False  # only 1 death strictly before 06-22


def test_recurrence_warn_only_shape():
    """The consumer contract: result is a dict with warn/matches; nothing in
    it can block (no verdict field)."""
    rec = recurrence_check(empty_store(), candidate_families("canivel/x"))
    assert set(rec) == {"warn", "min_deaths", "families_checked", "matches"}
    assert rec["warn"] is False


# ---------------------------------------------------------------------------
# 5. Wrapper vs ROOT error (2026-08-16)
#    On the Kaggle rail every death is re-raised as CalledProcessError by the
#    cell and PapermillExecutionError by papermill, so the LAST error line is a
#    wrapper carrying no diagnosis. The A17 canary deaths (missing model mount)
#    and the LoRA canary death (NameError in the setup heredoc) share that
#    wrapper surface and are NOT the same failure.
# ---------------------------------------------------------------------------

def _wrapped(inner: str):
    return [
        {"stream_name": "stdout", "time": 1.0, "data": "PYSETUP\n"},
        {"stream_name": "stderr", "time": 2.0, "data": "Traceback (most recent call last):\n"},
        {"stream_name": "stderr", "time": 2.1, "data": '  File "<stdin>", line 256, in f\n'},
        {"stream_name": "stderr", "time": 2.2, "data": inner + "\n"},
        {"stream_name": "stderr", "time": 3.0, "data": "Traceback (most recent call last):\n"},
        {"stream_name": "stderr", "time": 3.1, "data": '  File "/x/subprocess.py", line 571, in run\n'},
        {"stream_name": "stderr", "time": 3.2,
         "data": "CalledProcessError: Command '\"$PYTHON\" - <<'PYSETUP'\n"},
    ]


def test_root_error_is_captured_behind_the_wrapper():
    sig = extract_failure_signal(
        _wrapped("NameError: name '_source_path_entries' is not defined"))
    assert sig["error"].startswith("CalledProcessError")        # terminal line
    assert sig["root_error"] == \
        "NameError: name '_source_path_entries' is not defined"  # the diagnosis


def test_same_wrapper_different_roots_get_different_root_families():
    a = extract_failure_signal(
        _wrapped("NameError: name '_source_path_entries' is not defined"))
    b = extract_failure_signal(
        _wrapped("RuntimeError: A17-CANARY FATAL: model not found under /kaggle/input"))
    fa = tier1_fingerprint(a["stage"], a["error"])
    fb = tier1_fingerprint(b["stage"], b["error"])
    assert fa["fingerprint"] == fb["fingerprint"]      # same wrapper surface
    ra = tier1_root_fingerprint(a["root_error"])
    rb = tier1_root_fingerprint(b["root_error"])
    assert ra["fingerprint"] != rb["fingerprint"]      # different actual deaths


def test_root_error_none_when_the_only_error_is_the_root():
    recs = [
        {"stream_name": "stdout", "time": 1.0, "data": "stage: boot\n"},
        {"stream_name": "stderr", "time": 2.0, "data": "Traceback (most recent call last):\n"},
        {"stream_name": "stderr", "time": 2.2, "data": "KeyError: 'ft09'\n"},
    ]
    sig = extract_failure_signal(recs)
    assert sig["error"] == "KeyError: 'ft09'"
    assert sig["root_error"] is None
    assert tier1_root_fingerprint(None) is None


# ---------------------------------------------------------------------------
# 6. Log inventory: one incident per RUN
# ---------------------------------------------------------------------------

def _mk_log(path: Path, text: str = "boot ok\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_log_inventory_one_source_per_run():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _mk_log(root / "runs/kernel_pulls/q38_v1/q38.log", '[{"data":"x"}]')
        _mk_log(root / "runs/kernel_pulls/q38_v1/q38_flat.log")     # derived dup
        _mk_log(root / "runs/kernel_pulls/b122_v1/arc3-b122.log")
        _mk_log(root / "runs/kernel_pulls/b122_v1/vllm-openai-server.log")  # sidecar
        _mk_log(root / "runs/kernel_logs/lora_v1.log.json", '[{"data":"x"}]')
        _mk_log(root / "runs/kernel_logs/other_v1.log.json", '[{"data":"x"}]')
        rels = sorted(s["rel"] for s in iter_log_sources(root))
    assert rels == [
        "runs/kernel_logs/lora_v1.log.json",
        "runs/kernel_logs/other_v1.log.json",   # kernel_logs = one run per FILE
        "runs/kernel_pulls/b122_v1/arc3-b122.log",
        "runs/kernel_pulls/q38_v1/q38.log",     # not q38_flat.log
    ], rels


# ---------------------------------------------------------------------------
# 7. STALENESS BANNER — the 2026-08-16 defect itself
#    The report is READ-ONLY; the store is written only by the backfill. Nobody
#    ran the backfill between 07-18 and 08-16, so the reader described a store
#    3 incidents behind the logs on disk and the 08-09 weekly wrote down "no
#    new incidents in ~4.5 weeks" as a finding. The reader must now say so.
# ---------------------------------------------------------------------------

def _src(rel: str, when: str) -> dict:
    return {"rel": rel, "run_key": rel,
            "mtime": dt.datetime.fromisoformat(when).replace(
                tzinfo=dt.timezone.utc)}


def _store_written(when: str, scanned: list[str], dates: list[str]) -> dict:
    st = empty_store()
    st["updated"] = when
    st["scan_log"] = [{"log": r, "has_error": False} for r in scanned]
    st["incidents"] = [{"id": f"i{n}", "date": d, "families": [], "tier": 1}
                       for n, d in enumerate(dates)]
    return st


def test_staleness_fires_on_never_scanned_log():
    """The exact 2026-08-09 situation: two logs on disk the writer never saw."""
    store = _store_written("2026-07-18T12:00:00Z",
                           scanned=["runs/kernel_pulls/war_v1/arc3-duck-war.log"],
                           dates=["2026-07-08"])
    sources = [
        _src("runs/kernel_pulls/war_v1/arc3-duck-war.log", "2026-07-01T00:00:00"),
        _src("runs/kernel_pulls/a17_canary_v1/arc3-a17-72b-canary.log",
             "2026-07-25T08:24:00"),
        _src("runs/kernel_logs/lora_serve_canary_v1.log.json",
             "2026-08-16T08:24:00"),
    ]
    st = staleness_report(store, sources)
    assert st["stale"] is True
    assert st["n_unscanned"] == 2
    assert "runs/kernel_logs/lora_serve_canary_v1.log.json" in st["unscanned"]
    banner = format_staleness_banner(st)
    assert "STALE FAILURE-FINGERPRINT STORE" in banner
    assert "NEVER SCANNED" in banner
    assert "fingerprint_backfill.py" in banner          # names its own fix
    assert "runs/kernel_logs/lora_serve_canary_v1.log.json" in banner
    assert st["newest_incident_date"] == "2026-07-08"   # store is 39 days behind
    assert st["newest_log_date"] == "2026-08-16"


def test_staleness_fires_when_a_log_is_newer_than_the_store():
    """Scanned once, then the same log was re-pulled after the store was written."""
    rel = "runs/kernel_pulls/q38_v1/q38.log"
    store = _store_written("2026-08-15T10:00:00Z", scanned=[rel],
                           dates=["2026-08-15"])
    st = staleness_report(store, [_src(rel, "2026-08-15T19:15:00")])
    assert st["stale"] is True
    assert st["n_unscanned"] == 0
    assert any("modified AFTER" in r for r in st["reasons"])
    assert "STALE" in format_staleness_banner(st)


def test_staleness_fires_on_empty_store():
    st = staleness_report(empty_store(), [])
    assert st["stale"] is True
    assert any("ZERO incidents" in r for r in st["reasons"])


def test_no_staleness_when_store_is_current():
    rel = "runs/kernel_pulls/q38_v1/q38.log"
    store = _store_written("2026-08-16T12:00:00Z", scanned=[rel],
                           dates=["2026-08-15"])
    st = staleness_report(store, [_src(rel, "2026-08-15T19:15:00")])
    assert st["stale"] is False, st["reasons"]
    assert "STALE" not in format_staleness_banner(st)
    assert "FRESH" in format_staleness_banner(st)


def test_live_store_is_not_stale():
    """Guards the duty order: if this fails, the backfill has not been run."""
    st = staleness_report(load_store())
    assert st["stale"] is False, (
        f"{STORE_PATH} is STALE -- run scripts/fingerprint_backfill.py: "
        f"{st['reasons']}")


# ---------------------------------------------------------------------------
# 8. Live-store content regressions (the incidents that were missing)
# ---------------------------------------------------------------------------

PYSETUP_FAMILY = "t1:fb1e96c3815797ad"


def test_live_store_pysetup_family_present_with_two_deaths():
    idx = family_index(load_store())
    incs = idx.get(PYSETUP_FAMILY, [])
    assert len(incs) >= 2, (
        f"{PYSETUP_FAMILY} has {len(incs)} incident(s); the A17 canary pair "
        "(2026-07-25) must be in the store")
    assert all("PYSETUP" in i.get("material", "") for i in incs)
    assert all(i.get("date") == "2026-07-25" for i in incs)


def test_live_store_has_the_august_deaths():
    incs = load_store()["incidents"]
    by_src = {i.get("source"): i for i in incs}
    lora = by_src.get("runs/kernel_logs/lora_serve_canary_v1.log.json")
    assert lora is not None, "lora-serve-canary v1 (08-14) missing from the store"
    assert lora["kernel"] == "canivel/arc3-lora-serve-canary"
    # the log was PULLED 08-16; the run DIED 08-14 -- mtime must not misdate it
    assert lora["date"] == "2026-08-14", lora["date"]
    assert "_source_path_entries" in (lora.get("root_error") or "")

    q38 = by_src.get("runs/kernel_pulls/q38_v1/q38.log")
    assert q38 is not None, "Q38 v1 (08-15) missing from the store"
    assert q38["kernel"] == "canivel/arc3-q38-engine-eval"
    assert q38["date"] == "2026-08-15"

    b122 = by_src.get("runs/kernel_pulls/b122_v1/arc3-b122-boot-canary.log")
    assert b122 is not None and b122["date"] == "2026-08-14"
    # ...and the three 08-14/08-15 deaths are NOT the same failure
    roots = {lora["root_fingerprint"], q38["root_fingerprint"],
             b122["root_fingerprint"]}
    assert len(roots) == 3


def test_live_store_pysetup_wrapper_is_not_the_lora_death():
    """Honesty guard: the A17 PYSETUP family did not predict the LoRA NameError.
    Same heredoc surface, different root error, different family."""
    incs = load_store()["incidents"]
    a17 = [i for i in incs if i.get("fingerprint") == "fb1e96c3815797ad"]
    lora = [i for i in incs
            if i.get("source") == "runs/kernel_logs/lora_serve_canary_v1.log.json"]
    assert a17 and lora
    assert lora[0]["fingerprint"] != a17[0]["fingerprint"]
    assert lora[0]["root_fingerprint"] != a17[0]["root_fingerprint"]
    assert "A17-CANARY FATAL" in (a17[0].get("root_error") or "")


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
