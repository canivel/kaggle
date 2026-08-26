"""Synthetic-queue tests for the PROPOSED fall-through daemon (2026-08-20 incident fix).

Coordinator-required states:
  1. blocked head        -> entry 2 submits; blocked head STAYS queued; fall-through logged
  2. ALL entries blocked -> alarm artifact written, exit 1, nothing submitted
  3. healthy head        -> byte-for-byte old behavior (head fires, never skipped) [inverse guard]
Run: python scripts/proposed/test_fallthrough.py
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ds_prop", Path(__file__).parent / "daily_submit_fallthrough.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)

FAILURES = []


def check(name, cond, detail=""):
    print(("  [OK] " if cond else "  [FAIL] ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


def scenario(name, entries, preflight_verdicts):
    """Run main() against a synthetic queue. preflight_verdicts: kernel -> ALLOW|BLOCK."""
    root = Path(tempfile.mkdtemp())
    (root / "runs").mkdir()
    import datetime
    (root / "ITERATION_LOG.md").write_text(
        f"### {datetime.date.today().isoformat()}\n- synthetic\n", encoding="utf-8")
    queue = root / "submission_queue.json"
    queue.write_text(json.dumps({"pending": entries, "history": []}), encoding="utf-8")

    events, submitted = [], []
    ds.ROOT = root
    ds.QUEUE = queue
    ds.log = lambda rec: events.append(rec)
    ds._kaggle_cli = lambda: "kaggle"
    ds.already_submitted_today = lambda k: False
    ds.run_preflight = lambda item: {
        "verdict": preflight_verdicts.get(item["kernel"], "ALLOW"),
        "checks": [{"check": "SYN", "status": "FAIL", "message": "synthetic block"}]
        if preflight_verdicts.get(item["kernel"]) == "BLOCK" else [],
    }
    ds.submit = lambda kaggle, item: (submitted.append(item["kernel"]) or (True, "", ""))
    rc = ds.main()
    q_after = json.loads(queue.read_text(encoding="utf-8"))
    alarms = list((root / "runs").glob("submit_alarm_*.json"))
    print(f"== {name} ==")
    return rc, events, submitted, q_after, alarms


E = lambda k: {"kernel": k, "version": 1, "file": "submission.parquet", "message": "syn"}

# --- 1. blocked head -> entry 2 fires; head stays queued
rc, ev, sub, q, al = scenario("blocked head falls through",
                              [E("a/blocked-head"), E("a/healthy-2")],
                              {"a/blocked-head": "BLOCK"})
check("exit 0", rc == 0)
check("entry 2 submitted", sub == ["a/healthy-2"], repr(sub))
check("blocked head STAYS in queue", any(x["kernel"] == "a/blocked-head" for x in q["pending"]))
check("submitted entry removed from queue", not any(x["kernel"] == "a/healthy-2" for x in q["pending"]))
check("preflight-block logged loudly", any(e.get("skip") == "preflight-blocked" for e in ev))
check("fall-through event logged", any(e.get("event") == "fall-through" for e in ev))
check("no alarm artifact", not al)
check("history gained the fired entry", any(h["kernel"] == "a/healthy-2" for h in q["history"]))

# --- 2. ALL blocked -> alarm, exit 1, nothing submitted
rc, ev, sub, q, al = scenario("all entries blocked -> alarm",
                              [E("a/blocked-1"), E("a/blocked-2")],
                              {"a/blocked-1": "BLOCK", "a/blocked-2": "BLOCK"})
check("exit 1", rc == 1)
check("nothing submitted", sub == [])
check("ALL-ENTRIES-BLOCKED logged", any(e.get("event") == "ALL-ENTRIES-BLOCKED" for e in ev))
check("alarm artifact written", len(al) == 1)
check("alarm lists both blocks", len(json.loads(al[0].read_text())["blocked"]) == 2 if al else False)
check("queue untouched (both stay for fixing)", len(q["pending"]) == 2)

# --- 3. healthy head -> old behavior byte-for-byte; INVERSE GUARD: head never skipped
rc, ev, sub, q, al = scenario("healthy head unchanged (inverse guard)",
                              [E("a/healthy-head"), E("a/filler")], {})
check("exit 0", rc == 0)
check("HEAD fired, not skipped", sub == ["a/healthy-head"], repr(sub))
check("filler still queued", any(x["kernel"] == "a/filler" for x in q["pending"]))
check("no fall-through event on healthy head", not any(e.get("event") == "fall-through" for e in ev))
check("no block events", not any("blocked" in str(e.get("skip", "")) for e in ev))
check("no alarm", not al)

# --- 4. evidence-gate block also falls through (same class as preflight)
rc, ev, sub, q, al = scenario("evidence-blocked head falls through",
                              [dict(E("a/needs-ev"), requires_evidence=True), E("a/healthy-3")], {})
check("exit 0", rc == 0)
check("entry 2 submitted past evidence block", sub == ["a/healthy-3"], repr(sub))
check("evidence block logged", any(e.get("skip") == "evidence-gate-blocked" for e in ev))

print()
if FAILURES:
    print(f"TESTS FAILED ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("ALL FALL-THROUGH TESTS PASSED (counted at runtime)")

# ===== SUBMIT SUCCESS-CHECK TESTS (2026-08-21 00:31Z incident) =====
print()
print("== submit() success-check scenarios ==")
import types

# The fall-through scenarios above monkeypatched ds.submit with a lambda; use a
# FRESH module instance so these tests exercise the real submit().
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)

def run_submit_scenario(name, list_sequence, submit_rc, submit_out, msg="A21 EXPLORATION DRAW xyz"):
    """list_sequence: successive CSV stdouts for `submissions -v` calls (repeats last)."""
    calls = {"n": 0}
    warns = []
    def fake_run(cmd, capture_output=True, text=True, timeout=0):
        r = types.SimpleNamespace(returncode=0, stdout="", stderr="")
        if "submissions" in cmd:
            idx = min(calls["n"], len(list_sequence) - 1)
            r.stdout = list_sequence[idx]
            calls["n"] += 1
        else:  # the submit command
            r.returncode = submit_rc
            r.stdout = submit_out
        return r
    ds.subprocess.run = fake_run
    ds.time.sleep = lambda s: None
    ds.log = lambda rec: warns.append(rec)
    ok, out, err = ds.submit("kaggle", {"kernel": "a/k", "version": 1, "message": msg})
    return ok, warns

HDR = "ref,fileName,date,description,status,publicScore,privateScore"
OLD = HDR + "\n111,submission.parquet,2026-08-20 00:07:11,old row,COMPLETE,0.41,"
NEW = OLD + "\n222,submission.parquet,2026-08-21 00:31:23,\"A21 EXPLORATION DRAW xyz\",PENDING,,"

# A. tonight's exact failure class: list lags forever, CLI succeeded -> assumed-ok + loud warn
ok, warns = run_submit_scenario("list lags forever", [OLD], 0, "Successfully submitted to X")
check("listlag + CLI success -> ok=True (head pops, no double-fire)", ok is True)
check("listlag warned loudly", any(w.get("warn") == "submit-listlag-assumed-ok" for w in warns))

# B. lagging list that catches up on the 3rd read -> ok, no warn
ok, warns = run_submit_scenario("lag then appear", [OLD, OLD, NEW], 0, "Successfully submitted")
check("lag-then-appear -> ok=True via retry", ok is True)
check("no listlag warn when the row appears", not warns)

# C. real failure: CLI non-zero, list never changes -> ok=False
ok, warns = run_submit_scenario("real failure", [OLD], 1, "403 Forbidden")
check("real failure -> ok=False", ok is False)

# D. immediate appearance (healthy path) -> ok, first read
ok, warns = run_submit_scenario("immediate", [OLD, NEW], 0, "Successfully submitted")
check("immediate new row -> ok=True", ok is True)

# E. empty list output on BOTH sides (tonight's ''=='' trap) + CLI success -> ok=True, warned
ok, warns = run_submit_scenario("empty list both sides", [""], 0, "Successfully submitted")
check("empty-list trap -> ok=True (CLI trusted), not false-negative", ok is True)
check("empty-list trap warned", any(w.get("warn") == "submit-listlag-assumed-ok" for w in warns))

# F. new row appears but description truncated (no msg token) + CLI success -> corroborated ok
TRUNC = OLD + "\n333,submission.parquet,2026-08-21 00:31:23,\"A21 EXPLORA...\",PENDING,,"
ok, warns = run_submit_scenario("truncated desc", [OLD, TRUNC], 0, "Successfully submitted")
check("new row + truncated desc + marker -> ok=True", ok is True)

print()
if FAILURES:
    print(f"TESTS FAILED ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("ALL TESTS PASSED (fall-through + submit success-check)")
