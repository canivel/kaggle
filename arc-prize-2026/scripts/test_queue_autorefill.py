"""Unit tests for the submission-queue AUTO-REFILL (eternal fallback).

Standing defect (flagged in ITERATION_LOG.md 2026-08-11 / 2026-08-12): the
daemon popped the head of `submission_queue.json` and nothing ever refilled
it, so every morning the queue was `{"pending": []}` and only a human noticing
at the 06:00 check kept the daily draw alive. A missed fire is unrecoverable —
the window refreshes at 20:00 and there is no catch-up.

These tests pin the fix in scripts/daily_submit.py:
  * submit -> queue drained -> fallback re-armed (phase "post-submit")
  * wake with an already-empty queue -> self-arm and FIRE (phase "pre-fire"),
    never a no-op "queue-empty" skip
  * the refill is idempotent within a day (a second wake is a no-op)
  * a hand-queued EXPERIMENTAL entry is never clobbered
  * every arming writes one `queue_autorefill` record to the submission log

Everything is hermetic: ROOT/QUEUE/LOG are redirected to tmp_path and the
kaggle CLI, the submissions listing, the preflight runner and submit() are all
monkeypatched. No network, no kaggle CLI, no writes to the real queue or log.

Run:
  uv run python -m pytest scripts/test_queue_autorefill.py -q
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_submit as ds  # noqa: E402


# --- helpers ---------------------------------------------------------------

EXPERIMENTAL = {
    "kernel": "canivel/arc3-duck-lane-a",
    "version": 7,
    "file": "submission.parquet",
    "message": "EXPERIMENTAL lane (a) P1 state-externalisation draw 1",
    "preflight_mode": "trusted-fork",
    "upstream": "canivel/arc3-duck-lane-a",
}


def _read_log(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    return [json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _events(log_path: Path, name: str) -> list[dict]:
    return [r for r in _read_log(log_path) if r.get("event") == name]


def _skips(log_path: Path) -> list[str]:
    return [r["skip"] for r in _read_log(log_path) if "skip" in r]


def _queue(queue_path: Path) -> dict:
    return json.loads(queue_path.read_text(encoding="utf-8"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Hermetic daemon environment. Returns a small handle object."""
    queue_path = tmp_path / "submission_queue.json"
    log_path = tmp_path / "runs" / "submission_log.jsonl"
    queue_path.write_text(json.dumps({"pending": [], "history": []}), encoding="utf-8")

    # Satisfy the audit-trail gate with a same-day (LOCAL date) entry.
    (tmp_path / "ITERATION_LOG.md").write_text(
        f"### {dt.date.today().isoformat()}\nhermetic test entry\n", encoding="utf-8")

    monkeypatch.setattr(ds, "ROOT", tmp_path)
    monkeypatch.setattr(ds, "QUEUE", queue_path)
    monkeypatch.setattr(ds, "LOG", log_path)

    # No CLI, no network, no preflight subprocess.
    monkeypatch.setattr(ds, "_kaggle_cli", lambda: "kaggle")
    monkeypatch.setattr(ds, "already_submitted_today", lambda kaggle: False)
    monkeypatch.setattr(ds, "run_preflight", lambda item: {"verdict": "ALLOW", "checks": []})

    submitted: list[dict] = []

    def fake_submit(kaggle, item):
        submitted.append(item)
        return True, "ok", ""

    monkeypatch.setattr(ds, "submit", fake_submit)

    class Env:
        pass

    e = Env()
    e.root, e.queue, e.log, e.submitted, e.mp = tmp_path, queue_path, log_path, submitted, monkeypatch
    return e


def _set_pending(env, pending: list[dict]) -> None:
    q = _queue(env.queue)
    q["pending"] = pending
    env.queue.write_text(json.dumps(q, indent=2), encoding="utf-8")


# --- the fallback entry itself ---------------------------------------------

def test_fallback_entry_is_the_frozen_fork():
    e = ds.eternal_fallback_entry()
    assert e["kernel"] == "canivel/arc3-duck-repro"
    assert e["version"] == 3
    assert e["file"] == "submission.parquet"
    assert e["preflight_mode"] == "trusted-fork"
    assert e["upstream"] == "jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner"
    assert e["auto_refill"] is True


def test_fallback_message_marked_autorefill_with_utc_date():
    msg = ds.eternal_fallback_entry()["message"]
    assert msg.startswith("AUTO-REFILL ")
    assert dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d") in msg


# --- 1. submit -> autorefill ------------------------------------------------

def test_submit_then_autorefill(env):
    _set_pending(env, [ds.eternal_fallback_entry(note="seed")])
    assert ds.main() == 0

    q = _queue(env.queue)
    assert len(env.submitted) == 1
    assert len(q["history"]) == 1
    # queue did NOT drain: exactly one fresh fallback is armed for tomorrow
    assert len(q["pending"]) == 1
    assert q["pending"][0]["kernel"] == ds.FALLBACK_KERNEL
    assert q["pending"][0]["auto_refill"] is True
    assert ds.AUTOREFILL_TAG in q["pending"][0]["message"]

    ev = _events(env.log, "queue_autorefill")
    assert len(ev) == 1 and ev[0]["phase"] == "post-submit"


def test_submit_of_experimental_head_still_refills_when_it_drains(env):
    _set_pending(env, [dict(EXPERIMENTAL)])
    assert ds.main() == 0
    q = _queue(env.queue)
    assert env.submitted[0]["kernel"] == EXPERIMENTAL["kernel"]
    assert [p["kernel"] for p in q["pending"]] == [ds.FALLBACK_KERNEL]


# --- 2. wake with an empty queue -------------------------------------------

def test_wake_with_empty_queue_self_refills_and_fires(env):
    _set_pending(env, [])
    assert ds.main() == 0

    # It FIRED — no lost draw, and no no-op skip was logged.
    assert len(env.submitted) == 1
    assert env.submitted[0]["kernel"] == ds.FALLBACK_KERNEL
    assert "queue-empty" not in _skips(env.log)

    phases = [e["phase"] for e in _events(env.log, "queue_autorefill")]
    assert phases == ["pre-fire", "post-submit"]
    assert len(_queue(env.queue)["pending"]) == 1


def test_wake_empty_but_already_submitted_still_arms_the_queue(env):
    """The already-submitted-today guard is UNCHANGED — but we still leave the
    queue armed so tomorrow's wake has a head."""
    env.mp.setattr(ds, "already_submitted_today", lambda kaggle: True)
    _set_pending(env, [])
    assert ds.main() == 0

    assert env.submitted == []                      # max 1 submission/day held
    assert "already-submitted-today" in _skips(env.log)
    assert len(_queue(env.queue)["pending"]) == 1


# --- 3. does not fire twice in one day -------------------------------------

def test_autorefill_does_not_fire_twice_in_one_day(env):
    _set_pending(env, [ds.eternal_fallback_entry(note="seed")])
    assert ds.main() == 0
    first = _events(env.log, "queue_autorefill")
    assert len(first) == 1

    # Second wake the same day: the day guard (already-submitted-today) holds
    # and the queue is non-empty, so nothing is appended and nothing is logged.
    env.mp.setattr(ds, "already_submitted_today", lambda kaggle: True)
    assert ds.main() == 0

    assert len(env.submitted) == 1
    assert len(_events(env.log, "queue_autorefill")) == 1
    assert len(_queue(env.queue)["pending"]) == 1


def test_autorefill_is_a_noop_when_queue_non_empty():
    q = {"pending": [dict(EXPERIMENTAL)], "history": []}
    # QUEUE is never touched because the guard short-circuits first.
    assert ds.autorefill(q, phase="unit") is False
    assert q["pending"] == [EXPERIMENTAL]


# --- 4. never clobbers a pending EXPERIMENTAL entry -------------------------

def test_autorefill_does_not_clobber_pending_experimental(env):
    _set_pending(env, [dict(EXPERIMENTAL)])
    env.mp.setattr(ds, "already_submitted_today", lambda kaggle: True)
    assert ds.main() == 0

    q = _queue(env.queue)
    assert q["pending"] == [EXPERIMENTAL]           # byte-for-byte untouched
    assert _events(env.log, "queue_autorefill") == []


def test_experimental_tail_survives_a_fire(env):
    _set_pending(env, [ds.eternal_fallback_entry(note="seed"), dict(EXPERIMENTAL)])
    assert ds.main() == 0

    q = _queue(env.queue)
    assert q["pending"] == [EXPERIMENTAL]           # tail promoted, not overwritten
    assert _events(env.log, "queue_autorefill") == []


def test_blocked_experimental_head_is_not_replaced_by_fallback(env):
    """Preflight BLOCK must leave the entry at the head for tomorrow — the
    refill must not paper over a gate failure by swapping in the frozen fork."""
    _set_pending(env, [dict(EXPERIMENTAL)])
    env.mp.setattr(ds, "run_preflight", lambda item: {
        "verdict": "BLOCK", "checks": [{"check": "S1", "status": "FAIL"}]})
    assert ds.main() == 1

    q = _queue(env.queue)
    assert q["pending"] == [EXPERIMENTAL]
    assert env.submitted == []
    assert "preflight-blocked" in _skips(env.log)
    assert _events(env.log, "queue_autorefill") == []


# --- 5. the log event ------------------------------------------------------

def test_autorefill_log_event_shape(env):
    _set_pending(env, [])
    ds.main()
    ev = _events(env.log, "queue_autorefill")[0]
    assert set(ev) >= {"t", "event", "phase", "kernel", "version", "date", "queue_remaining"}
    assert ev["kernel"] == ds.FALLBACK_KERNEL
    assert ev["version"] == ds.FALLBACK_VERSION
    assert ev["date"] == dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    assert ev["queue_remaining"] == 1


def test_real_queue_and_log_untouched(env):
    """Guard against the tests ever regressing into the live campaign state."""
    repo = Path(__file__).resolve().parents[1]
    before = (repo / "submission_queue.json").read_bytes()
    n_log = len((repo / "runs" / "submission_log.jsonl").read_bytes())
    _set_pending(env, [])
    ds.main()
    assert (repo / "submission_queue.json").read_bytes() == before
    assert len((repo / "runs" / "submission_log.jsonl").read_bytes()) == n_log


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
