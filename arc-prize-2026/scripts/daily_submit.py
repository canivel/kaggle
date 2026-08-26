"""Daily ARC-AGI-3 submission daemon.

Reads `submission_queue.json` at the repo root. Pops the next pending entry
and submits it to Kaggle. Idempotent: skips when a submission already
landed today (UTC). On submit failure, leaves the entry in the queue
so the next day retries — no entries are silently dropped.

Runs unattended via Windows Task Scheduler. The campaign deadline is
2026-11-02, so this needs to be reliable for ~4.5 months.

Adding work to the queue: append a dict to `pending`. Schema:
  {
    "kernel":  "<owner>/<slug>",
    "version": <int>,
    "file":    "submission.parquet",   # optional, default shown
    "message": "<text>"
  }

ETERNAL FALLBACK / AUTO-REFILL (2026-08-12): the queue can never be left
empty. The daemon self-arms the frozen-fork fallback entry in two places:
  * defensively, on wake, if `pending` is already empty (then it fires that
    fallback rather than logging a no-op skip — losing a daily draw is
    strictly worse than re-submitting the frozen fork); and
  * after a successful submit, if popping the head drained `pending`.
Each arming appends one `{"event": "queue_autorefill", ...}` record to
runs/submission_log.jsonl so the morning check can see it happened.

Logs every fire — submit or skip — to runs/submission_log.jsonl.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / "submission_queue.json"
LOG = ROOT / "runs" / "submission_log.jsonl"
COMP = "arc-prize-2026-arc-agi-3"

# --- eternal fallback ------------------------------------------------------
# ARM 0 (coordinator standing order, 2026-08-21, attribution_2026-08-21.md): the
# DEFAULT nightly head is the FIELD-FLOOR config -- certified SIGNAL (lc 28),
# scored 1.59 as a typical draw of ~N(1.6, 0.2), byte-faithful trusted fork of
# the field's board-verified 2.23 artifact. The frozen June-30 duck-repro fork
# (mean 0.93, hardcoded old mount paths) is FULLY RETIRED as fallback; pathsafe
# exists as deeper insurance and is queued explicitly when needed.
# THE single definition -- never inline this dict.
FALLBACK_KERNEL = "canivel/arc3-q38-field-eval"
FALLBACK_VERSION = 1
FALLBACK_FILE = "submission.parquet"
FALLBACK_UPSTREAM = "foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b"
AUTOREFILL_TAG = "AUTO-REFILL"


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def utc_date() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def eternal_fallback_entry(note: str = "") -> dict:
    """The Arm-0 field-floor fallback queue entry, auto-stamped and clearly marked.

    Single source of truth for the fallback — `scripts/queue.py refill` and
    both auto-refill call sites in this module all go through here.
    """
    msg = (
        f"{AUTOREFILL_TAG} {utc_date()} — FIELD-FLOOR filler (Arm 0 default nightly head; "
        f"auto-armed by scripts/daily_submit.py because the queue was empty). "
        f"Replace with the day's build if one clears the promotion gates."
    )
    if note:
        msg += f" [{note}]"
    return {
        "kernel": FALLBACK_KERNEL,
        "version": FALLBACK_VERSION,
        "file": FALLBACK_FILE,
        "message": msg,
        "preflight_mode": "trusted-fork",
        "upstream": FALLBACK_UPSTREAM,
        "auto_refill": True,
    }


def log(rec: dict) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"t": utcnow_iso(), **rec}
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


def _kaggle_cli() -> str | None:
    return shutil.which("kaggle")


def autorefill(q: dict, phase: str) -> bool:
    """Arm the eternal fallback iff `pending` is empty. Returns True if armed.

    The empty-check is the whole guard: at most one auto entry can ever sit in
    the queue, so a second daemon wake on the same day (or any hand-queued
    EXPERIMENTAL entry sitting at the head) is a no-op. Writes the queue file
    and logs a `queue_autorefill` event so the defect stays observable.
    """
    if q.get("pending"):
        return False
    entry = eternal_fallback_entry(note=phase)
    q["pending"] = [entry]
    QUEUE.write_text(json.dumps(q, indent=2), encoding="utf-8")
    log({"event": "queue_autorefill", "phase": phase,
         "kernel": entry["kernel"], "version": entry["version"],
         "date": utc_date(), "queue_remaining": 1})
    return True


def run_preflight(item: dict) -> dict:
    """Run scripts/preflight.py for a queue entry; returns its JSON report."""
    pf_cmd = [sys.executable, str(ROOT / "scripts" / "preflight.py"),
              "--kernel", item["kernel"], "--json-only"]
    if item.get("preflight_mode") == "trusted-fork" and item.get("upstream"):
        pf_cmd += ["--mode", "trusted-fork", "--upstream", item["upstream"]]
    pf = subprocess.run(pf_cmd, capture_output=True, text=True, timeout=300)
    return json.loads(pf.stdout.strip().splitlines()[-1])


def already_submitted_today(kaggle: str) -> bool:
    """True if the most recent submission row's date prefix matches today (UTC)."""
    try:
        r = subprocess.run(
            [kaggle, "competitions", "submissions", COMP],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as e:
        log({"warn": "submissions-list-failed", "err": repr(e)})
        return False
    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    for line in r.stdout.splitlines():
        if line.startswith("submission") and today in line:
            return True
    return False


def _submission_rows(kaggle: str) -> set[str]:
    """Full data rows from the CSV submissions list (-v): stable columns, full
    description field, no human-table truncation. Empty set on any failure.

    2026-08-21 00:31Z incident: the old check compared only the FIRST line
    starting with the literal "submission" -- a CLI-output-format dependency
    that returns "" under other formats/lag, making before==after ("" == "")
    a FALSE NEGATIVE on a successful submit and leaving the fired head queued
    (double-fire landmine). Rows + set-difference + retries instead.
    """
    try:
        r = subprocess.run(
            [kaggle, "competitions", "submissions", COMP, "-v"],
            capture_output=True, text=True, timeout=120,
        )
    except Exception:
        return set()
    rows = set()
    for line in r.stdout.splitlines():
        line = line.strip()
        # data rows carry a date; skip headers/banners/warnings
        if line and ("20" in line and "-" in line and "," in line) and not line.lower().startswith("ref,"):
            rows.add(line)
    return rows


def submit(kaggle: str, item: dict) -> tuple[bool, str, str]:
    before = _submission_rows(kaggle)
    cmd = [
        kaggle, "competitions", "submit", COMP,
        "-k", item["kernel"],
        "-v", str(item["version"]),
        "-f", item.get("file", "submission.parquet"),
        "-m", item["message"],
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    out, err = r.stdout, r.stderr
    exit_ok = r.returncode == 0
    marker_ok = "successfully submitted" in (out + err).lower()  # nit-1 (review 08-21): no 'unsuccessful' false-match
    msg_token = item["message"][:60]
    # ID/row-based verification with retries: the submissions list can LAG the
    # accepted submission (observed 2026-08-21), so poll up to ~75s before
    # concluding anything from it.
    for attempt in range(6):
        new_rows = _submission_rows(kaggle) - before
        if any(msg_token in row for row in new_rows):
            return True, out[-600:], err[-600:]
        if new_rows and exit_ok and marker_ok:
            # a new row appeared and the CLI reported success; the description
            # may be truncated/escaped differently -- corroborated enough.
            return True, out[-600:], err[-600:]
        if attempt < 5:
            time.sleep(15)
    if exit_ok and marker_ok:
        # The CLI said success but the list never showed it. Trust the CLI so
        # the head is POPPED (a queued already-submitted head is the proven
        # double-fire landmine); log loudly so the morning check verifies the
        # submissions list by hand.
        log({"warn": "submit-listlag-assumed-ok", "kernel": item["kernel"],
             "detail": "CLI exit 0 + success marker, but submissions list "
                       "never showed a new row within ~75s"})
        return True, out[-600:], err[-600:]
    return False, out[-600:], err[-600:]


def _entry_gates(cand: dict):
    """Per-entry gates. None = healthy (fire it); dict = why blocked."""
    # EVIDENCE GATE — experimental kernels must carry local-eval evidence.
    # Queue entries with "requires_evidence": true are refused unless
    # "evidence" points to an existing results file with >=3 seeds recorded.
    # (Frozen known-good builds and byte-identical trusted forks are exempt.)
    _why = None
    if cand.get("requires_evidence"):
        ev = cand.get("evidence")
        ev_path = (ROOT / ev) if ev else None
        ok_ev = False
        if ev_path and ev_path.exists():
            try:
                ev_data = json.loads(ev_path.read_text(encoding="utf-8"))
                seeds = ev_data.get("seeds") or ev_data.get("n_seeds")
                n_seeds = len(seeds) if isinstance(seeds, list) else int(seeds or 0)
                ok_ev = n_seeds >= 3
            except Exception:
                ok_ev = False
        if not ok_ev:
            _why = {"gate": "evidence", "evidence": cand.get("evidence")}
            log({"skip": "evidence-gate-blocked", "kernel": cand["kernel"],
                 "evidence": cand.get("evidence")})
            return _why

    # MANDATORY PREFLIGHT — blocks structural drift from arc3-baseline.
    # See scripts/preflight.py for the checks. If it BLOCKs, we do NOT
    # submit; entry stays in queue for the next day after the user fixes.
    # Queue entries may set "preflight_mode": "trusted-fork" plus
    # "upstream": "<owner>/<slug>" for unmodified forks of proven public
    # kernels (baseline structural checks don't apply to foreign harnesses).
    try:
        pf_report = run_preflight(cand)
    except Exception as e:
        _why = {"gate": "preflight-runner", "err": repr(e)}
        log({"skip": "preflight-runner-failed", "kernel": cand.get("kernel"), "err": repr(e)})
        return _why
    if pf_report.get("verdict") == "BLOCK":
        _why = {"gate": "preflight",
                "fails": [c for c in pf_report["checks"] if c["status"] == "FAIL"]}
        log({
            "skip": "preflight-blocked",
            "kernel": cand["kernel"],
            "fails": [c for c in pf_report["checks"] if c["status"] == "FAIL"],
        })
        return _why
    return None


def main() -> int:
    kaggle = _kaggle_cli()
    if not kaggle:
        log({"skip": "kaggle-cli-missing"})
        return 0
    if not QUEUE.exists():
        log({"skip": "queue-file-missing"})
        return 0
    q = json.loads(QUEUE.read_text(encoding="utf-8"))
    pending = q.get("pending", [])
    if not pending:
        # DEFENSIVE REFILL (2026-08-12): woke with a drained queue. Do NOT
        # log a no-op skip — a missed fire is an unrecoverable lost daily
        # draw. Self-arm the frozen fork and fall through to fire it.
        autorefill(q, phase="pre-fire")
        pending = q["pending"]
    if already_submitted_today(kaggle):
        log({"skip": "already-submitted-today"})
        return 0

    # AUDIT-TRAIL GATE (panel round8 N12, 2026-07-13) — refuse to fire unless
    # ITERATION_LOG.md contains a same-day (local date) entry: the one action
    # taken outside the log must never be a scored submission.
    try:
        import datetime as _dt
        today = _dt.date.today().isoformat()
        log_text = (ROOT / "ITERATION_LOG.md").read_text(encoding="utf-8")
        if f"### {today}" not in log_text:
            log({"skip": "audit-trail-gate-blocked",
                 "reason": f"no ITERATION_LOG.md entry '### {today}'",
                 "kernel": pending[0].get("kernel")})
            return 1
    except FileNotFoundError:
        log({"skip": "audit-trail-gate-blocked", "reason": "ITERATION_LOG.md missing"})
        return 1

    # FALL-THROUGH (2026-08-20 incident, coordinator-sequenced deploy): a
    # preflight- or evidence-blocked entry must not silently spend the night's
    # draw. Try each pending entry in order; the FIRST healthy entry fires.
    # Blocked entries STAY in the queue (loudly logged) for the user to fix.
    # If ALL entries block, write an alarm artifact the morning check picks up
    # and exit non-zero -- never submit nothing silently.
    blocked_all = []
    item = None
    for cand_idx, cand in enumerate(pending):
        verdict = _entry_gates(cand)
        if verdict is None:
            item = cand
            item_idx = cand_idx
            if cand_idx > 0:
                log({"event": "fall-through", "skipped_blocked": cand_idx,
                     "firing": cand["kernel"]})
            break
        blocked_all.append({"kernel": cand.get("kernel"), "why": verdict})
    if item is None:
        alarm = {"t": utcnow_iso(), "event": "ALL-ENTRIES-BLOCKED",
                 "blocked": blocked_all, "n_pending": len(pending)}
        log(alarm)
        (ROOT / "runs" / f"submit_alarm_{utc_date()}.json").write_text(
            json.dumps(alarm, indent=2), encoding="utf-8")
        return 1

    ok, out, err = submit(kaggle, item)
    rec = {"submit": item, "ok": ok, "stdout_tail": out, "stderr_tail": err}
    if ok:
        q["pending"] = pending[:item_idx] + pending[item_idx + 1:]
        q.setdefault("history", []).append({**item, "submitted_at": utcnow_iso()})
        QUEUE.write_text(json.dumps(q, indent=2), encoding="utf-8")
        rec["queue_remaining"] = len(q["pending"])
        log(rec)
        # AUTO-REFILL: the head we just fired may have drained the queue.
        # Re-arm the eternal fallback immediately so tomorrow's wake always
        # has something to fire (2026-08-12 standing defect).
        autorefill(q, phase="post-submit")
        return 0
    log(rec)
    return 1


if __name__ == "__main__":
    sys.exit(main())
