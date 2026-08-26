"""Retain a dated full-leaderboard CSV so per-team draw-level deltas become computable.

WHY THIS EXISTS (2026-08-19). The campaign has tracked a "step signature" — teams gaining
>= +1.0 on a single draw — for days, but no full-leaderboard snapshot was ever retained.
Every such claim was therefore derived from same-day aggregates and recollection, not from a
diffable artifact: the analysis had no instrument. This writes one snapshot per day into
runs/lb/ and, when a prior snapshot exists, prints the per-team deltas that were previously
unmeasurable.

    uv run python scripts/lb_snapshot.py            # fetch + retain + diff vs newest prior
    uv run python scripts/lb_snapshot.py --diff-only  # no fetch; diff the two newest on disk

Deltas are DESCRIPTIVE. A movement is not a method: every step on this board remains UNKNOWN
and undisclosed, and `LastSubmissionDate` is LATEST while `Score` is BEST, so a date on a row
does NOT date the score it sits beside.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import zipfile
from pathlib import Path

COMP = "arc-prize-2026-arc-agi-3"
LB_DIR = Path(__file__).resolve().parent.parent / "runs" / "lb"


def snapshots() -> list[Path]:
    return sorted(LB_DIR.glob("*publicleaderboard*.csv"))


def fetch() -> Path | None:
    LB_DIR.mkdir(parents=True, exist_ok=True)
    before = set(snapshots())
    cmd = ["uvx", "--from", "kaggle==2.0.0", "kaggle", "competitions",
           "leaderboard", COMP, "--download", "-p", str(LB_DIR)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FETCH FAILED (rc={r.returncode}): {r.stderr.strip()[:300]}", file=sys.stderr)
        return None
    for z in LB_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(LB_DIR)
        z.unlink()
    new = set(snapshots()) - before
    return sorted(new)[-1] if new else (snapshots()[-1] if snapshots() else None)


def load(p: Path) -> dict[str, dict]:
    with p.open(encoding="utf-8-sig") as fh:
        return {r["TeamId"]: r for r in csv.DictReader(fh)}


def diff(prev: Path, cur: Path, threshold: float) -> None:
    a, b = load(prev), load(cur)
    print(f"\nDELTA  {prev.name}\n    -> {cur.name}")
    print(f"  teams {len(a)} -> {len(b)}")
    moved = []
    for tid, row in b.items():
        if tid not in a:
            continue
        d = float(row["Score"]) - float(a[tid]["Score"])
        if d > 1e-9:
            drew = row["SubmissionCount"] != a[tid]["SubmissionCount"]
            moved.append((d, row, int(row["SubmissionCount"]) - int(a[tid]["SubmissionCount"]), drew))
    moved.sort(key=lambda t: -t[0])
    carried = len(set(a) & set(b))
    print(f"  teams that GAINED: {len(moved)} of {carried} carried over "
          f"({100.0 * len(moved) / carried:.1f}%)" if carried else "  no carried-over teams")
    if moved:
        med = sorted(d for d, *_ in moved)[len(moved) // 2]
        print(f"  median gain among gainers: {med:+.4f}")
    print(f"\n  STEPS >= +{threshold:.2f} (single-window jumps; method UNKNOWN unless disclosed):")
    steps = [m for m in moved if m[0] >= threshold]
    if not steps:
        print("    (none)")
    for d, row, dsubs, _ in steps:
        flag = "  <-- ONE draw" if dsubs == 1 else ""
        print(f"    {d:+.2f}  {row['TeamName'][:34]:<34} -> {row['Score']:>5}  "
              f"draws_used={dsubs} lifetime={row['SubmissionCount']}{flag}")
    print("\n  Reminder: movement is not method. Do not infer mechanism from a step.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff-only", action="store_true", help="do not fetch; diff two newest on disk")
    ap.add_argument("--threshold", type=float, default=1.0, help="step-signature threshold")
    args = ap.parse_args()

    cur = snapshots()[-1] if args.diff_only and snapshots() else fetch()
    if cur is None:
        print("no snapshot available", file=sys.stderr)
        return 1
    print(f"snapshot: {cur.name}  ({len(load(cur))} teams)")
    prior = [p for p in snapshots() if p != cur]
    if not prior:
        print("\nNo prior snapshot — deltas are not computable yet. This run establishes the "
              "baseline; the next run is the first that can measure a step.")
        return 0
    diff(prior[-1], cur, args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
