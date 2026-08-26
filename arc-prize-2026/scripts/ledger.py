#!/usr/bin/env python
"""Canonical frozen-fork draw ledger.

Until 2026-08-10 the ledger headline (n / mean / s) existed ONLY as prose carried
forward by hand in ITERATION_LOG.md and in each night's submission message. That is
how a wrong trailing-4 (0.9025, when the true value was 0.8275) propagated from the
08-09 log into the 08-10 daily brief. This script makes the statistic reproducible
from the Kaggle API, which is the only ground truth we have.

Ledger membership rule (explicit, so it stops being folklore):
  a submission counts as a frozen-fork DRAW iff its description contains
  "frozen-fork filler" AND it has a COMPLETE public score.
Everything else (capability arms, screens, early-campaign submissions) is excluded --
the ledger measures the *artifact-noise distribution* of one byte-identical artifact,
not our capability.

Usage:
    uv run python scripts/ledger.py            # re-derive, print, AND write runs/ledger.json
    uv run python scripts/ledger.py --no-write # print only, leave the file alone
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics as st
import subprocess
import sys

COMP = "arc-prize-2026-arc-agi-3"
DEFAULT_JSON = "runs/ledger.json"
# The ledger is the set of byte-identical resubmits of the Milestone-1 winner
# (Cottaar/Tufa duck-harness repro). The message text was standardised to
# "frozen-fork filler" on 2026-07-18; the five earlier members carry the older
# "Frozen-fork sigma draw #N" wording, and the origin draw (2026-07-08) carries
# neither. Matching only the modern tag yields n=22 and silently drops those five.
MEMBERSHIP_TAGS = ("frozen-fork filler", "frozen-fork sigma draw")
ORIGIN_DRAW = "2026-07-08"  # "Q2: Cottaar/Tufa duck-harness EXACT repro (Milestone-1 winner...)"
# kaggle 2.2.x drops kernel logs; 2.0.0 is the pinned CLI for this campaign.
CLI = ["uvx", "--from", "kaggle==2.0.0", "kaggle"]


def fetch_rows() -> list[dict]:
    out = subprocess.run(
        CLI + ["competitions", "submissions", COMP, "-v"],
        capture_output=True,
        text=True,
        encoding="cp1252",
        errors="replace",
    ).stdout
    # The pinned CLI prints an upgrade warning above the CSV header; find the real header.
    lines = out.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("fileName,"))
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def draws(rows: list[dict]) -> list[tuple[str, float]]:
    """Newest-first list of (date, publicScore) for ledger members."""
    got = []
    for r in rows:
        desc = (r.get("description") or "").lower()
        score = r.get("publicScore") or ""
        date = r["date"][:10]
        member = any(t in desc for t in MEMBERSHIP_TAGS) or date == ORIGIN_DRAW
        if member and "COMPLETE" in (r.get("status") or "") and score:
            got.append((date, float(score)))
    return got


def record(d: list[tuple[str, float]]) -> dict:
    v = [s for _, s in d]
    rec = {
        "n": len(v),
        "mean": round(st.mean(v), 4),
        "s": round(st.stdev(v), 4) if len(v) > 1 else None,
        "latest_date": d[0][0] if d else None,
        "latest": v[0] if v else None,
        "max": max(v) if v else None,
        "min": min(v) if v else None,
        "trailing4": round(sum(v[:4]) / 4, 4) if len(v) >= 4 else None,
        "trailing4_prev": round(sum(v[1:5]) / 4, 4) if len(v) >= 5 else None,
        "draws_newest_first": v,
    }
    if rec["s"]:
        # z of the latest draw against the record EXCLUDING it (the honest comparison:
        # a draw is not scored against a mean it is already inside).
        prior = v[1:]
        if len(prior) > 1:
            rec["prior_n"] = len(prior)
            rec["prior_mean"] = round(st.mean(prior), 4)
            rec["prior_s"] = round(st.stdev(prior), 4)
            rec["z_latest"] = round((v[0] - st.mean(prior)) / st.stdev(prior), 2)
        # Sealed mean-of-4 promotion bar, one-sided alpha=0.05. The sqrt(1/4 + 1/n)
        # term is load-bearing: the record mean is ESTIMATED, not known, so a plain
        # s/sqrt(4) understates the bar (1.0661 vs 1.0801 at n=27 -- i.e. a naive
        # bar would promote arms the sealed arithmetic rejects).
        from scipy.stats import t as _t
        rec["promotion_bar_mean_of_4"] = round(
            rec["mean"] + _t.ppf(0.95, rec["n"] - 1) * rec["s"] * (0.25 + 1 / rec["n"]) ** 0.5,
            4,
        )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    # Persist by DEFAULT. This used to be opt-in via --json, and on 2026-08-12 the
    # morning check invoked it without the flag: stdout showed the fresh n=29 record
    # while runs/ledger.json silently stayed at n=28, so the sealed mean-of-4
    # promotion bar read 1.0848 instead of 1.0876. Correct-looking output over a
    # stale file is the worst failure mode this script has, so writing is no longer
    # something a caller has to remember.
    ap.add_argument("--json", default=DEFAULT_JSON,
                    help=f"write the record here (default: {DEFAULT_JSON})")
    ap.add_argument("--no-write", action="store_true",
                    help="print the record without touching the file on disk")
    a = ap.parse_args()

    d = draws(fetch_rows())
    if not d:
        print("NO LEDGER MEMBERS FOUND -- membership rule may have drifted", file=sys.stderr)
        return 1
    rec = record(d)

    print(f"frozen-fork ledger  n={rec['n']}  mean={rec['mean']}  s={rec['s']}")
    print(f"  latest      {rec['latest_date']}  {rec['latest']}   z={rec.get('z_latest')} "
          f"(vs prior n={rec.get('prior_n')} {rec.get('prior_mean')}/{rec.get('prior_s')})")
    print(f"  trailing-4  {rec['trailing4']}   (prev {rec['trailing4_prev']})")
    print(f"  range       {rec['min']} .. {rec['max']}")
    print(f"  mean-of-4 promotion bar (a=0.05)  {rec['promotion_bar_mean_of_4']}")

    if a.no_write:
        print("  (--no-write: runs/ledger.json NOT updated)")
    else:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
