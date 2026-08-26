#!/usr/bin/env python
"""Scheduler EV re-derivation under TRUE competition scoring semantics.

Semantics (code-verified 2026-07-13, daily_brief_2026-07-13.md #1c):
  - one run per game_id (arc_agi/api.py:417, competition mode)
  - actions pool within the run; the wasted pre-restart actions land in the
    FIRST cleared level's bucket (scorecard.py:430); later levels count clean
  - a RESET costs +1 action (scorecard.py:655)

Policy simulated: restart if lc==0 at T actions since attempt start, cap K
restarts, then park. Attempts are drawn exchangeably from the same game's
null10 runs (cross-seed exchangeability validated in path_forward_v3 SE2).

Exact enumeration over (r1) for the no-trigger branch and (r1,r2), (r1,r2,r3)
for restart branches -- no Monte Carlo.

Wall-truncation sensitivity: recovered attempts' value scaled by disc in
{1.0, 0.6, 0.365} (SE2 measured disc(90)=0.365 as the harshest reading).

Outputs runs/sched_pooled_ev.json and prints a markdown table.
"""
from __future__ import annotations

import json
import statistics as S
from collections import defaultdict
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NULL = ROOT / "runs" / "null10" / "merged_null_benchmark.json"

T = 90          # restart trigger (actions since attempt start)
CAP = 2         # max restarts
RESET_COST = 1  # a RESET increments the run action counter


def level_scores(lc, apl, base, n, l1_prefix=0):
    """Per-level scores with optional pooled prefix added to the first
    cleared level's action count (the restart tax)."""
    out = []
    for i in range(n):
        a = apl[i] if apl and i < len(apl) else 0
        if lc is not None and i < lc and a and a > 0:
            eff = a + (l1_prefix if i == 0 else 0)
            out.append(min(115.0, (base[i] / eff) ** 2 * 100))
        else:
            out.append(0.0)
    return out


def rhae(lc, apl, base, n, l1_prefix=0):
    w = [i + 1 for i in range(n)]
    ls = level_scores(lc, apl, base, n, l1_prefix)
    tot = sum(w)
    s = sum(wi * li for wi, li in zip(w, ls))
    sw = sum(wi for wi, li in zip(w, ls) if li > 0)
    return min(s / tot, sw / tot * 100)


def t_first_clear(run):
    """Actions consumed at first L1 clear; None if never."""
    lc = run.get("levels_completed") or 0
    if lc < 1:
        return None
    return run["actions_per_level"][0]


def main():
    null = json.loads(NULL.read_text())
    games = defaultdict(list)
    base = {}
    nlv = {}
    for r in null:
        g = r["game_id"][:4]
        games[g].append(r)
        base[g] = r["base_actions_per_level"]
        nlv[g] = r["number_of_levels"]

    discs = [1.0, 0.6, 0.365]
    rows = []
    for g, runs in sorted(games.items()):
        b, n = base[g], nlv[g]
        null_ev = S.mean(rhae(r.get("levels_completed"), r.get("actions_per_level"), b, n) for r in runs)
        early = [r for r in runs if (t_first_clear(r) or 10**9) <= T]
        stuck = [r for r in runs if (t_first_clear(r) or 10**9) > T]
        p_stuck = len(stuck) / len(runs)

        # value of a recovered attempt k (k=1: prefix T+1 reset, k=2: 2T+2)
        def rec_val(r, k, disc):
            prefix = k * (T + RESET_COST)
            return disc * rhae(r.get("levels_completed"), r.get("actions_per_level"), b, n, l1_prefix=prefix)

        sched_ev = {}
        park_only = S.mean(
            (rhae(r.get("levels_completed"), r.get("actions_per_level"), b, n) if (t_first_clear(r) or 10**9) <= T else 0.0)
            for r in runs
        )
        for disc in discs:
            # exact enumeration: r1 uniform; if stuck -> r2 uniform; if r2 early -> recovered val
            # (an r2 that is itself stuck at T restarts again -> r3; else parked -> 0)
            total = 0.0
            N = len(runs)
            for r1 in runs:
                if (t_first_clear(r1) or 10**9) <= T:
                    total += rhae(r1.get("levels_completed"), r1.get("actions_per_level"), b, n) * N * N
                    continue
                for r2 in runs:
                    if (t_first_clear(r2) or 10**9) <= T:
                        total += rec_val(r2, 1, disc) * N
                        continue
                    for r3 in runs:
                        if (t_first_clear(r3) or 10**9) <= T:
                            total += rec_val(r3, 2, disc)
                        # else parked -> 0
            sched_ev[disc] = total / (N ** 3)
        rows.append({
            "game": g, "null_ev": null_ev, "p_stuck90": p_stuck,
            "n_early": len(early),
            "park_only": park_only,
            **{f"sched_disc{d}": sched_ev[d] for d in discs},
        })

    # aggregate
    agg = {
        "null_ev": S.mean(r["null_ev"] for r in rows),
        "park_only": S.mean(r["park_only"] for r in rows),
        **{f"sched_disc{d}": S.mean(r[f"sched_disc{d}"] for r in rows) for d in discs},
    }
    out = {"trigger": T, "cap": CAP, "rows": rows, "aggregate": agg}
    (ROOT / "runs" / "sched_pooled_ev.json").write_text(json.dumps(out, indent=1))

    print(f"{'game':6}{'null':>8}{'stuck%':>8}{'park0':>8}{'d=1.0':>8}{'d=.6':>8}{'d=.365':>8}")
    for r in rows:
        print(f"{r['game']:6}{r['null_ev']:8.3f}{100*r['p_stuck90']:8.0f}{r['park_only']:8.3f}"
              f"{r['sched_disc1.0']:8.3f}{r['sched_disc0.6']:8.3f}{r['sched_disc0.365']:8.3f}")
    print("-" * 54)
    print(f"{'MEAN':6}{agg['null_ev']:8.4f}{'':8}{agg['park_only']:8.4f}"
          f"{agg['sched_disc1.0']:8.4f}{agg['sched_disc0.6']:8.4f}{agg['sched_disc0.365']:8.4f}")
    print(f"\ndelta vs null: park-only {agg['park_only']-agg['null_ev']:+.4f}; "
          f"sched d=1.0 {agg['sched_disc1.0']-agg['null_ev']:+.4f}; "
          f"d=0.6 {agg['sched_disc0.6']-agg['null_ev']:+.4f}; "
          f"d=0.365 {agg['sched_disc0.365']-agg['null_ev']:+.4f}")


if __name__ == "__main__":
    main()
