"""C3 (prereg amendment 2026-07-23) — A17 symmetric error model.

Pre-registers BOTH error probabilities of the A17 screen rule

    GO iff CAPABILITY  [ Sigma_g (72B per-game MAX lc over k seeds) >= Sigma_g (27B per-game MAX lc) + 2 = 8 ]
           AND ( ACTION-PARITY [ Sigma N72 >= 0.90 * Sigma N27 ]
                 OR THROTTLED  [ Sigma_g (72B per-game MAX lc over k seeds) >= Sigma null_adj(rho) + 1 ] )

under the VERIFIED draw distribution: the 4 certified 27B rows
(war_eval_v1/v2/v3, w0_eval_s1) for the 4 screen games (ft09, sb26, lp85,
vc33), as frozen in runs/a17_repair/per_seed_table.json (recomputed 07-22 from
raw benchmark.json; verified against runs/verify_2026-07-21).

Model (all choices pre-registered here, before any 72B number is observed):
- 72B pseudo-seed draw = one full certified ROW drawn uniformly (row-wise,
  preserving within-seed correlation across games), transformed to the
  throttled regime via the frozen cumulative walk (throttled_lc at rho), plus
  a TRUE LIFT of L completed levels.
- Lift allocation schemes:
    * uniform  — each of the L levels lands on a game chosen uniformly at
                 random (with replacement), capped at the game's
                 number_of_levels; allocation redrawn per pseudo-seed.
    * concentrated — all L levels on a single game (uniform over games,
                 capped); the most detection-favorable allocation for the
                 per-game MAX statistic.
- Regime: expected world rho in {2.5, 3.0} (both anchors reported).
  ACTION-PARITY requires Sigma N72 >= 0.9 * Sigma N27, i.e. rho <= 1/0.9 =
  1.11 — structurally FALSE for all rho >= 2.4, so in the modeled regime
  GO reduces to CAPABILITY AND THROTTLED. (At parity-feasible rho the screen
  is a different, easier regime; out of scope of the anchors and noted in the
  filing.)
- k (72B seeds): 1 (the planned scored bench) and 2 (marginal rule expansion).
- Marginal-result rule (modeled, "procedure" variant): after the k=1 draw, if
  the capability statistic == 7 exactly, or either prong is within one level
  of its threshold, one extra pseudo-seed is drawn and the per-game MAX
  recomputed over both. Approximation: the extra seed is drawn on ALL 4 games
  (the sealed rule re-runs only the 2 decisive games); this can only raise
  P(GO), so the resulting false-GO is an UPPER bound and false-NO-GO a LOWER
  bound for the procedure — the conservative direction for the error the
  screen is accused of (false GO), noted in the filing.
- Errors:
    false NO-GO(L) = P( gate says NO-GO | true lift = +L ),  L in {1, 2, 3}
    false GO(L)    = P( gate says GO    | true lift = L <= 0 ), L in {0, -1}
- MC: 200_000 pseudo-screens per cell, fixed seed 20260723. Exact enumeration
  cross-check for k=1/uniform-allocation cells (4 rows x 4^L allocations).

Output: runs/a17_error_model/a17_error_model.json
"""
from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = json.load(open(ROOT / "runs/a17_repair/per_seed_table.json"))

GAMES = TABLE["games"]                      # ft09 sb26 lp85 vc33
RUNS = TABLE["runs"]                        # 3 certified war seeds + W0
CAP_BAR = TABLE["sigma_27B_max"] + 2        # = 8
MARGIN = 1
ANCHORS = ["2.5", "3.0"]
B = 200_000
SEED = 20260723

# number_of_levels caps, from the frozen actions_per_level array lengths
CAPS = {g: len(TABLE["actions_per_level"]["w0_eval_s1"][g]) for g in GAMES}


def rows_at(rho: str) -> list[dict]:
    return [TABLE["throttled_lc_by_rho"][rho][r] for r in RUNS]


def allocate(rng: random.Random, row: dict, L: int, scheme: str) -> dict:
    out = dict(row)
    if L > 0:
        if scheme == "concentrated":
            g = rng.choice(GAMES)
            out[g] = min(out[g] + L, CAPS[g])
        else:  # uniform
            for _ in range(L):
                g = rng.choice(GAMES)
                out[g] = min(out[g] + 1, CAPS[g])
    elif L < 0:
        for _ in range(-L):
            g = rng.choice(GAMES)
            out[g] = max(out[g] - 1, 0)
    return out


def sigma_max(rows: list[dict]) -> int:
    return sum(max(r[g] for r in rows) for g in GAMES)


def gate(rows: list[dict], bar_throttled: int) -> bool:
    s = sigma_max(rows)
    capability = s >= CAP_BAR
    throttled = s >= bar_throttled          # parity structurally FALSE here
    return capability and throttled


def marginal_triggers(rows: list[dict], bar_throttled: int) -> bool:
    s = sigma_max(rows)
    return (s == CAP_BAR - 1) or (abs(s - CAP_BAR) <= 1) or (abs(s - bar_throttled) <= 1)


def mc_cell(rho: str, L: int, scheme: str, k: int, procedure: bool) -> float:
    rng = random.Random((SEED, rho, L, scheme, k, procedure).__hash__())
    base_rows = rows_at(rho)
    bar = TABLE["sigma_null_adj"][rho] + MARGIN
    go = 0
    for _ in range(B):
        drawn = [allocate(rng, rng.choice(base_rows), L, scheme) for _ in range(k)]
        if procedure and k == 1 and marginal_triggers(drawn, bar):
            drawn.append(allocate(rng, rng.choice(base_rows), L, scheme))
        go += gate(drawn, bar)
    return go / B


def exact_k1_uniform(rho: str, L: int) -> float:
    """Exact P(GO) for k=1, uniform allocation, no marginal rule."""
    base_rows = rows_at(rho)
    bar = TABLE["sigma_null_adj"][rho] + MARGIN
    hits = 0.0
    total = 0
    allocs = list(itertools.product(GAMES, repeat=max(L, 0))) or [()]
    for row in base_rows:
        for alloc in allocs:
            out = dict(row)
            for g in alloc:
                out[g] = min(out[g] + 1, CAPS[g])
            hits += gate([out], bar)
            total += 1
    return hits / total


def detection_frontier(rho: str, scheme: str, k: int) -> dict:
    res = {}
    for L in range(1, 9):
        p = mc_cell(rho, L, scheme, k, procedure=False)
        res[str(L)] = round(1 - p, 5)
        if p >= 0.95:
            break
    return res


def main() -> None:
    out = {
        "prereg": "C3, prereg amendment 2026-07-23 (A17 symmetric error model)",
        "inputs": "runs/a17_repair/per_seed_table.json (verified rows)",
        "capability_bar": CAP_BAR,
        "margin": MARGIN,
        "caps_number_of_levels": CAPS,
        "B": B,
        "seed": SEED,
        "cells": {},
        "exact_check_k1_uniform": {},
        "false_nogo_frontier": {},
    }
    for rho in ANCHORS:
        bar = TABLE["sigma_null_adj"][rho] + MARGIN
        cell = {"sigma_null_adj": TABLE["sigma_null_adj"][rho], "bar_throttled": bar}
        for scheme in ("uniform", "concentrated"):
            for k in (1, 2):
                for L in (3, 2, 1, 0, -1):
                    p_go = mc_cell(rho, L, scheme, k, procedure=False)
                    key = f"scheme={scheme},k={k},L={L:+d}"
                    cell[key] = {
                        "P_GO": round(p_go, 5),
                        "error": round(1 - p_go, 5) if L > 0 else round(p_go, 5),
                        "error_kind": "false_NOGO" if L > 0 else "false_GO",
                    }
        # procedure variant (k=1 + marginal rule)
        for scheme in ("uniform", "concentrated"):
            for L in (3, 2, 1, 0, -1):
                p_go = mc_cell(rho, L, scheme, 1, procedure=True)
                key = f"procedure,scheme={scheme},L={L:+d}"
                cell[key] = {
                    "P_GO": round(p_go, 5),
                    "error": round(1 - p_go, 5) if L > 0 else round(p_go, 5),
                    "error_kind": "false_NOGO" if L > 0 else "false_GO",
                }
        out["cells"][f"rho={rho}"] = cell
        out["exact_check_k1_uniform"][f"rho={rho}"] = {
            f"L={L:+d}": round(exact_k1_uniform(rho, L), 5) for L in (0, 1, 2, 3)
        }
        out["false_nogo_frontier"][f"rho={rho}"] = {
            f"scheme={s},k={k}": detection_frontier(rho, s, k)
            for s in ("uniform", "concentrated") for k in (1, 2)
        }

    dst = ROOT / "runs/a17_error_model/a17_error_model.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
