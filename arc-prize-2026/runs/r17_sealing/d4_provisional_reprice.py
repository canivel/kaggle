"""D4 — provisional re-pricing of efficiency-denominated channels through the
VERIFIED level-number-weighted scoring oracle (OBJ-B / R-DEPTH-REPOINT).

Panel ruling R-DEPTH-REPOINT: adopt duck_eval/scoring_oracle.py (validated to
0.00e+00 vs harness, runs/atlas_oracle/validation.md) as the sealed deterministic
local scoring authority, and republish every efficiency-denominated price
(§7 EWM tu93/ls20/ft09 channels, §2 (a)/(b)/EWM-in pair rows) through the true
aggregate  game_score = Σ score_i·i / Σ i  (level-number weighted).

NO SEAL MOVES: methodology confirms the binding sign test is on Δlc (depth
events) and is unaffected. This is a $0 recompute + PRE-SCORECARD/PROVISIONAL
labeling pass only.

Baselines: methodology N1 requires the LEGAL control w0_s1; we price against
runs/kernel_pulls/w0_eval_s1/benchmark.json per-run baselines
(load_baselines_from_benchmark), NOT the atlas (which drifts 20/25).

Output: d4_provisional_reprice.json (+ the .md is written by hand from these).
Runtime-tested; stdlib + duck_eval only; $0.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from duck_eval.scoring_oracle import (  # noqa: E402
    score_game, load_baselines_from_benchmark, using_real_scorer,
)

CTRL = ROOT / "runs" / "kernel_pulls" / "w0_eval_s1" / "benchmark.json"
BL = load_baselines_from_benchmark(str(CTRL))
BENCH = json.loads(CTRL.read_text(encoding="utf-8"))
RUNS = {g["game_id"]: g for g in BENCH["game_runs"]}
LB_DIV = 25.0  # LB = mean game score over 25 games; 1 game-pt = 1/25 rail


def realized(gid):
    """Full-length (actions, completed) as the legal control actually played."""
    g = RUNS[gid]
    nl, lc, apl = g["number_of_levels"], g["levels_completed"], g["actions_per_level"]
    actions = [apl[i] if i < len(apl) else 0 for i in range(nl)]
    completed = [i < lc for i in range(nl)]
    return actions, completed, BL[gid], nl, lc


def sg(gid, actions, completed, base):
    return score_game(gid, actions, completed, baselines=base)


def weight_frac(level_idx0, nl):
    """Fraction of a game's weight carried by level (1-based = idx0+1)."""
    return (level_idx0 + 1) / sum(range(1, nl + 1))


def price_l1_speed(gid):
    """Efficiency channel: shave L1 actions on an ALREADY-COMPLETED L1.
    Best case = shave to human parity; realistic = shave a modest slice.
    If L1 not completed, the channel cannot pay as efficiency (see l1_complete)."""
    actions, completed, base, nl, lc = realized(gid)
    if lc < 1:
        return {"applicable": False,
                "reason": "L1 not completed in the control run -> no efficiency "
                          "headroom; only a completion channel can pay (see below)"}
    s0 = sg(gid, actions, completed, base)
    a_par = list(actions); a_par[0] = base[0]
    s_par = sg(gid, a_par, completed, base)
    # L1 already at/below human baseline => already at the 100 cap, zero headroom
    at_cap = actions[0] <= base[0]
    # realistic partial shave: halve the gap to parity
    a_half = list(actions)
    a_half[0] = base[0] + (actions[0] - base[0]) // 2 if actions[0] > base[0] else base[0]
    s_half = sg(gid, a_half, completed, base)
    return {
        "applicable": True,
        "l1_actions_control": actions[0], "l1_baseline": base[0],
        "l1_at_cap_no_headroom": at_cap,
        "game_score_control": round(s0, 4),
        "best_case_shave_to_parity": {"game_score": round(s_par, 4),
                                      "delta_pts": round(s_par - s0, 4),
                                      "delta_lb_rail": round((s_par - s0) / LB_DIV, 5)},
        "realistic_half_gap_shave": {"game_score": round(s_half, 4),
                                     "delta_pts": round(s_half - s0, 4),
                                     "delta_lb_rail": round((s_half - s0) / LB_DIV, 5)},
        "l1_weight_frac": round(weight_frac(0, nl), 4),
    }


def price_l1_completion(gid):
    """Completion channel: if L1 is UNcompleted, value of newly completing it."""
    actions, completed, base, nl, lc = realized(gid)
    if lc >= 1:
        return {"applicable": False, "reason": "L1 already completed"}
    s0 = sg(gid, actions, completed, base)
    a2 = list(actions); c2 = list(completed); c2[0] = True; a2[0] = base[0]
    s1 = sg(gid, a2, c2, base)
    return {"applicable": True, "game_score_control": round(s0, 4),
            "complete_l1_at_parity": {"game_score": round(s1, 4),
                                      "delta_pts": round(s1 - s0, 4),
                                      "delta_lb_rail": round((s1 - s0) / LB_DIV, 5)},
            "note": "this is a DEPTH (new-clear) event at L1, not an efficiency "
                    "channel; L1 depth weight is the smallest of the game"}


def price_frontier_depth(gid):
    """Reference DEPTH event: +1 level at the current frontier at human parity.
    This is what the binding Δlc sign test actually rewards — priced for contrast."""
    actions, completed, base, nl, lc = realized(gid)
    if lc >= nl:
        return {"applicable": False, "reason": "game fully completed"}
    s0 = sg(gid, actions, completed, base)
    a2 = list(actions); c2 = list(completed); c2[lc] = True; a2[lc] = base[lc]
    s1 = sg(gid, a2, c2, base)
    return {"applicable": True, "frontier_level_1based": lc + 1,
            "level_weight_frac": round(weight_frac(lc, nl), 4),
            "game_score_control": round(s0, 4),
            "complete_frontier_at_parity": {"game_score": round(s1, 4),
                                            "delta_pts": round(s1 - s0, 4),
                                            "delta_lb_rail": round((s1 - s0) / LB_DIV, 5)}}


def main():
    assert using_real_scorer(), "expected the shipped arc_agi.scorecard scorer"
    out = {
        "ruling": "R-DEPTH-REPOINT (OBJ-B); PRE-SCORECARD/PROVISIONAL; NO SEAL MOVES",
        "scorer": "duck_eval/scoring_oracle.py (level-number-weighted "
                  "Sigma score_i*i / Sigma i; validated 0.00e+00 vs harness)",
        "baselines": "runs/kernel_pulls/w0_eval_s1/benchmark.json (legal control "
                     "w0_s1 per methodology N1; per-run base_actions_per_level)",
        "lb_divisor": LB_DIV,
        "channels": {},
    }
    # §7 EWM efficiency channels
    out["channels"]["tu93_L1_speed"] = {
        "old_price_pts": "+0.1-0.9", "type": "efficiency (L1 speed)",
        "reprice": price_l1_speed("tu93-0768757b")}
    out["channels"]["ft09_L1_reliability"] = {
        "old_price_pts": "+0.0-1.0", "type": "efficiency/reliability (L1)",
        "reprice": price_l1_speed("ft09-0d8bbf25")}
    out["channels"]["ls20_L1_speed"] = {
        "old_price_pts": "+0.0-0.6", "type": "efficiency (L1 speed) [MISNOMER: lc=0]",
        "reprice_as_efficiency": price_l1_speed("ls20-9607627b"),
        "reprice_as_completion": price_l1_completion("ls20-9607627b")}
    # DEPTH contrast (the channel the Δlc sign test actually prices)
    out["depth_contrast_frontier"] = {
        gid.split("-")[0]: price_frontier_depth(gid)
        for gid in ["tu93-0768757b", "ka59-38d34dbb", "re86-8af5384d",
                    "sc25-635fd71a"]}
    OUT = ROOT / "runs" / "r17_sealing" / "d4_provisional_reprice.json"
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
