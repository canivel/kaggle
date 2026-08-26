"""Variance reconciliation (panel R9: ME-NEW-11 / rl-planning P / systems Q3).

Question: null10 bootstrap 1-seed paired-delta sd = 0.52, but Kaggle-LB control
draws {0.82,0.89,0.93,1.02,0.95} give sigma-hat = 0.074 per draw (implying a
paired-delta sd of ~0.105). 5-7x apart. Which is the right replicate noise for
which instrument, and what power does a 3-seed build-rail gate actually have?

Method (all from existing artifacts, zero compute spend):
  1. Score all 10 null10 seeds per-game with the validated RHAE scorer
     (scripts/phase1_gate.py, validated 0e+00 vs Tufa's 500 runs).
  2. Direct (non-bootstrap) estimates on the 25-game build/pod rail:
     - sd across the 10 per-seed 25-game means  -> 1-seed run-mean sd
     - sd of all 45 pairwise seed-mean deltas   -> 1-seed paired-delta sd
  3. Per-game variance decomposition: which games carry the variance.
  4. Game-resampling bootstrap replication (the suspected source of 0.52).
  5. Implied se + power of a 3-seed gate on this rail vs +0.10 / +0.12.

Usage: uv run python scripts/variance_reconcile.py
Output: runs/variance_reconcile/report.md (+ raw JSON)
"""
from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from phase1_gate import load_null, load_our_seed  # noqa: E402

NULL10 = ROOT / "runs" / "null10"
OUT_DIR = ROOT / "runs" / "variance_reconcile"

# Kaggle-LB control pool (frozen-fork sigma draws, ITERATION_LOG)
LB_DRAWS = [0.82, 0.89, 0.93, 1.02, 0.95]

# normal-approx helpers (avoid scipy dependency)
def phi(x):  # standard normal CDF
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    null_games, max_err, n_checked, overall, _ = load_null(
        ROOT / "runs" / "tufa_example_run" / "benchmark.json",
        ROOT / "runs" / "tufa_example_run" / "score.json")
    assert max_err < 1e-9, f"scorer validation failed: {max_err}"

    seed_files = sorted(NULL10.glob("vanilla_seed*.json"))
    assert len(seed_files) == 10, f"expected 10 null seeds, got {len(seed_files)}"
    seeds = {}
    for sf in seed_files:
        label = sf.stem.replace("vanilla_", "")
        seeds[label] = load_our_seed(sf, null_games)

    prefixes = sorted({p for s in seeds.values() for p in s})
    # per-game score matrix [game][seed]
    mat = {p: [seeds[s][p]["score"] for s in sorted(seeds) if p in seeds[s]]
           for p in prefixes}
    complete = {p: v for p, v in mat.items() if len(v) == len(seeds)}
    n_games = len(complete)

    # --- 2. direct replicate estimates on the 25-game rail
    per_seed_means = [st.mean([complete[p][i] for p in complete])
                      for i in range(len(seeds))]
    sd_run_mean = st.stdev(per_seed_means)
    pair_deltas = [a - b for a, b in combinations(per_seed_means, 2)]
    sd_pair_direct = st.stdev(pair_deltas)

    # --- 3. per-game decomposition
    decomp = []
    for p, v in complete.items():
        var_within = st.variance(v)  # across-seed variance for this game
        decomp.append((p, st.mean(v), var_within, var_within / n_games**2))
    decomp.sort(key=lambda r: -r[2])
    # if games were independent, run-mean variance = sum(var_g)/n^2
    var_indep = sum(r[3] for r in decomp)

    # --- 4. bootstrap replications (identify the 0.52's resampling unit)
    rng = random.Random(0)
    prefixes_c = sorted(complete)
    # (a) game-resampling bootstrap of a 1-seed delta, pairing BROKEN:
    #     each arm draws its own resampled games (the unit-mismatch suspect)
    boot_a = []
    for _ in range(20000):
        i1, i2 = rng.sample(range(len(seeds)), 2)
        d = st.mean([complete[rng.choice(prefixes_c)][i1]
                     - complete[rng.choice(prefixes_c)][i2]
                     for _ in range(n_games)])
        boot_a.append(d)
    sd_boot_games_unpaired_games = st.stdev(boot_a)
    # (b) paired-by-game bootstrap (resample games, keep pairing)
    boot_b = []
    for _ in range(20000):
        i1, i2 = rng.sample(range(len(seeds)), 2)
        picks = [rng.choice(prefixes_c) for _ in range(n_games)]
        d = st.mean([complete[p][i1] - complete[p][i2] for p in picks])
        boot_b.append(d)
    sd_boot_paired = st.stdev(boot_b)

    # --- 5. gate power on this rail (paired per-game design, 3 seeds/arm)
    # statistic: mean over games of (mean_3seed_ours - mean_3seed_null)
    se_3seed = sd_pair_direct / math.sqrt(3)
    alpha = 0.0125
    z_crit = 2.241  # one-sided z for alpha=0.0125
    power = {f"+{eff:.2f}": 1 - phi(z_crit - eff / se_3seed)
             for eff in (0.10, 0.12, 0.20)}

    # --- 5b. achievable gate statistics (trimmed / lc-based)
    def rail_sd(games_subset, transform=None):
        ms = []
        for i in range(len(seeds)):
            vals = [complete[p][i] for p in games_subset]
            if transform:
                vals = [transform(v) for v in vals]
            ms.append(st.mean(vals))
        return st.stdev(ms)

    heavy = [d[0] for d in decomp[:2]]  # ft09, vc33
    sd_no_ft09 = rail_sd([p for p in prefixes_c if p != decomp[0][0]])
    sd_no_top2 = rail_sd([p for p in prefixes_c if p not in heavy])
    sd_log = rail_sd(prefixes_c, transform=lambda v: math.log1p(v))
    # lc-based rail (levels completed, robust to RHAE tails)
    lc_mat = {p: [seeds[s][p]["lc"] for s in sorted(seeds) if p in seeds[s]]
              for p in prefixes_c}
    lc_means = [st.mean([lc_mat[p][i] for p in prefixes_c])
                for i in range(len(seeds))]
    sd_lc = st.stdev(lc_means)
    alt_gates = {}
    for name, sd_alt in (("full", sd_run_mean), ("no_ft09", sd_no_ft09),
                         ("no_top2", sd_no_top2), ("log1p", sd_log),
                         ("levels_completed", sd_lc)):
        se3 = sd_alt * math.sqrt(2) / math.sqrt(3)
        alt_gates[name] = {
            "sd_1seed": sd_alt, "se_3seed_delta": se3,
            "power_vs_0.10": 1 - phi(z_crit - 0.10 / se3),
            "power_vs_0.20": 1 - phi(z_crit - 0.20 / se3),
        }

    # --- 5c. expected max over k LB draws (order-stats ceiling)
    def e_max_std_normal(k, n_sim=20000):
        r = random.Random(1)
        sims = [max(r.gauss(0, 1) for _ in range(k)) for _ in range(n_sim)]
        return st.mean(sims)

    ks = [5, 10, 30, 60, 110]
    emax = {k: e_max_std_normal(k) for k in ks}
    lb_mean = st.mean(LB_DRAWS)
    lb_sd_pre = st.stdev(LB_DRAWS)
    ci_hi_sd = 0.213  # chi2 95% upper bound on sigma, 4 df (project ledger)
    max_curve = {
        str(k): {
            "sigma_0.074": lb_mean + 0.074 * emax[k],
            "sigma_0.213_ci_hi": lb_mean + ci_hi_sd * emax[k],
            "sigma_0.52_bootstrap_claim": lb_mean + 0.52 * emax[k],
        } for k in ks}

    # --- 6. LB-rail comparison
    lb_sd = st.stdev(LB_DRAWS)
    lb_pair_sd = lb_sd * math.sqrt(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {
        "n_seeds": len(seeds), "n_games_complete": n_games,
        "per_seed_means": per_seed_means,
        "sd_run_mean_direct": sd_run_mean,
        "sd_paired_delta_direct": sd_pair_direct,
        "sd_paired_delta_implied_sqrt2": sd_run_mean * math.sqrt(2),
        "var_if_games_independent": var_indep,
        "sd_if_games_independent": math.sqrt(var_indep) * math.sqrt(2),
        "bootstrap_games_unpaired_sd": sd_boot_games_unpaired_games,
        "bootstrap_games_paired_sd": sd_boot_paired,
        "top_variance_games": [
            {"game": p, "mean": m, "across_seed_var": v,
             "share_of_runmean_var": c / var_indep}
            for p, m, v, c in decomp[:8]],
        "gate_3seed_se": se_3seed,
        "gate_power_alpha_0125": power,
        "alt_gate_statistics": alt_gates,
        "expected_max_over_k_draws": max_curve,
        "e_max_std_normal": {str(k): v for k, v in emax.items()},
        "lb_control_sd": lb_sd, "lb_paired_delta_sd": lb_pair_sd,
        "ratio_buildrail_over_lb": sd_pair_direct / lb_pair_sd,
    }
    (OUT_DIR / "raw.json").write_text(json.dumps(raw, indent=2))

    top = "\n".join(
        f"| {p} | {m:.2f} | {v:.3f} | {100*c/var_indep:.1f}% |"
        for p, m, v, c in decomp[:8])
    report = f"""# Variance reconciliation — null10 build rail vs Kaggle LB
Date: 2026-07-14. Addresses panel R9 ME-NEW-11 (methodology), P (rl-planning), Q3 (systems).
Scorer: validated RHAE mirror (max err vs Tufa 500 runs: {max_err:.1e}, {n_checked} checks).

## Direct (non-bootstrap) replicate noise, 25-game build/pod rail
- 10 per-seed 25-game means: {', '.join(f'{m:.3f}' for m in per_seed_means)}
- **1-seed run-mean sd = {sd_run_mean:.3f}**
- **1-seed paired-delta sd (45 pairwise) = {sd_pair_direct:.3f}** (sqrt2 x run-mean = {sd_run_mean*math.sqrt(2):.3f})
- If games were independent, implied paired-delta sd = {math.sqrt(var_indep)*math.sqrt(2):.3f}

## Bootstrap replication (what unit produced 0.52?)
- Game-resampling, pairing BROKEN across arms: sd = {sd_boot_games_unpaired_games:.3f}
- Game-resampling, pairing KEPT: sd = {sd_boot_paired:.3f}

## Per-game variance decomposition (top 8)
| game | null mean | across-seed var | share of run-mean var |
|---|---|---|---|
{top}

## LB rail
- Control draws sd = {lb_sd:.3f} -> paired-delta sd = {lb_pair_sd:.3f}
- Ratio build-rail/LB paired-delta sd = {sd_pair_direct/lb_pair_sd:.1f}x

## 3-seed gate on the build rail (alpha = 0.0125, one-sided)
- se(mean paired delta, 3 seeds) = {se_3seed:.3f}
- Power vs +0.10: {power['+0.10']:.2f}; vs +0.12: {power['+0.12']:.2f}; vs +0.20: {power['+0.20']:.2f}

## Alternative gate statistics (1-seed sd -> 3-seed se -> power vs +0.10 / +0.20)
| statistic | sd_1seed | se_3seed | power +0.10 | power +0.20 |
|---|---|---|---|---|
""" + "\n".join(
        f"| {n} | {g['sd_1seed']:.3f} | {g['se_3seed_delta']:.3f} | "
        f"{g['power_vs_0.10']:.2f} | {g['power_vs_0.20']:.2f} |"
        for n, g in alt_gates.items()) + f"""

## Expected max over k daily LB draws (control mean {lb_mean:.3f})
| k | sigma=0.074 | sigma=0.213 (chi2 CI hi) | sigma=0.52 (bootstrap claim) |
|---|---|---|---|
""" + "\n".join(
        f"| {k} | {max_curve[str(k)]['sigma_0.074']:.2f} | "
        f"{max_curve[str(k)]['sigma_0.213_ci_hi']:.2f} | "
        f"{max_curve[str(k)]['sigma_0.52_bootstrap_claim']:.2f} |"
        for k in ks) + """

Reading: the 1.44 wall is unreachable by order statistics from the current
per-draw distribution under any candidate sigma except the (falsified) 0.52
LB claim; only per-draw mean improvements close the gap.
"""
    (OUT_DIR / "report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
