"""LB draw-process generative model + stress test of the 07-18 'variance bombshell'.

Panel R14 support artifact. DESCRIPTIVE / SIMULATION ONLY:
- consumes NO gate look (the sealed A5/A8 n=5 war look is untouched);
- uses only already-pulled local artifacts (13 instrumented seeds) and the
  public LB ledger numbers already on the record in ITERATION_LOG.md.

Model: the nightly LB score is the RHAE mean over the FIXED official 110-game
set, one pass per game.  We proxy the unknown official games by cloning the 25
public games to 110 slots (multiplicity 4-5, randomized per 'world'), with two
difficulty calibrations:
  SCALE: multiply every game score by c so E[night] hits the target mean
         (the 0.55 ratio of position_analysis / path_forward);
  TRIM:  drop the top-mean public games (ft09, vc33, r11l, ar25) until the
         clone-set mean is near target, then small c adjust (gap_forensics C:
         'ft09-like outliers rarer in the official mix').
Per-slot outcomes are drawn from per-game EMPIRICAL distributions fitted from
the 13 fully-instrumented local seeds (runs/null10 x10 + war_eval_v1-3),
scored with the validated RHAE mirror (scripts/phase1_gate.py) and the Tufa
null base_actions_per_level.  Cross-game correlation variants:
  IND   all 110 slots independent;
  COR   IND + additive common night effect (ANOVA estimate from the 13 seeds);
  BLOCK one seed per night drives all slots (upper bound, degenerate 13-point).

Usage: uv run python runs/lb_process_model/lb_process_model.py
Writes: runs/lb_process_model/fits_and_sims.json (+ prints report tables)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase1_gate import load_null, load_our_seed  # noqa: E402

OUT = ROOT / "runs" / "lb_process_model"

# ------------------------------------------------------------------ ledgers
FROZEN5 = [0.82, 0.89, 0.93, 1.02, 0.95]
FROZEN6 = FROZEN5 + [1.33]
WAR4 = [0.91, 1.08, 0.88, 1.05]
POOLED10 = FROZEN6 + WAR4

N_OFFICIAL = 110
WALLS = [1.33, 1.44, 1.56, 1.61, 1.86]
KS_FUTURE = [30, 60, 107]
KS_OBS = [6, 10, 13]

rng = np.random.default_rng(20260718)


def sd(xs):
    return float(np.std(xs, ddof=1))


# ------------------------------------------------------------------ 1. fits
def fit_games():
    null_games, max_err, n_checked, overall, _ = load_null(
        ROOT / "runs" / "tufa_example_run" / "benchmark.json",
        ROOT / "runs" / "tufa_example_run" / "score.json")
    assert max_err < 1e-9, f"scorer validation failed: {max_err}"
    seed_paths = sorted((ROOT / "runs" / "null10").glob("vanilla_seed*.json"))
    assert len(seed_paths) == 10
    seed_paths += [ROOT / "runs" / "kernel_pulls" / v / "benchmark.json"
                   for v in ("war_eval_v1", "war_eval_v2", "war_eval_v3")]
    seeds = [load_our_seed(p, null_games) for p in seed_paths]
    prefixes = sorted(null_games)
    score = {},
    fits = {}
    for p in prefixes:
        obs = [(s[p]["score"], s[p]["lc"]) for s in seeds if p in s]
        tufa = null_games[p]["scores"]  # 20 vanilla passes, same scorer
        fits[p] = {
            "scores13": [o[0] for o in obs],
            "lc13": [o[1] for o in obs],
            "tufa20": tufa,
            "n13": len(obs),
        }
    return fits, max_err, n_checked, [str(p) for p in seed_paths]


# ------------------------------------------------------- 2. night simulator
def simulate(fits, games, use_tufa, target_mean, corr, n_worlds=50,
             nights_per_world=400, keep_draws_game=None):
    """Return dict of simulated night stats (+ optional attribution arrays)."""
    vals = {}
    lcs = {}
    for p in games:
        v = list(fits[p]["scores13"]) + (list(fits[p]["tufa20"]) if use_tufa else [])
        l = list(fits[p]["lc13"]) + ([np.nan] * len(fits[p]["tufa20"]) if use_tufa else [])
        vals[p] = np.asarray(v, float)
        lcs[p] = np.asarray(l, float)
    n_games = len(games)
    mu_g = {p: vals[p].mean() for p in games}

    # common night effect (ANOVA from the 13-seed local rail, ALL 25 games)
    seed_mat = np.array([fits[p]["scores13"] for p in sorted(fits)])  # 25 x 13
    seed_means = seed_mat.mean(axis=0)
    s_run = np.std(seed_means, ddof=1)
    s_ind = math.sqrt(float(np.var(seed_mat, axis=1, ddof=1).sum()) / seed_mat.shape[0] ** 2)
    sigma_u_local = math.sqrt(max(0.0, s_run ** 2 - s_ind ** 2))

    all_nights = []
    contrib = {p: [] for p in (keep_draws_game or [])}
    lc_draws = {p: [] for p in (keep_draws_game or [])}
    base, extra = divmod(N_OFFICIAL, n_games)
    for _ in range(n_worlds):
        mult = np.full(n_games, base)
        mult[rng.choice(n_games, size=extra, replace=False)] += 1
        world_mean = sum(m * mu_g[p] for m, p in zip(mult, games)) / N_OFFICIAL
        c = target_mean / world_mean
        night_tot = np.zeros(nights_per_world)
        if corr == "BLOCK":
            si = rng.integers(0, len(vals[games[0]]), size=nights_per_world)
            for m, p in zip(mult, games):
                night_tot += m * vals[p][si]
        else:
            for m, p in zip(mult, games):
                idx = rng.integers(0, len(vals[p]), size=(nights_per_world, m))
                gsum = vals[p][idx].sum(axis=1)
                night_tot += gsum
                if p in contrib:
                    contrib[p].append(c * gsum / N_OFFICIAL)
                    lc_draws[p].append(np.nanmean(lcs[p][idx], axis=1))
        nights = c * night_tot / N_OFFICIAL
        if corr == "COR":
            nights = nights + c * rng.normal(0.0, sigma_u_local, size=nights_per_world)
        all_nights.append(nights)
    nights = np.concatenate(all_nights)
    out = {"nights": nights, "sigma_u_local": sigma_u_local,
           "s_run_local": float(s_run), "s_ind_local": float(s_ind)}
    if keep_draws_game:
        out["contrib"] = {p: np.concatenate(a) for p, a in contrib.items()}
        out["lc_draws"] = {p: np.concatenate(a) for p, a in lc_draws.items()}
    return out


def night_stats(nights, label):
    n = np.sort(nights)
    N = len(n)
    def emax(k):
        i = np.arange(1, N + 1, dtype=float)
        w = (i / N) ** k - ((i - 1) / N) ** k
        return float((n * w).sum())
    def pmax(x, k):
        F = np.searchsorted(n, x, side="left") / N
        return float(1 - F ** k)
    # sample-sd sampling distributions (windows of 5 and 6 nights)
    w5 = nights[: N // 5 * 5].reshape(-1, 5)
    w6 = nights[: N // 6 * 6].reshape(-1, 6)
    sd5 = np.std(w5, axis=1, ddof=1)
    sd6 = np.std(w6, axis=1, ddof=1)
    return {
        "label": label,
        "mean": float(n.mean()), "sd": float(np.std(n, ddof=1)),
        "q": {q: float(np.quantile(n, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
        "min": float(n[0]), "max": float(n[-1]),
        "P_night_ge": {str(x): float((nights >= x).mean()) for x in WALLS},
        "P_max_ge_1.33_k": {str(k): pmax(1.33, k) for k in KS_OBS},
        "P_max_ge_walls_k107": {str(x): pmax(x, 107) for x in WALLS},
        "E_max_k": {str(k): emax(k) for k in KS_OBS + KS_FUTURE},
        "sdhat5": {"median": float(np.median(sd5)),
                   "P_le_0.074": float((sd5 <= 0.074).mean()),
                   "P_ge_0.179": float((sd5 >= 0.179).mean())},
        "sdhat6": {"median": float(np.median(sd6)),
                   "P_le_0.074": float((sd6 <= 0.074).mean()),
                   "P_ge_0.179": float((sd6 >= 0.179).mean())},
    }


# ------------------------------------------------- 3. normal-theory module
CHI2 = {  # (df) -> (q025, q975)
    4: (0.484419, 11.1433), 5: (0.831211, 12.8325), 9: (2.70039, 19.0228),
}


def emax_std_normal(k, nsim=200_000):
    return float(rng.standard_normal((nsim, 1)).max()) if k == 1 else \
        float(np.max(rng.standard_normal((nsim, k)), axis=1).mean())


def normal_module(ledger, name, ks=KS_FUTURE):
    x = np.asarray(ledger, float)
    n, m, s = len(x), float(x.mean()), sd(x)
    df = n - 1
    lo, hi = CHI2[df]
    ci = (s * math.sqrt(df / hi), s * math.sqrt(df / lo))
    ek = {k: emax_std_normal(k, 60_000) for k in ks}
    def pmax(x0, mu, sig, k):
        z = (x0 - mu) / sig
        return float(1 - (0.5 * (1 + math.erf(z / math.sqrt(2)))) ** k)
    fixed = {}
    for tag, sig in (("point", s), ("ci_lo", ci[0]), ("ci_hi", ci[1])):
        fixed[tag] = {
            "sigma": sig,
            "E_max_k": {str(k): m + sig * ek[k] for k in ks},
            "P_max_ge": {str(k): {str(w): pmax(w, m, sig, k) for w in WALLS}
                         for k in ks},
            "P_single_ge_1.33": pmax(1.33, m, sig, 1),
        }
    # posterior predictive (noninformative prior): sigma^2 ~ df s^2 / chi2_df,
    # mu | sigma ~ N(m, sigma^2/n), future draws iid N(mu, sigma)
    nsim = 20_000
    sig_post = np.sqrt(df * s * s / rng.chisquare(df, nsim))
    mu_post = rng.normal(m, sig_post / math.sqrt(n))
    post = {}
    for k in ks:
        z = rng.standard_normal((nsim, k))
        mx = mu_post + sig_post * z.max(axis=1)
        post[str(k)] = {
            "E_max": float(mx.mean()),
            "P_max_ge": {str(w): float((mx >= w).mean()) for w in WALLS},
        }
    z1 = rng.standard_normal(nsim)
    d1 = mu_post + sig_post * z1
    post["single_draw"] = {"P_ge_1.33": float((d1 >= 1.33).mean()),
                           "P_ge_1.44": float((d1 >= 1.44).mean())}
    return {"name": name, "n": n, "mean": m, "sd": s, "df": df,
            "sigma_ci95": list(ci), "e_max_std_normal": {str(k): ek[k] for k in ks},
            "fixed_sigma": fixed, "posterior_predictive": post}


# ------------------------------------------------------------------- main
def main():
    fits, max_err, n_checked, seed_files = fit_games()
    games_all = sorted(fits)
    # TRIM set: drop top-mean games until clone-set mean ~ LB target band
    means13 = {p: float(np.mean(fits[p]["scores13"])) for p in games_all}
    trim_drop = ["ft09", "vc33", "r11l", "ar25"]
    games_trim = [p for p in games_all if p not in trim_drop]

    grand13 = float(np.mean([means13[p] for p in games_all]))
    per_game = {p: {"mean13": means13[p],
                    "sd13": sd(fits[p]["scores13"]),
                    "max13": float(np.max(fits[p]["scores13"])),
                    "lc13": fits[p]["lc13"],
                    "scores13": [round(v, 3) for v in fits[p]["scores13"]],
                    "tufa_mean": float(np.mean(fits[p]["tufa20"]))}
                for p in games_all}

    combos = [
        ("A_scale0.922_IND", games_all, False, 0.922, "IND"),
        ("B_scale0.922_COR", games_all, False, 0.922, "COR"),
        ("C_scale0.922_BLOCK", games_all, False, 0.922, "BLOCK"),
        ("D_scale0.986_IND", games_all, False, 0.986, "IND"),
        ("E_scale0.986_COR", games_all, False, 0.986, "COR"),
        ("F_trim0.922_IND", games_trim, False, 0.922, "IND"),
        ("G_trim0.922_COR", games_trim, False, 0.922, "COR"),
        ("H_scale0.922_IND_tufa33", games_all, True, 0.922, "IND"),
    ]
    watch = ["ft09", "vc33", "tn36", "r11l", "ar25", "cn04", "re86", "tu93"]
    sims = {}
    attribution = None
    for name, games, use_tufa, target, corr in combos:
        keep = watch if name == "B_scale0.922_COR" else None
        r = simulate(fits, games, use_tufa, target, corr, keep_draws_game=keep)
        sims[name] = night_stats(r["nights"], name)
        sims[name]["sigma_u_local"] = r["sigma_u_local"]
        if keep:
            nights = r["nights"]
            hot = nights >= 1.30
            att = {}
            for p in watch:
                cg, lg = r["contrib"][p], r["lc_draws"][p]
                att[p] = {
                    "mean_contrib": float(cg.mean()),
                    "mean_contrib_hot": float(cg[hot].mean()),
                    "excess": float(cg[hot].mean() - cg.mean()),
                    "mean_lc": float(np.nanmean(lg)),
                    "mean_lc_hot": float(np.nanmean(lg[hot])),
                }
            tot_excess = 1.30 - nights.mean() if False else None
            attribution = {"threshold": 1.30, "n_hot": int(hot.sum()),
                           "n_nights": len(nights), "games": att,
                           "hot_mean": float(nights[hot].mean())}

    normals = {
        "frozen5": normal_module(FROZEN5, "frozen5 (pre-1.33)"),
        "frozen6": normal_module(FROZEN6, "frozen6 (pooled with 1.33, brief 1a)"),
        "pooled10": normal_module(POOLED10, "pooled10 (frozen6 + war4 quasi-control)"),
    }

    # was sigma-hat5=0.074 just lucky? under each sim combo it's in sdhat5.
    # ledger summary stats
    ledgers = {
        "frozen5": {"draws": FROZEN5, "mean": float(np.mean(FROZEN5)), "sd": sd(FROZEN5)},
        "frozen6": {"draws": FROZEN6, "mean": float(np.mean(FROZEN6)), "sd": sd(FROZEN6)},
        "war4": {"draws": WAR4, "mean": float(np.mean(WAR4)), "sd": sd(WAR4)},
        "pooled10": {"draws": POOLED10, "mean": float(np.mean(POOLED10)), "sd": sd(POOLED10)},
    }

    # marginal-window economics: P(single filler draw >= 1.33) under
    # posterior predictive of each ledger + primary sim combo; break-even
    # mean-shift delta for an experimental draw with infra-death prob q.
    econ = {}
    prim = sims["B_scale0.922_COR"]
    p_f_sim = prim["P_night_ge"]["1.33"]
    for lname, nm in normals.items():
        p_f = nm["posterior_predictive"]["single_draw"]["P_ge_1.33"]
        mu, s = nm["mean"], nm["sd"]
        be = {}
        for q in (0.0, 0.15, 0.30):
            # solve (1-q) * P(N(mu+d, s) >= 1.33) = p_f  for d (point sigma)
            target_p = min(0.999, p_f / (1 - q)) if q < 1 else 1
            # invert normal tail
            from math import sqrt, erf
            def tail(d):
                z = (1.33 - (mu + d)) / s
                return 1 - 0.5 * (1 + erf(z / sqrt(2)))
            lo_d, hi_d = -0.5, 1.0
            for _ in range(80):
                mid = (lo_d + hi_d) / 2
                if tail(mid) < target_p:
                    lo_d = mid
                else:
                    hi_d = mid
            be[str(q)] = round((lo_d + hi_d) / 2, 4)
        econ[lname] = {"P_filler_ge_1.33_postpred": p_f,
                       "breakeven_delta_vs_infradeath_q": be}
    econ["sim_primary_P_night_ge_1.33"] = p_f_sim

    raw = {
        "meta": {
            "date": "2026-07-18",
            "discipline": "descriptive/simulation only; NO sealed A5/A8 look consumed",
            "scorer_validation": {"max_err": max_err, "n_checked": n_checked},
            "seed_files": seed_files,
            "grand_mean_13seed_25game": grand13,
            "n_official_games": N_OFFICIAL,
            "trim_dropped": trim_drop,
        },
        "ledgers": ledgers,
        "per_game_fit": per_game,
        "sim_combos": {k: {kk: vv for kk, vv in v.items() if kk != "nights"}
                       for k, v in sims.items()},
        "attribution_B_scale0.922_COR_ge_1.30": attribution,
        "normal_theory": normals,
        "window_economics": econ,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fits_and_sims.json").write_text(json.dumps(raw, indent=2))
    print(json.dumps({k: raw[k] for k in ("ledgers",)}, indent=2))
    print(f"\ngrand13={grand13:.4f}  scorer max_err={max_err:.2e} ({n_checked} checks)")
    print(f"\nsigma_u_local={sims['B_scale0.922_COR']['sigma_u_local']:.4f}")
    print("\n=== per-game 13-seed fit (mean/sd/max, lc set) ===")
    for p in sorted(per_game, key=lambda q: -per_game[q]["mean13"]):
        g = per_game[p]
        print(f"{p}: mean {g['mean13']:6.2f} sd {g['sd13']:6.2f} max {g['max13']:7.2f} "
              f"lc {sorted(set(g['lc13']))} tufa_mean {g['tufa_mean']:6.2f}")
    print("\n=== sim combos ===")
    for k, v in sims.items():
        print(f"\n-- {k}: mean {v['mean']:.3f} sd {v['sd']:.3f} "
              f"q95 {v['q'][0.95]:.3f} q99 {v['q'][0.99]:.3f} max {v['max']:.3f}")
        print(f"   P(night>=1.33) {v['P_night_ge']['1.33']:.4f}  "
              f"P(max6>=1.33) {v['P_max_ge_1.33_k']['6']:.3f}  "
              f"P(max10>=1.33) {v['P_max_ge_1.33_k']['10']:.3f}")
        print(f"   sdhat5: med {v['sdhat5']['median']:.3f} P(<=0.074) {v['sdhat5']['P_le_0.074']:.3f}; "
              f"sdhat6: med {v['sdhat6']['median']:.3f} P(>=0.179) {v['sdhat6']['P_ge_0.179']:.3f}")
        print(f"   E[max]: k30 {v['E_max_k']['30']:.3f} k60 {v['E_max_k']['60']:.3f} "
              f"k107 {v['E_max_k']['107']:.3f}; P(max107>=1.44) {v['P_max_ge_walls_k107']['1.44']:.3f}")
    if attribution:
        print(f"\n=== attribution (B, nights>=1.30: {attribution['n_hot']}/{attribution['n_nights']}) ===")
        for p, a in sorted(attribution["games"].items(), key=lambda kv: -kv[1]["excess"]):
            print(f"{p}: contrib {a['mean_contrib']:.4f} -> hot {a['mean_contrib_hot']:.4f} "
                  f"(excess {a['excess']:+.4f}); lc {a['mean_lc']:.2f} -> {a['mean_lc_hot']:.2f}")
    print("\n=== normal-theory E[max] ===")
    for k, nm in normals.items():
        print(f"\n-- {nm['name']}: n={nm['n']} mean {nm['mean']:.3f} sd {nm['sd']:.3f} "
              f"CI sigma [{nm['sigma_ci95'][0]:.3f}, {nm['sigma_ci95'][1]:.3f}]")
        for tag in ("point", "ci_lo", "ci_hi"):
            f = nm["fixed_sigma"][tag]
            print(f"   {tag} sigma={f['sigma']:.3f}: "
                  + "  ".join(f"E[max{k}]={f['E_max_k'][str(k)]:.3f}" for k in KS_FUTURE)
                  + f"  P(max107>=1.44)={f['P_max_ge']['107']['1.44']:.3f}"
                  + f" >=1.56 {f['P_max_ge']['107']['1.56']:.3f}"
                  + f" >=1.61 {f['P_max_ge']['107']['1.61']:.3f}")
        pp = nm["posterior_predictive"]
        print("   postpred: " + "  ".join(
            f"E[max{k}]={pp[str(k)]['E_max']:.3f}" for k in KS_FUTURE)
            + f"  P(max107>=1.44)={pp['107']['P_max_ge']['1.44']:.3f}"
            + f" >=1.56 {pp['107']['P_max_ge']['1.56']:.3f}"
            + f" >=1.61 {pp['107']['P_max_ge']['1.61']:.3f}"
            + f"  P(1draw>=1.33)={pp['single_draw']['P_ge_1.33']:.4f}")
    print("\n=== window economics ===")
    print(json.dumps(econ, indent=2))
    (OUT / "report.md").write_text(REPORT, encoding="utf-8")
    print(f"\n[saved: {OUT / 'report.md'}]")


REPORT = """# LB draw-process generative model - stress test of the 07-18 "variance bombshell"

Date: 2026-07-18. Audience: panel R14 (Q-A window-pricing, Q-B frozen-vs-war allocation).
Code: `runs/lb_process_model/lb_process_model.py` (deterministic, seeded; regenerates this
report). Raw fits/sims: `runs/lb_process_model/fits_and_sims.json`.

**Discipline statement.** This analysis is DESCRIPTIVE / SIMULATION ONLY. It consumes no
gate look; the sealed A5/A8 n=5 war variance look (after tonight's draw #5) is untouched.
All inputs are already-on-record artifacts: the 13 fully-instrumented local seeds
(`runs/null10/` seeds 101-110 + `runs/kernel_pulls/war_eval_v1-3`, warpack but
delta-lc-null-ish) and the LB ledger numbers already published in ITERATION_LOG.md.
Local CPU only; no pushes, no API spend, no submissions.

## 0. Data, scorer, model

- **Scorer:** exact RHAE mirror from `scripts/phase1_gate.py`, re-validated this run:
  max abs error **1.78e-15** over 1000 cross-checks vs Tufa's 500 stored runs.
- **Ledgers (verified vs ITERATION_LOG):** frozen5 {0.82, 0.89, 0.93, 1.02, 0.95} mean
  0.922 sigma-hat 0.0740; frozen6 (+1.33) mean 0.990 sigma-hat 0.1792; war4
  {0.91, 1.08, 0.88, 1.05} mean 0.980 sigma-hat 0.0997; pooled10 mean 0.986 sigma-hat
  **0.1455** (chi2 95% CI on sigma, df 9: **[0.100, 0.266]**).
- **Per-game fit:** empirical 13-draw distributions per public game (score +
  levels_completed), scored offline with Tufa null baselines. 13-seed grand mean 1.594
  (25 games). Heavy tails confirmed: ft09 mean 8.98 sd 8.87 max 28.6 (lc 0-3); vc33
  3.59/4.67/16.7 (lc 0-3); tn36 1.85/3.10/10.7; ar25 3.52/3.05/8.3; r11l 3.36/1.82/4.8
  (lc <= 1 - r11l cannot spike).
- **LB night model:** score = mean over the FIXED official 110-game set, one pass each
  (gap_forensics 2026-07-09). Official games proxied by cloning the public 25 to 110 slots
  (multiplicity 4-5, randomized per world). Difficulty calibration: multiplicative c to hit
  a target mean - c = 0.922/1.594 = **0.58** (out-of-sample target, pre-1.33 mean) or
  0.986/1.594 = 0.62 (all-10-draw mean); this is the ~0.55 ratio of position_analysis refit
  on the 13 seeds. TRIM variant (drop ft09/vc33/r11l/ar25 = "official mix has fewer
  ft09-likes") run as sensitivity. Correlation variants: **IND** (all 110 slots
  independent), **COR** (IND + additive common night effect, ANOVA estimate from the 13
  seeds: sigma_u,local = 0.168 -> ~0.10 at LB scale), **BLOCK** (one seed drives all slots;
  upper bound, degenerate). 20,000 simulated nights per combo.

## 1. Headline 1 - does the local model reproduce the 1.33?

**Only partially, and only with cross-game correlation aboard.** In plain terms: if nightly
LB noise were nothing but independent per-game pass noise over 110 fixed games, a 1.33 draw
was a ~1-in-100 event in our observed window and the bombshell would be evidence of
something the bench doesn't capture. Adding the *measured* local common-night effect
(and/or accepting that the true frozen-fork mean is nearer 0.99 than 0.92) makes 1.33 an
unlucky-but-ordinary right-tail draw.

| model (mean calib, corr) | sd(night) | P(night>=1.33) | P(>=1.33 in 6) | in 10 | in 13 |
|---|---|---|---|---|---|
| A scale->0.922, IND | 0.128 | 0.0011 | 0.007 | 0.011 | 0.014 |
| B scale->0.922, COR | 0.161 | 0.0069 | **0.041** | 0.067 | 0.086 |
| D scale->0.986, IND | 0.137 | 0.0079 | 0.046 | 0.076 | 0.098 |
| E scale->0.986, COR | 0.172 | 0.0251 | **0.142** | 0.225 | 0.281 |
| F trim->0.922, IND | 0.109 | 0.0003 | 0.002 | 0.003 | 0.004 |
| G trim->0.922, COR | 0.195 | 0.0200 | 0.114 | 0.183 | 0.232 |
| C scale->0.922, BLOCK (bound) | 0.284 | 0.135 | 0.580 | 0.765 | 0.837 |
| H scale->0.922, IND, +Tufa20 pool | 0.121 | 0.0006 | 0.003 | 0.005 | 0.007 |

- The **pooled-10 observed sigma-hat 0.1455 sits exactly inside the model bracket**
  (IND 0.11-0.14, COR 0.16-0.20). The model does NOT need new physics to match the ledger's
  dispersion.
- The **sigma-hat5 = 0.074 of 07-12 was a sampling fluke, not a property of the process**:
  under the model, P(5-night sample sd <= 0.074) = 0.05-0.14 (IND) and 0.04-0.07 (COR).
  Unlucky-tight, not impossible. Symmetrically, P(6-night sigma-hat >= 0.179) = 0.02-0.13
  (IND), 0.28-0.53 (COR): the brief's 0.179 point estimate is itself the noisy side of an
  n=6 look.
- **Which games flip (attribution, COR@0.922, nights >= 1.30):** ft09-family clones carry
  **+0.152** of the required ~+0.35 exceedance (44%), with mean ft09 lc rising 1.31 ->
  **2.00** - i.e. a 1.33 night is, first of all, an **"ft09-analogues complete level 2
  efficiently"** night. vc33 adds +0.055 (16%, lc 1.46->1.74), tn36 +0.021, ar25 +0.015,
  re86 +0.009; **r11l contributes almost nothing (+0.007)** - its local distribution is
  capped at lc 1 and cannot spike. R14-question ranking: **ft09 2-level >> vc33 > tn36 >>
  r11l**.

## 2. Headline 2 - regime verdict (R13's load-bearing question)

**No regime-transfer gap is required to explain 1.33; the "8h LB plays deeper than the
bench" hypothesis is NOT supported as a necessary claim - but the pure-independence model
IS refuted, so *something* night-correlated is real.** Quantified:

- The needed depth already exists locally: ft09 lc=2-3 and vc33 lc=2-3 appear in the 13
  bench seeds. Simulated 1.33 nights are built from outcome patterns the bench has produced.
- To make a single night >= 1.33 a >=5% event you need either night-sd >= ~0.19 or true
  mean >= ~1.05. The model delivers sd 0.16-0.20 through the measured common-night effect
  (vLLM/server health, sampling-temperature luck shared across all 110 games in one 8h
  run) - no extra depth needed. What the model **cannot** do is produce 1.33 from
  independent per-game noise at mean 0.922 (p ~ 0.001/night). The honest residual: either
  (a) the common-night effect transfers to the LB rail, or (b) the frozen fork's true
  official mean is ~0.95-1.00 (first five draws mildly unlucky - consistent with
  gap_forensics placing our early draws at the 7th-28th pct), or (c) some combination. All
  three are ordinary; none require "the LB regime plays deeper".
- **Gap quantification for the record:** IND@0.922 under-disperses reality (predicted sd
  0.13 vs pooled observed 0.146, and P(1.33 event) ~1%); COR@0.986 slightly over-disperses
  (0.172). The truth is inside the bracket. Nothing about the ledger, including 1.33, is
  >2 sigma outside the local generative family.

## 3. Headline 3 - honest E[max] table and what a filler window actually buys

The brief's arithmetic replicates (frozen6 point sigma-hat 0.179 -> E[max@107] = 1.444).
But the point estimate is the fragile edge of the honest range. Full table ("postpred" =
posterior predictive integrating sigma AND mu uncertainty, noninformative prior; walls at
107 remaining windows):

| ledger / model | sigma basis | E[max@30] | E[max@60] | E[max@107] | P(max@107>=1.44) | >=1.56 | >=1.61 |
|---|---|---|---|---|---|---|---|
| frozen5 (pre-1.33) | 0.074 point | 1.07 | 1.09 | 1.11 | 0.00 | 0.00 | 0.00 |
| frozen6 | 0.179 point | 1.36 | 1.41 | **1.44** | 0.48 | 0.08 | 0.03 |
| frozen6 | CI [0.112, 0.440] | 1.22-1.89 | 1.25-2.01 | 1.27-2.10 | 0.00-1.00 | 0.00-1.00 | 0.00-1.00 |
| frozen6 | postpred | 1.43 | 1.49 | 1.53 | 0.57 | 0.34 | 0.28 |
| **pooled10** (frozen6+war4) | 0.146 point | 1.28 | 1.32 | 1.36 | 0.09 | 0.00 | 0.00 |
| **pooled10** | CI [0.100, 0.266] | 1.19-1.53 | 1.22-1.60 | 1.24-1.66 | 0.00-0.99 | 0.00-0.81 | 0.00-0.64 |
| **pooled10** | postpred | 1.31 | 1.36 | **1.39** | **0.29** | 0.11 | 0.07 |
| generative model (IND-COR, both calibs) | sim | 1.19-1.34 | 1.23-1.39 | 1.26-1.43 | 0.01-0.39 | - | - |

**On poolability (R14 Q-B input, argued both ways):** For: war4 is the same duck harness +
warpack whose compound gate just FAILED both prongs (no measurable mechanism effect;
delta-lc-null-ish in all three instrumented evals), war mean 0.980 ~ frozen6 mean 0.990,
and Welch t on war4-vs-frozen6 is ~0.1 - statistically indistinguishable, so treating war4
as quasi-control draws of the same process is defensible and buys df 9 instead of df 5.
Against: warpack is formally UNTESTED-IN-REGIME (A9) and its banking/soft-end paths could
in principle truncate right tails (max war draw 1.08 vs frozen 1.33), so pooling may
slightly shrink sigma-hat. Both ledgers are therefore reported; they agree on the
decision-relevant band: **E[max@107] ~ 1.35-1.53, P(touch 1.44 wall) ~ 10-50%, central
~30%.**

**Decision-relevant conclusion (window pricing):**

- The 07-14 ruling "order stats never break the wall" is overturned in its absolute form,
  but the brief's "expected max AT the wall" is the optimistic edge: the honest statement
  is **E[max@107] ~ 1.39 (pooled postpred), P(reach 1.44) ~ 0.3** - filler is a genuine
  wall-path lottery, not a wall-path plan. P(reach the 1.56-1.61 top-5 band) ~ 0.07-0.11;
  P(reach 1.86) ~ 0.01. Volume alone still does not credibly reach the leader.
- **One marginal filler window buys P(new campaign best, >=1.33) ~ 0.025 (pooled postpred;
  0.007-0.025 under the generative model)** and ~0.01 of touching 1.44. Marginal E[max]
  value decays fast: E[max@30]->E[max@107] adds only ~+0.08 over 77 windows
  (~0.001/window at the tail).
- **Break-even for an experimental draw** (same sigma, true official mean lift delta,
  infra-death prob q): to match the filler's P(new-best), an experimental build needs
  delta >= **+0.06** (q=0), **+0.07** (q=0.15), **+0.08** (q=0.30) under pooled10;
  +0.08/+0.09/+0.11 under frozen6. So: **filler strictly dominates null-EV and
  hygiene-only experimental draws, but does NOT dominate any candidate credibly claiming
  >= +0.10-0.12 official** - i.e. the existing A10/track thresholds (+0.12) already price
  windows correctly. "Filler-as-strategy" is the right default for windows with no gated
  candidate; it is not a reason to stop shipping gated candidates.

## 4. Caveats

1. Official-games proxy: 110 unknown games modeled as clones of the public 25 (SCALE) or a
   trimmed subset (TRIM). Truth is a different game population; SCALE/TRIM bracket the
   plausible variance structures, but a genuinely different official tail game would evade
   both.
2. The common-night effect (sigma_u,local 0.168) is estimated from 13 seeds on the pod rail
   (F-ratio not individually significant); its transfer to the Kaggle 8h rail is assumed in
   COR, not proven. IND and COR are presented as a bracket for exactly this reason.
3. The 13 seeds pool 10 vanilla + 3 warpack-eval runs (delta-lc-null-ish); a null10-only
   and a +Tufa-20-pass fit were run as sensitivities and do not change any conclusion.
4. Stationarity assumed across nights (no game-version drift term); the 07-11
   monotone-drift watch was cleared on the record, but drift would inflate observed
   sigma-hat relative to any stationary model.
5. LOO fragility of frozen6 sigma-hat 0.179 is inherited by everything using it - which is
   why the pooled10 and generative-model rows are the recommended pricing basis.

## 5. Three headline answers (plain sentences)

1. **Does the local model reproduce the 1.33?** Yes, but only as a right-tail event and
   only if cross-game night correlation (or a true mean near 0.99) carries to the LB: with
   the measured common-night effect the model gives a 4-14% chance of seeing >=1.33 within
   the observed 6-10 draws; from independent per-game noise alone it is ~1%, effectively
   ruling that reading out. A simulated 1.33 night is primarily an "ft09-analogues clear
   level 2" night (44% of the exceedance), with vc33 second (16%); r11l contributes
   nothing.
2. **Regime verdict:** the 8h LB regime does NOT need to play deeper than the bench - every
   ingredient of a 1.33 night already exists in the 13 local seeds, and the pooled-10
   observed sigma-hat 0.146 falls inside the model's 0.11-0.20 bracket. What is refuted is
   the independence assumption (and the old 0.074 point sigma, which the model shows was a
   ~1-in-10 lucky-tight n=5 sample), not the bench's depth calibration.
3. **Honest E[max]:** the brief's 1.44@110 is the optimistic point-estimate edge. Central,
   honest numbers: E[max@107] ~ 1.39 (pooled-10 posterior predictive; 1.26-1.53 across
   ledgers/models), P(touch 1.44) ~ 30% (10-50%), P(reach 1.56+) ~ 10%. A marginal filler
   window buys ~2.5% chance of a new campaign best; an experimental window beats that only
   if its build credibly claims >= +0.06-0.11 true official lift after infra-death risk -
   so filler dominates null-EV draws, and the +0.12 gate thresholds already separate the
   two correctly.
"""


if __name__ == "__main__":
    main()
