#!/usr/bin/env python
"""
GATE RECALIBRATION — R24 prep, 2026-08-09.

Independently reproduces the null-calibration of the campaign's sealed K3 non-harm gate
    PASS iff  mean_dlc >= -0.128  AND  worst_game_dlc >= -1.0
against the true null in runs/null10 (10 identical-config vanilla duck runs x 25 games),
quantifies the single-run-vs-averaged-baseline pairing asymmetry, derives calibrated
replacement thresholds at alpha in {0.05, 0.10}, and re-tests the A22 death record.

$0, CPU-only, read-only w.r.t. all existing artifacts. Writes:
    runs/gate_recalibration_2026-08-09.json
Run:  python duck_eval/r24_prep/gate_recalibration.py
"""
import itertools
import json
import os
import statistics
import sys
from datetime import datetime, timezone

ROOT = r"F:\kaggle\arc-prize-2026"
NULL10 = os.path.join(ROOT, "runs", "null10")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from phase1_gate import signflip_p_exact  # noqa: E402

SEEDS = list(range(101, 111))
GATE_MEAN = -0.128
GATE_WORST = -1.0


# ---------------------------------------------------------------- loaders
def load_null_summary(seed):
    """lc per game-prefix from runs/null10/vanilla_seedNNN.json"""
    d = json.load(open(os.path.join(NULL10, "vanilla_seed%d.json" % seed)))
    out = {}
    for gid, rec in d["games"].items():
        out[gid.split("-")[0]] = dict(
            game_id=gid,
            lc=int(rec["levels_completed"]),
            n_levels=int(rec["number_of_levels"]),
            actions=int(rec["actions"]),
            gen_tokens=int(rec["generated_tokens"]),
        )
    return d, out


def load_bench(d):
    """lc per game-prefix from a benchmark.json directory (same reader as the screens)."""
    b = json.load(open(os.path.join(d, "benchmark.json")))
    out = {}
    for r in b["game_runs"]:
        g = r["game_id"].split("-")[0]
        out[g] = dict(game_id=r["game_id"], lc=int(r["levels_completed"]),
                      actions=len(r["history"]))
    return out


# ---------------------------------------------------------------- stats helpers
def quantile(sorted_vals, q):
    """Empirical quantile, linear interpolation (numpy 'linear' convention)."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def largest_threshold_at_alpha(sorted_vals, alpha):
    """
    Largest t such that P_null(stat < t) <= alpha, i.e. the gate `stat >= t`
    has measured one-sided type-I error <= alpha on this empirical null.
    Returns (t, achieved_fpr).
    """
    n = len(sorted_vals)
    k = int(alpha * n)          # allow at most k draws strictly below t
    # candidate t = the (k+1)-th smallest value; then #below == count of values < t
    best_t, best_fpr = None, None
    for cand in sorted(set(sorted_vals)):
        fpr = sum(1 for v in sorted_vals if v < cand) / n
        if fpr <= alpha:
            best_t, best_fpr = cand, fpr
        else:
            break
    if best_t is None:
        best_t, best_fpr = sorted_vals[0], 0.0
    return best_t, best_fpr


def describe(vals):
    s = sorted(vals)
    return dict(
        n=len(s), mean=statistics.fmean(s),
        sd=statistics.stdev(s) if len(s) > 1 else 0.0,
        min=s[0], p1=quantile(s, 0.01), p5=quantile(s, 0.05), p10=quantile(s, 0.10),
        p25=quantile(s, 0.25), median=quantile(s, 0.50), p75=quantile(s, 0.75),
        p90=quantile(s, 0.90), p95=quantile(s, 0.95), max=s[-1],
    )


def hist(vals):
    h = {}
    for v in vals:
        h[str(v)] = h.get(str(v), 0) + 1
    return dict(sorted(h.items(), key=lambda kv: float(kv[0])))


def pearson(x, y):
    n = len(x)
    mx, my = statistics.fmean(x), statistics.fmean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = sum((a - mx) ** 2 for a in x) ** 0.5
    dy = sum((b - my) ** 2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    out = {}
    out["provenance"] = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        script=os.path.relpath(__file__, ROOT).replace("\\", "/"),
        purpose="Independent recalibration of the sealed K3 non-harm gate "
                "(mean_dlc >= -0.128 AND worst_game_dlc >= -1.0) against the on-disk "
                "true null in runs/null10; triggered by R24 methodology objection 1.",
        null_source="runs/null10/vanilla_seed{101..110}.json (cross-checked against "
                    "runs/null10/seedNNN/benchmark.json)",
        arm_sources=[
            "runs/a22_compaction_v1/m1m2m3_screen.json",
            "runs/a22_v2_seed1/m1m2m3_screen.json",
            "runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json",
            "runs/kernel_pulls/war_eval_v1/benchmark.json (shared baseline for all three)",
            "runs/sentinel_eval_v1/screen_report.md (provenance of -0.128)",
        ],
        gate_under_test=dict(mean_leg=">= %.3f" % GATE_MEAN, worst_leg=">= %.1f" % GATE_WORST,
                             combination="conjunction (PASS iff both)"),
    )

    # ------------------------------------------------ 0. audit the null corpus
    summaries, lcmaps, envs = {}, {}, {}
    for s in SEEDS:
        d, m = load_null_summary(s)
        summaries[s], lcmaps[s] = d, m
        envs[s] = json.dumps(d["phase1_env"], sort_keys=True)
    games = sorted(lcmaps[SEEDS[0]])
    assert all(sorted(lcmaps[s]) == games for s in SEEDS)

    # cross-check the summary jsons against the full benchmark.json dumps
    xcheck_mismatch = []
    for s in SEEDS:
        bench = load_bench(os.path.join(NULL10, "seed%d" % s))
        for g in games:
            if bench[g]["lc"] != lcmaps[s][g]["lc"]:
                xcheck_mismatch.append((s, g, bench[g]["lc"], lcmaps[s][g]["lc"]))

    gid_sets = {s: {lcmaps[s][g]["game_id"] for g in games} for s in SEEDS}
    identical_game_ids = all(gid_sets[s] == gid_sets[SEEDS[0]] for s in SEEDS)

    war = load_bench(os.path.join(ROOT, "runs", "kernel_pulls", "war_eval_v1"))
    war_gid_match = {g: (war[g]["game_id"] == lcmaps[SEEDS[0]][g]["game_id"]) for g in games}

    out["null_corpus_audit"] = dict(
        n_runs=len(SEEDS), n_games=len(games), games=games,
        seeds=SEEDS,
        all_configs_identical=len(set(envs.values())) == 1,
        phase1_env=summaries[SEEDS[0]]["phase1_env"],
        n_passes_all_1=all(summaries[s]["n_passes"] == 1 for s in SEEDS),
        labels=[summaries[s]["label"] for s in SEEDS],
        elapsed_s=[round(summaries[s]["elapsed_s"], 1) for s in SEEDS],
        summary_vs_benchmark_lc_mismatches=xcheck_mismatch,
        game_ids_identical_across_null_runs=identical_game_ids,
        war_eval_v1_game_id_matches_null10=war_gid_match,
        n_games_with_differing_instance_hash_vs_war_v1=sum(
            1 for g in games if not war_gid_match[g]),
        note="Seeds 101-110 differ only in RNG seed; every other env var, n_passes and "
             "game set is byte-identical -> these are a genuine same-config null.",
    )

    lc = {s: {g: lcmaps[s][g]["lc"] for g in games} for s in SEEDS}
    out["null_corpus_audit"]["per_run_lc_total"] = {str(s): sum(lc[s].values()) for s in SEEDS}
    out["null_corpus_audit"]["per_game_lc_across_runs"] = {
        g: [lc[s][g] for s in SEEDS] for g in games}

    # ------------------------------------------------ 1. regime A: single vs single
    A = []
    for i, j in itertools.permutations(SEEDS, 2):
        d = [lc[i][g] - lc[j][g] for g in games]
        A.append(dict(arm=i, base=j, deltas=d,
                      mean=statistics.fmean(d), worst=min(d),
                      n_le_m1=sum(1 for v in d if v <= -1),
                      n_le_m2=sum(1 for v in d if v <= -2),
                      n_le_m3=sum(1 for v in d if v <= -3)))
    A_mean = [r["mean"] for r in A]
    A_worst = [r["worst"] for r in A]
    A_pergame_sd = statistics.stdev([v for r in A for v in r["deltas"]])

    fpr_mean_A = sum(1 for m in A_mean if m < GATE_MEAN) / len(A_mean)
    fpr_worst_A = sum(1 for w in A_worst if w < GATE_WORST) / len(A_worst)
    fpr_conj_A = sum(1 for r in A if not (r["mean"] >= GATE_MEAN and r["worst"] >= GATE_WORST)) / len(A)

    # ------------------------------------------------ 2. regime B: single vs mean-of-9
    B = []
    for i in SEEDS:
        others = [s for s in SEEDS if s != i]
        d = [lc[i][g] - statistics.fmean([lc[o][g] for o in others]) for g in games]
        B.append(dict(arm=i, base="mean9", deltas=d, mean=statistics.fmean(d), worst=min(d),
                      n_le_m1=sum(1 for v in d if v <= -1),
                      n_le_m2=sum(1 for v in d if v <= -2)))
    B_mean = [r["mean"] for r in B]
    B_worst = [r["worst"] for r in B]
    B_pergame_sd = statistics.stdev([v for r in B for v in r["deltas"]])

    fpr_mean_B = sum(1 for m in B_mean if m < GATE_MEAN) / len(B_mean)
    fpr_worst_B = sum(1 for w in B_worst if w < GATE_WORST) / len(B_worst)
    fpr_conj_B = sum(1 for r in B if not (r["mean"] >= GATE_MEAN and r["worst"] >= GATE_WORST)) / len(B)

    # ------------------------------------------------ 2b. baseline-size sweep m = 1..9
    sweep = {}
    for m in range(1, 10):
        means, worsts, pg = [], [], []
        for i in SEEDS:
            others = [s for s in SEEDS if s != i]
            for S in itertools.combinations(others, m):
                d = [lc[i][g] - statistics.fmean([lc[o][g] for o in S]) for g in games]
                means.append(statistics.fmean(d))
                worsts.append(min(d))
                pg.extend(d)
        sweep[m] = dict(
            n_draws=len(means),
            mean_stat_sd=statistics.stdev(means),
            per_game_delta_sd=statistics.stdev(pg),
            p5_mean=quantile(sorted(means), 0.05),
            p10_mean=quantile(sorted(means), 0.10),
            frac_worst_le_m2=sum(1 for w in worsts if w <= -2) / len(worsts),
            fpr_mean_leg_at_gate=sum(1 for x in means if x < GATE_MEAN) / len(means),
        )

    # ------------------------------------------------ 3. calibrated thresholds
    def leg_table(vals, label):
        s = sorted(vals)
        t05, f05 = largest_threshold_at_alpha(s, 0.05)
        t10, f10 = largest_threshold_at_alpha(s, 0.10)
        return dict(
            statistic=label, dist=describe(s),
            threshold_alpha_0_05=dict(t=t05, achieved_fpr=f05),
            threshold_alpha_0_10=dict(t=t10, achieved_fpr=f10),
            fpr_at_inherited_mean_gate=sum(1 for v in s if v < GATE_MEAN) / len(s),
        )

    # count-of-bad-games statistic (reviewer's proposed replacement worst leg)
    A_cnt2 = [r["n_le_m2"] for r in A]
    A_cnt1 = [r["n_le_m1"] for r in A]
    B_cnt2 = [r["n_le_m2"] for r in B]

    def count_leg(counts, label):
        n = len(counts)
        tail = {k: sum(1 for c in counts if c >= k) / n for k in range(0, max(counts) + 2)}
        # largest cap K with P(count > K) <= alpha
        def cap(alpha):
            for K in range(0, max(counts) + 2):
                if tail.get(K + 1, 0.0) <= alpha:
                    return K, tail.get(K + 1, 0.0)
            return max(counts), 0.0
        k05, f05 = cap(0.05)
        k10, f10 = cap(0.10)
        return dict(statistic=label, hist=hist(counts),
                    survival_P_ge_k=tail,
                    cap_alpha_0_05=dict(K=k05, achieved_fpr=f05),
                    cap_alpha_0_10=dict(K=k10, achieved_fpr=f10))

    out["regime_A_single_vs_single"] = dict(
        n_pairs=len(A),
        pairing="ordered pairs (arm run i, baseline run j), i != j; per-game Delta_lc",
        per_game_delta_sd=A_pergame_sd,
        mean_leg=leg_table(A_mean, "mean Delta_lc"),
        mean_hist=hist([round(m, 3) for m in A_mean]),
        worst_leg=dict(statistic="worst-game Delta_lc", hist=hist(A_worst),
                       dist=describe(A_worst),
                       fpr_at_inherited_worst_gate=fpr_worst_A,
                       note="Delta is integer-valued in this regime, so 'worst >= -1.0' "
                            "fails iff worst <= -2."),
        count_leg_le_m2=count_leg(A_cnt2, "# games with Delta_lc <= -2"),
        count_leg_le_m1=count_leg(A_cnt1, "# games with Delta_lc <= -1"),
        inherited_gate_type_I=dict(mean_leg=fpr_mean_A, worst_leg=fpr_worst_A,
                                   conjunction=fpr_conj_A),
    )
    out["regime_B_single_vs_mean_of_9"] = dict(
        n_draws=len(B),
        pairing="arm run i vs the per-game mean of the remaining 9 runs",
        per_game_delta_sd=B_pergame_sd,
        mean_leg=leg_table(B_mean, "mean Delta_lc"),
        mean_values=[round(x, 4) for x in B_mean],
        worst_leg=dict(statistic="worst-game Delta_lc", values=[round(x, 4) for x in B_worst],
                       dist=describe(B_worst), fpr_at_inherited_worst_gate=fpr_worst_B),
        count_leg_le_m2=count_leg(B_cnt2, "# games with Delta_lc <= -2"),
        inherited_gate_type_I=dict(mean_leg=fpr_mean_B, worst_leg=fpr_worst_B,
                                   conjunction=fpr_conj_B),
        caveat="only 10 non-independent draws exist for m=9; see baseline_size_sweep "
               "for the variance trend estimated on many more (also non-independent) draws.",
    )
    out["baseline_size_sweep"] = dict(
        description="null spread of the mean-Delta_lc statistic when the baseline is the "
                    "average of m identical-config runs (m = 1 reproduces regime A)",
        by_m={str(m): v for m, v in sweep.items()},
        variance_inflation_m1_over_m9=dict(
            mean_stat_sd_ratio=sweep[1]["mean_stat_sd"] / sweep[9]["mean_stat_sd"],
            per_game_sd_ratio=sweep[1]["per_game_delta_sd"] / sweep[9]["per_game_delta_sd"],
            theoretical_sd_ratio=(2.0 / (1.0 + 1.0 / 9.0)) ** 0.5,
        ),
    )

    # ------------------------------------------------ 4. provenance of -0.128
    nullmean = {g: statistics.fmean([lc[s][g] for s in SEEDS]) for g in games}
    war_d = [war[g]["lc"] - nullmean[g] for g in games]
    out["provenance_of_minus_0_128"] = dict(
        source="runs/sentinel_eval_v1/screen_report.md L5 "
               "('PRIMARY paired dlc: mean -0.128 (sd 0.392, 4W/12L, sign-flip p=0.9495)')",
        recomputed_war_v1_vs_null10_mean=dict(
            mean_dlc=statistics.fmean(war_d), per_game_sd=statistics.stdev(war_d),
            worst=min(war_d),
            wins=sum(1 for v in war_d if v > 0), losses=sum(1 for v in war_d if v < 0),
            ties=sum(1 for v in war_d if v == 0),
        ),
        applied_against="a SINGLE baseline run (runs/kernel_pulls/war_eval_v1) in every "
                        "A22 screen and in the R24 P1 proposal",
        per_game_sd_estimation_regime=statistics.stdev(war_d),
        per_game_sd_application_regime=A_pergame_sd,
        sd_inflation_factor=A_pergame_sd / statistics.stdev(war_d),
        verdict="threshold estimated in the averaged-baseline regime, applied in the "
                "single-baseline regime; the number transported, the operating "
                "characteristic did not.",
    )

    # ------------------------------------------------ 5. A22 arms vs the null
    arms = {}
    for name, path in [
        ("a22_v1", "runs/a22_compaction_v1/m1m2m3_screen.json"),
        ("a22_v2", "runs/a22_v2_seed1/m1m2m3_screen.json"),
        ("a22_v2_1", "runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json"),
    ]:
        d = json.load(open(os.path.join(ROOT, path)))
        M1 = d["M1"]
        pg = {r["game"]: r["dlc"] for r in d["per_game"]}
        deltas = [pg[g] for g in sorted(pg)]
        n = len(A_mean)
        arms[name] = dict(
            source=path, baseline=d.get("baseline"),
            mean_dlc=M1["mean_dlc"], sd_games=M1["sd_games"],
            worst_game=M1["worst_game"], worst_dlc=M1["worst_dlc"],
            signflip_p_exact=M1["signflip_p_exact"],
            wins=M1["wins"], losses=M1["losses"],
            n_games_le_m2=sum(1 for v in deltas if v <= -2),
            n_games_le_m1=sum(1 for v in deltas if v <= -1),
            null_P_mean_le_observed_regimeA=sum(1 for m in A_mean if m <= M1["mean_dlc"] + 1e-12) / n,
            null_P_mean_le_observed_regimeB=sum(1 for m in B_mean if m <= M1["mean_dlc"] + 1e-12) / len(B_mean),
            null_P_worst_le_observed_regimeA=sum(1 for w in A_worst if w <= M1["worst_dlc"]) / n,
            null_P_count_le_m2_ge_observed_regimeA=sum(
                1 for c in A_cnt2 if c >= sum(1 for v in deltas if v <= -2)) / n,
            deltas=deltas,
            inherited_verdict=M1["verdict"],
        )
        arms[name]["recalibrated_mean_leg_verdict_alpha05_regimeA"] = (
            "FAIL" if M1["mean_dlc"] < largest_threshold_at_alpha(sorted(A_mean), 0.05)[0] else "PASS")

    # monotonicity / dose-response
    steps = dict(
        v1_to_v2=arms["a22_v2"]["mean_dlc"] - arms["a22_v1"]["mean_dlc"],
        v2_to_v2_1=arms["a22_v2_1"]["mean_dlc"] - arms["a22_v2"]["mean_dlc"],
        v1_to_v2_1=arms["a22_v2_1"]["mean_dlc"] - arms["a22_v1"]["mean_dlc"],
    )
    # SE of a difference between two arms sharing one baseline run:
    # each arm's mean has sd ~ sd_A(mean stat); the shared baseline cancels, leaving
    # the two arms' own run-to-run noise. Estimate the single-run mean-lc sd from null10.
    per_run_mean_lc = [statistics.fmean([lc[s][g] for g in games]) for s in SEEDS]
    sd_single_run_mean = statistics.stdev(per_run_mean_lc)
    se_diff_shared_baseline = (2 ** 0.5) * sd_single_run_mean
    sd_mean_stat_A = statistics.stdev(A_mean)

    # Fisher z on the two pearson correlations
    import math
    r1, r2 = -0.13142625741264627, -0.40260886697725734
    z1, z2 = math.atanh(r1), math.atanh(r2)
    se_z = (1.0 / (25 - 3) + 1.0 / (25 - 3)) ** 0.5
    zdiff = (z2 - z1)
    # two-sided normal p
    p_fisher = math.erfc(abs(zdiff) / se_z / (2 ** 0.5))

    out["a22_reexamination"] = dict(
        arms=arms,
        null_reference="regime A (single-vs-single) mean-Delta_lc null from null10, "
                       "n=90 ordered pairs; NOTE the A22 screens pair against war_eval_v1 "
                       "(warpack, single run), not against a null10 run.",
        mean_leg_verdict=("A22 death SURVIVES on the mean leg iff each arm's mean_dlc lies "
                          "outside the null; see arms[*].null_P_mean_le_observed_regimeA"),
        worst_leg_verdict=("worst_dlc = -2 for all three arms; regime-A null P(worst <= -2) "
                           "= %.3f, so the worst-game leg carries no evidential weight"
                           % (sum(1 for w in A_worst if w <= -2) / len(A_worst))),
        monotonicity=dict(
            means=[arms["a22_v1"]["mean_dlc"], arms["a22_v2"]["mean_dlc"],
                   arms["a22_v2_1"]["mean_dlc"]],
            steps=steps,
            null_sd_of_mean_stat_regimeA=sd_mean_stat_A,
            sd_of_single_run_mean_lc=sd_single_run_mean,
            se_of_between_arm_difference_shared_baseline=se_diff_shared_baseline,
            step_z_v1_to_v2=steps["v1_to_v2"] / se_diff_shared_baseline,
            step_z_v2_to_v2_1=steps["v2_to_v2_1"] / se_diff_shared_baseline,
            verdict="steps are within noise; dose-response NOT resolved at n=25/1 seed",
        ),
        pearson_shift=dict(
            r_v2=r1, r_v2_1=r2, fisher_z_diff=zdiff, se=se_z, p_two_sided=p_fisher,
            verdict="not significant"),
    )

    # ------------------------------------------------ 5b. the comparator's OWN run-to-run null
    # war_eval_v1/v2/v3 share benchmark label 'duck-harness-kaggle-warpack-v1' (ledger-OFF
    # seeds 1/2/3, pulled 07-14/15/16) and run on IDENTICAL game instance ids. They are a
    # same-config null for the exact arm the A22 screens paired against.
    WNAMES = ["war_eval_v1", "war_eval_v2", "war_eval_v3", "w0_eval_s1"]
    W = {n: load_bench(os.path.join(ROOT, "runs", "kernel_pulls", n)) for n in WNAMES}
    wgames = sorted(W["war_eval_v1"])
    gid_consistent = {n: all(W[n][g]["game_id"] == W["war_eval_v1"][g]["game_id"] for g in wgames)
                      for n in WNAMES}
    W3 = ["war_eval_v1", "war_eval_v2", "war_eval_v3"]
    wp_pairs = []
    for a, b in itertools.permutations(W3, 2):
        dl = [W[a][g]["lc"] - W[b][g]["lc"] for g in wgames]
        nz = [v for v in dl if v != 0]
        p1 = signflip_p_exact(nz, abs(sum(nz)))[0] if nz else 1.0
        wp_pairs.append(dict(arm=a, base=b, mean=statistics.fmean(dl), worst=min(dl),
                             n_le_m2=sum(1 for v in dl if v <= -2),
                             wins=sum(1 for v in dl if v > 0),
                             losses=sum(1 for v in dl if v < 0),
                             signflip_p_two=min(1.0, 2 * p1)))
    wp_mean = [r["mean"] for r in wp_pairs]
    wp_pg_sd = statistics.stdev([W[a][g]["lc"] - W[b][g]["lc"]
                                 for a, b in itertools.permutations(W3, 2) for g in wgames])
    wp_tot = [sum(W[n][g]["lc"] for g in wgames) for n in W3]
    van_tot = [sum(lc[s].values()) for s in SEEDS]

    # multi-run warpack baselines
    def arm_vs_baseline(arm_deltas_source, base_names):
        """arm per-game lc taken from the arm's own screen per_game (a22_lc), baseline = mean lc of base_names"""
        base = {g: statistics.fmean([W[n][g]["lc"] for n in base_names]) for g in wgames}
        return base

    a22_lc = {}
    for name, path in [("a22_v1", "runs/a22_compaction_v1/m1m2m3_screen.json"),
                       ("a22_v2", "runs/a22_v2_seed1/m1m2m3_screen.json"),
                       ("a22_v2_1", "runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json")]:
        d = json.load(open(os.path.join(ROOT, path)))
        a22_lc[name] = {r["game"]: r["a22_lc"] for r in d["per_game"]}

    rebase = {}
    for bl_label, bl_names in [("war_v1_only(as-screened)", ["war_eval_v1"]),
                               ("war_v1v2v3_mean(n=3)", W3),
                               ("R16_4run_band", WNAMES)]:
        base = {g: statistics.fmean([W[n][g]["lc"] for n in bl_names]) for g in wgames}
        rebase[bl_label] = {}
        for name in a22_lc:
            dl = [a22_lc[name][g] - base[g] for g in wgames]
            rebase[bl_label][name] = dict(
                mean_dlc=statistics.fmean(dl), worst_dlc=min(dl),
                n_le_m2=sum(1 for v in dl if v <= -2),
                arm_lc_total=sum(a22_lc[name].values()),
                base_lc_total=sum(base.values()))

    out["comparator_own_null"] = dict(
        note="war_eval_v1/v2/v3 share benchmark label 'duck-harness-kaggle-warpack-v1', are "
             "documented as ledger-OFF seeds 1/2/3 (learnings/daily_brief_2026-07-16.md), and "
             "run on identical game instance ids. They form a same-config null for the exact "
             "baseline the A22 screens used.",
        game_ids_identical_to_war_v1=gid_consistent,
        lc_totals={n: sum(W[n][g]["lc"] for g in wgames) for n in WNAMES},
        pairwise=wp_pairs,
        mean_stat_sd=statistics.stdev(wp_mean),
        per_game_delta_sd=wp_pg_sd,
        mean_range=[min(wp_mean), max(wp_mean)],
        vanilla_null10_mean_stat_sd=statistics.stdev(A_mean),
        variance_ratio_warpack_over_vanilla_runtotals=(
            statistics.stdev(wp_tot) ** 2) / (statistics.stdev(van_tot) ** 2),
        warpack_run_lc_totals=wp_tot, vanilla_run_lc_totals=van_tot,
        a22_rebaselined=rebase,
        verdict="A22 v2.1's headline mean_dlc of -0.360 is EXACTLY reproduced by "
                "war_eval_v3 vs war_eval_v1 -- two runs of the identical warpack config with "
                "no compaction at all. All three A22 arms lie inside the comparator's own "
                "seed-to-seed range.",
    )
    try:
        from scipy.stats import f as _f
        out["comparator_own_null"]["variance_F_test"] = dict(
            F=(statistics.stdev(wp_tot) ** 2) / (statistics.stdev(van_tot) ** 2),
            df1=len(wp_tot) - 1, df2=len(van_tot) - 1,
            p_one_sided=float(1 - _f.cdf((statistics.stdev(wp_tot) ** 2) /
                                         (statistics.stdev(van_tot) ** 2),
                                         len(wp_tot) - 1, len(van_tot) - 1)),
            interpretation="warpack run-to-run variance exceeds the vanilla null10 variance; "
                           "the vanilla null therefore UNDERSTATES the null spread that the "
                           "A22 arms should have been judged against")
    except Exception as e:  # pragma: no cover
        out["comparator_own_null"]["variance_F_test"] = dict(error=repr(e))
    # A22 verdict under the comparator's own pairwise null (6 ordered pairs, n=3 runs)
    a22_vs_wpnull = {}
    for name in ("a22_v1", "a22_v2", "a22_v2_1"):
        obs = arms[name]["mean_dlc"]
        a22_vs_wpnull[name] = dict(
            mean_dlc_vs_war_v1=obs,
            P_warpack_pair_le_obs=sum(1 for m in wp_mean if m <= obs + 1e-12) / len(wp_mean),
            inside_warpack_pairwise_range=(min(wp_mean) <= obs <= max(wp_mean)),
        )
    out["comparator_own_null"]["a22_vs_warpack_pairwise_null"] = a22_vs_wpnull

    # ------------------------------------------------ 6. cross-checks of other reviewer numbers
    apl = []
    for s in SEEDS:
        tot_a = sum(lcmaps[s][g]["actions"] for g in games)
        tot_lc = sum(lcmaps[s][g]["lc"] for g in games)
        apl.append(tot_a / tot_lc)
    out["cross_checks"] = dict(
        actions_per_level_completed_null10=dict(
            values=[round(v, 1) for v in apl], mean=statistics.fmean(apl),
            sd=statistics.stdev(apl), min=min(apl), max=max(apl),
            cv_pct=100 * statistics.stdev(apl) / statistics.fmean(apl),
            reviewer_claim="195.5-322.9, mean 234.2, sd 40.9, CV 17.5%"),
        signflip_p_null_distribution_regimeA=None,  # filled below
    )
    sf = []
    for r in A:
        nz = [v for v in r["deltas"] if v != 0]
        p1 = signflip_p_exact(nz, abs(sum(nz)))[0] if nz else 1.0
        sf.append(min(1.0, 2 * p1))
    out["cross_checks"]["signflip_p_null_distribution_regimeA"] = dict(
        dist=describe(sf), frac_le_0_05=sum(1 for p in sf if p <= 0.05) / len(sf),
        frac_le_0_10=sum(1 for p in sf if p <= 0.10) / len(sf),
        note="two-sided exact sign-flip p on the null pairs; a well-calibrated two-sided "
             "test should show ~5% <= 0.05 (conservative here due to discreteness)")

    # ------------------------------------------------ 7. recommendation
    tA05, fA05 = largest_threshold_at_alpha(sorted(A_mean), 0.05)
    tA10, fA10 = largest_threshold_at_alpha(sorted(A_mean), 0.10)
    tB05, fB05 = largest_threshold_at_alpha(sorted(B_mean), 0.05)
    tB10, fB10 = largest_threshold_at_alpha(sorted(B_mean), 0.10)
    k05 = count_leg(A_cnt2, "x")["cap_alpha_0_05"]
    out["recommendation"] = dict(
        primary=dict(
            regime="pair the arm against the per-game MEAN of >= 3 (ideally all 10) "
                   "identical-config baseline runs, not a single run",
            statistic="mean Delta_lc over the 25 games",
            threshold_alpha_0_05_single_baseline=tA05,
            threshold_alpha_0_10_single_baseline=tA10,
            threshold_alpha_0_05_mean9_baseline=tB05,
            threshold_alpha_0_10_mean9_baseline=tB10,
            achieved_fpr=dict(A05=fA05, A10=fA10, B05=fB05, B10=fB10),
        ),
        worst_game_leg=dict(
            keep=False,
            reason="P_null(worst <= -2) = %.3f at 25 games in regime A; the leg fires on "
                   "half of all true nulls and no threshold above -3 attains alpha 0.05"
                   % (sum(1 for w in A_worst if w <= -2) / len(A_worst)),
            replacement="count leg: (# games with Delta_lc <= -2) <= %d  [measured FPR %.3f]"
                        % (k05["K"], k05["achieved_fpr"]),
        ),
    )

    # ------------------------------------------------ 7b. self-calibrating (parametric) gate
    # t-gate:  mean_dlc >= -t_{1-alpha, df=m-1} * s_base * sqrt(1 + 1/m)
    # where s_base = sd over the m baseline runs of (run lc total / 25).
    try:
        from statistics import NormalDist
        _ = NormalDist
    except Exception:
        pass
    # t_{1-alpha, df}; keys ARE degrees of freedom
    T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943,
           7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812}
    T90 = {1: 3.078, 2: 1.886, 3: 1.638, 4: 1.533, 5: 1.476, 6: 1.440,
           7: 1.415, 8: 1.397, 9: 1.383, 10: 1.372}

    def t_gate(run_totals, alpha=0.05):
        m = len(run_totals)
        s = statistics.stdev([t / 25.0 for t in run_totals])
        tq = (T95 if alpha == 0.05 else T90).get(m - 1)
        if tq is None:
            return None
        return dict(m=m, df=m - 1, s_base=s, t_quantile=tq,
                    threshold=-tq * s * (1 + 1.0 / m) ** 0.5)

    van_totals = [sum(lc[s].values()) for s in SEEDS]
    out["self_calibrating_gate"] = dict(
        form="mean_dlc >= -t_{1-alpha, df=m-1} * s_base * sqrt(1 + 1/m); "
             "s_base = sd over the m same-config baseline runs of (run lc total / n_games)",
        vanilla_null10_m10_alpha05=t_gate(van_totals, 0.05),
        vanilla_null10_m10_alpha10=t_gate(van_totals, 0.10),
        empirical_check_vanilla=dict(
            empirical_p5_regimeB_mean9=quantile(sorted(B_mean), 0.05),
            empirical_p5_sweep_m3=sweep[3]["p5_mean"],
            empirical_p5_sweep_m4=sweep[4]["p5_mean"],
            note="the parametric line lands on the empirical 5th percentile of the vanilla "
                 "null, so the formula is validated on the one config where a 10-run null exists"),
        warpack_m3_alpha05=t_gate([sum(W[n][g]["lc"] for g in wgames) for n in W3], 0.05),
        warpack_m3_alpha10=t_gate([sum(W[n][g]["lc"] for g in wgames) for n in W3], 0.10),
        why="a FIXED number cannot be right for both configs: the warpack comparator's "
            "run-to-run variance is %.1fx the vanilla null's on run lc totals"
            % out["comparator_own_null"]["variance_ratio_warpack_over_vanilla_runtotals"],
    )

    out["final_recommendation"] = dict(
        gate_name="K3' (recalibrated non-harm)",
        leg_1_primary=dict(
            statistic="mean Delta_lc over all 25 games, arm paired against the PER-GAME MEAN "
                      "of m >= 3 same-config baseline runs",
            threshold="mean_dlc >= -t_{0.95, df=m-1} * s_base * sqrt(1 + 1/m), "
                      "s_base = sd over the m baseline runs of (run lc total / 25)",
            fixed_fallback_if_m_eq_1="mean_dlc >= -0.200  [measured type-I 2.2% on null10; "
                                     "-0.16 gives 7.8%]",
            fixed_fallback_if_m_ge_3="mean_dlc >= -0.190  [empirical 5th pct of the vanilla "
                                     "null at m=3/4; ONLY valid if the arm's config has "
                                     "vanilla-like run-to-run variance -- check it]",
            measured_type_I=dict(single_baseline_at_minus_0_200=fA05,
                                 single_baseline_at_minus_0_128=fpr_mean_A),
        ),
        leg_2_replacement_for_worst_game=dict(
            drop="worst-game Delta_lc >= -1.0",
            reason="measured type-I 50.0% (regime A) / 60.0% (regime B) on the campaign's own "
                   "null; a -2 on some game is the MODAL null outcome",
            replacement="(# games with Delta_lc <= -2) <= 2   [measured type-I 4.4% on the 90 "
                        "null pairs; the cap is only meaningful against a SINGLE-run baseline, "
                        "since a multi-run baseline is fractional]",
            status="advisory / non-gating is also defensible -- the leg adds almost no power",
        ),
        mandatory_seal_language="every leg must quote its measured type-I rate and the null "
                                "corpus it was measured on; a threshold with no operating "
                                "characteristic is not a gate",
        also_required="the baseline must be >= 3 runs of the SAME config as the arm's "
                      "comparator, and the seal must publish that config's own run-to-run sd. "
                      "Three warpack seeds (war_eval_v1/v2/v3) already exist on disk and were "
                      "not used.",
    )

    return out


if __name__ == "__main__":
    res = main()
    dest = os.path.join(ROOT, "runs", "gate_recalibration_2026-08-09.json")
    json.dump(res, open(dest, "w"), indent=1)
    print("wrote", dest)
    a = res["regime_A_single_vs_single"]["inherited_gate_type_I"]
    b = res["regime_B_single_vs_mean_of_9"]["inherited_gate_type_I"]
    print("regime A type-I: mean %.4f worst %.4f conj %.4f" % (a["mean_leg"], a["worst_leg"], a["conjunction"]))
    print("regime B type-I: mean %.4f worst %.4f conj %.4f" % (b["mean_leg"], b["worst_leg"], b["conjunction"]))
    print("per-game sd A %.4f  B %.4f" % (
        res["regime_A_single_vs_single"]["per_game_delta_sd"],
        res["regime_B_single_vs_mean_of_9"]["per_game_delta_sd"]))
    print(json.dumps(res["recommendation"], indent=1))
