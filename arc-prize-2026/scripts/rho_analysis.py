#!/usr/bin/env python
"""Estimate rho = corr(public_score, private_score) via subset-split resampling.

Local CPU only. Uses the validated RHAE scorer (scripts/phase1_gate.rhae_score).
No pushes/spend. Writes runs/rho_estimate/report.md + raw JSON.
"""
from __future__ import annotations
import json, math, random, statistics as st
from itertools import combinations
from pathlib import Path
import sys

ROOT = Path(r"F:\kaggle\arc-prize-2026")
sys.path.insert(0, str(ROOT / "scripts"))
from phase1_gate import rhae_score  # validated scorer

OUT = ROOT / "runs" / "rho_estimate"
OUT.mkdir(parents=True, exist_ok=True)
RNG = random.Random(20260809)

# ----------------------------------------------------------------- load
def load_bench(path):
    d = json.loads(Path(path).read_text())
    runs = d["game_runs"] if isinstance(d, dict) else d
    return d, runs

def recompute_check(runs):
    """Recompute rhae from raw, compare to stored final_score. Returns max_err, n."""
    max_err = 0.0; n = 0
    for r in runs:
        if r.get("final_score") is None:
            continue
        s = rhae_score(r.get("base_actions_per_level"), r["actions_per_level"],
                       r["levels_completed"], r["number_of_levels"])
        max_err = max(max_err, abs(s - r["final_score"])); n += 1
    return max_err, n

def per_game(runs):
    """{game_id: final_score}. n_passes==1 for all our runs."""
    return {r["game_id"]: r["final_score"] for r in runs}

# same-gset runs (gset=3acf6354, 25 games). compaction_v2 == a22_v2_1 (dropped).
KP = ROOT / "runs" / "kernel_pulls"
CONFIG_RUNS = {
    "sentinel_eval_v1": KP/"sentinel_eval_v1"/"benchmark.json",
    "war_v2_eval_s1":   KP/"war_v2_eval_s1"/"benchmark.json",
    "w0_cont_eval":     KP/"w0_cont_eval"/"benchmark.json",
    "war_eval_v3":      KP/"war_eval_v3"/"benchmark.json",
    "sched_v1":         KP/"sched_v1"/"benchmark.json",
    "gate_eval_v1":     KP/"gate_eval_v1"/"benchmark.json",
    "war_eval_v1":      KP/"war_eval_v1"/"benchmark.json",
    "war_eval_v2":      KP/"war_eval_v2"/"benchmark.json",
    "w0_eval_s1":       KP/"w0_eval_s1"/"benchmark.json",
    "phase1_v5":        KP/"phase1_v5"/"benchmark.json",
    "gate_eval_v2":     KP/"gate_eval_v2"/"benchmark.json",
    "sentinel_eval_v2": KP/"sentinel_eval_v2"/"benchmark.json",
    "a22_v2_1":         KP/"a22_v2_1"/"benchmark.json",
}
WAR_TRIPLET = ["war_eval_v1", "war_eval_v2", "war_eval_v3"]

# ----------------------------------------------------------------- validation
report = {"assumptions": {}, "validation": {}}

# war triplet + null10 reproduction
val = {}
for name in WAR_TRIPLET:
    _, runs = load_bench(CONFIG_RUNS[name])
    e, n = recompute_check(runs)
    val[name] = {"max_err": e, "n_games": n}
# null10
null_val = {}
for s in range(101, 111):
    d = json.loads((ROOT/"runs"/"null10"/f"seed{s}"/"benchmark.json").read_text())
    e, n = recompute_check(d["game_runs"])
    # null: base_actions null -> rhae returns 0; stored final_score also 0
    null_val[f"seed{s}"] = {"max_err": e, "n_games": n,
                            "n_nonzero": sum(1 for r in d["game_runs"] if r["final_score"]>0)}
report["validation"]["war_triplet"] = val
report["validation"]["null10"] = null_val
report["validation"]["null10_all_zero"] = all(v["n_nonzero"]==0 for v in null_val.values())

# ----------------------------------------------------------------- build matrix
# gid list from war_eval_v1
_, r1 = load_bench(CONFIG_RUNS["war_eval_v1"])
GIDS = sorted(g for g in per_game(r1))
assert len(GIDS) == 25
# matrix: config_scores[name] = {gid: score}
config_scores = {}
for name, p in CONFIG_RUNS.items():
    _, runs = load_bench(p)
    pg = per_game(runs)
    assert sorted(pg) == GIDS, f"{name} game set mismatch"
    config_scores[name] = pg

war_M = [[config_scores[n][g] for g in GIDS] for n in WAR_TRIPLET]      # 3 x 25
all_names = list(CONFIG_RUNS.keys())
all_M = [[config_scores[n][g] for g in GIDS] for n in all_names]        # 13 x 25

report["assumptions"]["n_games_local_run"] = 25
report["assumptions"]["war_triplet_run_means"] = [st.mean(r) for r in war_M]
report["assumptions"]["config_run_means"] = {n: st.mean(config_scores[n].values() if False else [config_scores[n][g] for g in GIDS]) for n in all_names}

# ----------------------------------------------------------------- helpers
def pearson(xs, ys):
    n = len(xs)
    if n < 2: return None
    mx, my = st.mean(xs), st.mean(ys)
    sx = math.sqrt(sum((x-mx)**2 for x in xs)); sy = math.sqrt(sum((y-my)**2 for y in ys))
    if sx == 0 or sy == 0: return None
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/(sx*sy)

def subset_mean(row, idx):
    return sum(row[i] for i in idx)/len(idx)

# ----------------------------------------------------------------- (2) direct subset-split
def subset_split_rho(M, p_size, q_size, n_resamples=4000):
    """For each random disjoint partition (sizes p,q) of the 25 games,
    compute per-run (publicMean, privateMean) and Pearson across runs.
    Return mean rho over resamples that yield a defined correlation."""
    n_games = len(M[0]); idx_all = list(range(n_games))
    rhos = []
    for _ in range(n_resamples):
        perm = idx_all[:]; RNG.shuffle(perm)
        P = perm[:p_size]; Q = perm[p_size:p_size+q_size]
        pub = [subset_mean(row, P) for row in M]
        pri = [subset_mean(row, Q) for row in M]
        r = pearson(pub, pri)
        if r is not None:
            rhos.append(r)
    if not rhos:
        return None
    rhos.sort()
    return {"mean": st.mean(rhos), "median": rhos[len(rhos)//2],
            "p05": rhos[int(0.05*len(rhos))], "p95": rhos[int(0.95*len(rhos))],
            "n_valid": len(rhos)}

# split sizes that fit in 25 games
SPLITS = [(12,13),(10,10),(5,20),(8,8),(5,5)]
report["direct_subset_split"] = {"war_triplet_n3": {}, "all_configs_n13": {}}
for (p,q) in SPLITS:
    report["direct_subset_split"]["war_triplet_n3"][f"{p}v{q}"] = subset_split_rho(war_M, p, q)
    report["direct_subset_split"]["all_configs_n13"][f"{p}v{q}"] = subset_split_rho(all_M, p, q)

# single-run public<->heldout within-seed correlation:
# within ONE run, split games, corr of per-game scores between the two halves
# (games are the replicate). This is a reliability/split-half proxy.
def within_run_splithalf(row, n_resamples=2000):
    n = len(row); idx = list(range(n)); rs=[]
    half = n//2
    for _ in range(n_resamples):
        perm = idx[:]; RNG.shuffle(perm)
        A = perm[:half]; B = perm[half:2*half]
        # pair games arbitrarily and correlate paired per-game scores? Not meaningful.
        # Instead: does subset-A MEAN track subset-B MEAN? one number per split -> no corr.
        pass
    return None  # not used; see note in report

# ---- within-single-run public<->heldout correlation (across splits, per seed)
def within_run_split_corr(row, p_size, q_size, n_resamples=4000):
    n=len(row); idx=list(range(n)); pubs=[]; helds=[]
    for _ in range(n_resamples):
        perm=idx[:]; RNG.shuffle(perm)
        P=perm[:p_size]; Q=perm[p_size:p_size+q_size]
        pubs.append(subset_mean(row,P)); helds.append(subset_mean(row,Q))
    return pearson(pubs,helds)
report["within_seed_split_corr_12v13"] = {
    WAR_TRIPLET[i]: within_run_split_corr(war_M[i],12,13) for i in range(3)}
report["within_seed_split_corr_5v5"] = {
    WAR_TRIPLET[i]: within_run_split_corr(war_M[i],5,5) for i in range(3)}

# ----------------------------------------------------------------- (variance components)
def variance_components(M):
    """Two-way random(run)+fixed(game) decomposition.
    Returns sigma2_a (draw/run-level), sigma2_e (game x run residual), grand, run_means."""
    I = len(M); J = len(M[0])
    grand = sum(sum(row) for row in M)/(I*J)
    run_means = [sum(row)/J for row in M]
    game_means = [sum(M[i][j] for i in range(I))/I for j in range(J)]
    SS_run = J*sum((rm-grand)**2 for rm in run_means)          # df I-1
    SS_resid = 0.0
    for i in range(I):
        for j in range(J):
            e = M[i][j]-run_means[i]-game_means[j]+grand
            SS_resid += e*e
    df_run = I-1; df_resid = (I-1)*(J-1)
    MS_run = SS_run/df_run if df_run>0 else float('nan')
    MS_resid = SS_resid/df_resid if df_resid>0 else float('nan')
    sig2_e = MS_resid
    sig2_a = (MS_run - MS_resid)/J
    return {"grand": grand, "run_means": run_means,
            "MS_run": MS_run, "MS_resid": MS_resid,
            "sigma2_a": sig2_a, "sigma2_a_clamped": max(0.0, sig2_a),
            "sigma2_e": sig2_e, "sigma_a": math.sqrt(max(0.0,sig2_a)),
            "sigma_e": math.sqrt(sig2_e), "I": I, "J": J}

vc_war = variance_components(war_M)
vc_all = variance_components(all_M)
report["variance_components"] = {"war_triplet_n3": vc_war, "all_configs_n13": vc_all}

def model_rho(sig2_a, sig2_e, p, q):
    va = sig2_a + sig2_e/p; vb = sig2_a + sig2_e/q
    if va<=0 or vb<=0: return None
    return sig2_a/math.sqrt(va*vb)

# model-based rho at real sizes, from war triplet variance components (draw-luck rho)
REAL_SIZES = [(25,25),(25,55),(25,85),(55,55),(50,50),(55,55)]
# unique
size_grid = [(12,13),(25,25),(50,50),(55,55),(25,55),(25,85),(55,110)]
report["model_rho_draw"] = {}
for (p,q) in size_grid:
    report["model_rho_draw"][f"{p}v{q}"] = model_rho(vc_war["sigma2_a_clamped"], vc_war["sigma2_e"], p, q)

# ----------------------------------------------------------------- bootstrap CI (game-resample)
def bootstrap_model_rho(names_list, p, q, n_boot=2000, use_ac=False):
    """Resample GAMES with replacement, recompute variance components + model rho.
    names_list: list of config names (rows)."""
    M = [[config_scores[n][g] for g in GIDS] for n in names_list]
    J = len(GIDS); vals=[]
    for _ in range(n_boot):
        cols = [RNG.randrange(J) for _ in range(J)]
        Mb = [[row[c] for c in cols] for row in M]
        vc = variance_components(Mb)
        r = model_rho(vc["sigma2_a_clamped"], vc["sigma2_e"], p, q)
        if r is not None:
            vals.append(r)
    if not vals: return None
    vals.sort()
    return {"point": model_rho(variance_components(M)["sigma2_a_clamped"],
                               variance_components(M)["sigma2_e"], p, q),
            "median": vals[len(vals)//2], "ci05": vals[int(0.05*len(vals))],
            "ci95": vals[int(0.95*len(vals))], "mean": st.mean(vals), "n": len(vals)}

report["bootstrap_ci_draw_rho"] = {}
for (p,q) in [(55,55),(25,55),(50,50),(25,25)]:
    report["bootstrap_ci_draw_rho"][f"{p}v{q}"] = bootstrap_model_rho(WAR_TRIPLET, p, q)

# ----------------------------------------------------------------- capability rho (all 13 configs)
# direct: across the 13 configs, corr(publicMean, privateMean) averaged over partitions
report["capability_rho"] = {
    "note": "across-config (capability spread 0.85->2.17), NOT draw luck",
    "direct_subset_split_12v13": report["direct_subset_split"]["all_configs_n13"]["12v13"],
    "model_rho_from_all13_vc": {f"{p}v{q}": model_rho(vc_all["sigma2_a_clamped"], vc_all["sigma2_e"], p, q)
                                for (p,q) in [(55,55),(25,55),(25,25)]},
}

# ----------------------------------------------------------------- (3) LB reconciliation
# observed public draw SD anchors
LB_DRAWS = [0.82, 0.89, 0.93, 1.02, 0.95]          # variance_reconcile frozen-fork pool
lb_sd = st.stdev(LB_DRAWS); lb_mean = st.mean(LB_DRAWS)
# model-implied public SD at p=55 from war-triplet vc:
def implied_sd(vc, p):
    return math.sqrt(max(0.0, vc["sigma2_a_clamped"]) + vc["sigma2_e"]/p)
report["lb_reconciliation"] = {
    "lb_draw_pool": LB_DRAWS, "lb_mean": lb_mean, "lb_sd_n5": lb_sd,
    "doc_sigma_hat_claim": 0.15,
    "war_triplet_25game_run_mean_sd_observed": st.stdev([st.mean(r) for r in war_M]),
    "model_implied_public_sd_p55_from_war_vc": implied_sd(vc_war, 55),
    "model_implied_public_sd_p25_from_war_vc": implied_sd(vc_war, 25),
    "sigma_a_war": vc_war["sigma_a"],
    "sigma_e_war": vc_war["sigma_e"],
}

# ----------------------------------------------------------------- (4) decision: E[selected private]
# Under bivariate normal (public,private) with corr rho, select the max-public of k draws;
# expected private of the selected draw, in units of private SD above the mean:
#   E[private_selected] - mu = rho * sigma_priv * E[Z_(k:k)]
# E[max of k std normals] approx:
def emax_normal(k):
    # Blom approximation for expected max order statistic
    if k==1: return 0.0
    return (k - math.pi/8)/(k - math.pi/4 + 1) * math.sqrt(2*math.log(k)) if k>1 else 0.0
# better: use known small-k values
EMAX = {1:0.0,2:0.5642,3:0.8463,4:1.0294,5:1.1630,6:1.2672,8:1.4236,10:1.5388,15:1.7359,20:1.8675,30:2.0428}
def emax(k):
    return EMAX.get(k, math.sqrt(2*math.log(max(k,2))))

# fraction of available private upside captured by public-max selection = rho
# (the selected-private gain over mean is rho * sigma_priv * emax(k))
sens = {}
for rho in [0.0,0.1,0.2,0.3,0.5,0.6,0.7,0.9,1.0]:
    row={}
    for k in [1,3,6,10,20]:
        # gain over mean, in units of sigma_priv
        row[f"k={k}"] = rho*emax(k)
    sens[f"rho={rho}"]=row
report["decision_Eselected_private_over_mean_in_sigma_priv"] = sens

# threshold logic
report["decision"] = {}

# write raw json
(OUT/"rho_raw.json").write_text(json.dumps(report, indent=2, default=lambda o: list(o) if hasattr(o,'__iter__') else str(o)))
print("WROTE", OUT/"rho_raw.json")

# ---- print key numbers
print("\n=== VALIDATION ===")
for n,v in val.items(): print(f"  {n}: max_err={v['max_err']:.2e} n={v['n_games']}")
print(f"  null10 all-zero: {report['validation']['null10_all_zero']}  (max_err all seeds "
      f"{max(v['max_err'] for v in null_val.values()):.2e})")
print("\n=== WAR TRIPLET (same config, 3 draws) variance components ===")
print(f"  run means (25g): {[round(x,3) for x in vc_war['run_means']]}")
print(f"  sigma2_a(draw)={vc_war['sigma2_a']:.4f} (clamped {vc_war['sigma2_a_clamped']:.4f})  sigma_a={vc_war['sigma_a']:.3f}")
print(f"  sigma2_e(game)={vc_war['sigma2_e']:.4f}  sigma_e={vc_war['sigma_e']:.3f}")
print("\n=== MODEL rho_DRAW (from war triplet) ===")
for k,v in report["model_rho_draw"].items(): print(f"  {k}: {None if v is None else round(v,3)}")
print("\n=== BOOTSTRAP CI rho_draw ===")
for k,v in report["bootstrap_ci_draw_rho"].items():
    if v: print(f"  {k}: point={v['point']:.3f} median={v['median']:.3f} CI90=[{v['ci05']:.3f},{v['ci95']:.3f}]")
print("\n=== DIRECT SUBSET-SPLIT rho (empirical) ===")
print("  war triplet n=3:")
for k,v in report["direct_subset_split"]["war_triplet_n3"].items():
    if v: print(f"    {k}: mean={v['mean']:.3f} median={v['median']:.3f} [{v['p05']:.3f},{v['p95']:.3f}]")
print("  all 13 configs (capability):")
for k,v in report["direct_subset_split"]["all_configs_n13"].items():
    if v: print(f"    {k}: mean={v['mean']:.3f} median={v['median']:.3f} [{v['p05']:.3f},{v['p95']:.3f}]")
print("\n=== CAPABILITY rho (all 13 configs, model) ===")
for k,v in report["capability_rho"]["model_rho_from_all13_vc"].items(): print(f"  {k}: {None if v is None else round(v,3)}")
print("\n=== LB RECONCILIATION ===")
lr=report["lb_reconciliation"]
for k,v in lr.items():
    if isinstance(v,float): print(f"  {k}: {v:.4f}")
