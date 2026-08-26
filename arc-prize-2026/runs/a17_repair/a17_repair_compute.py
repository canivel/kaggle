"""A17'' (R16 repair) — per-seed Sigma table, throttled-null walks, false-NO-GO bootstrap.

Inputs: runs/kernel_pulls/{war_eval_v1,war_eval_v2,war_eval_v3,w0_eval_s1}/benchmark.json
Outputs: runs/a17_repair/per_seed_table.json, runs/a17_repair/false_nogo_bootstrap.json

Definitions (frozen, mirror a17_72b_screen_scope_v2.md):
- lc(r,g)              = levels_completed for run r, game g (as recorded in benchmark.json).
- N(r,g)               = sum(actions_per_level) for run r, game g (includes the partial level).
- throttled_lc(r,g,rho)= number of COMPLETED levels (index < lc) whose cumulative action end
                         <= floor(N(r,g)/rho), walking r's own actions_per_level.
- null_adj(g,rho)      = throttled_lc(W0,g,rho)  [the sealed null, W0-anchored].
- Gate branch 2 bar    = Sigma_g null_adj(g,rho) + MARGIN (MARGIN=1).
- 72B statistic        = per-game MAX over k 72B seeds, then Sigma over the 4 games (both prongs).

Bootstrap models (methodology R1 spec):
(i)  null 72B==27B      : pseudo-72B seed = one of the 4 certified rows.
(ii) true +1/game shift : pseudo-72B seed = row + 1 level per game (capped at number_of_levels).
Regimes:
- parity regime  (branch 1 lens): rows = full-budget lc            -> capability prong Sigma>=8.
- throttled regime (branch 2 lens, expected world rho in [2.4,3.1]): rows = throttled_lc(rho)
                                                                   -> Sigma >= Sigma null_adj + 1.
Draw schemes: row-wise (preserves within-seed correlation) and independent-per-game
(conservative). Exact enumeration over the 4-row support for k=1,2; MC (100k) as bootstrap check.
"""
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[1].name == "runs") else Path(".")
ROOT = Path("F:/kaggle/arc-prize-2026")
RUNS = ["war_eval_v1", "war_eval_v2", "war_eval_v3", "w0_eval_s1"]
GAMES = ["ft09", "sb26", "lp85", "vc33"]
RHOS = [round(1.5 + 0.1 * i, 1) for i in range(21)]  # 1.5 .. 3.5
ANCHORS = [2.5, 3.0]
MARGIN = 1

def load():
    data = {}
    for r in RUNS:
        b = json.load(open(ROOT / "runs/kernel_pulls" / r / "benchmark.json"))
        row = {}
        for g in b["game_runs"]:
            pre = g["game_id"].split("-")[0]
            if pre in GAMES:
                apl = [a for a in g["actions_per_level"] if a > 0] or [0]
                row[pre] = {
                    "game_id": g["game_id"],
                    "lc": g["levels_completed"],
                    "actions_per_level": g["actions_per_level"],
                    "N": sum(g["actions_per_level"]),
                    "number_of_levels": g["number_of_levels"],
                }
        data[r] = row
    return data

def throttled_lc(row_g, rho):
    n72 = int(row_g["N"] / rho)  # floor
    cum, done = 0, 0
    for i, a in enumerate(row_g["actions_per_level"]):
        if i >= row_g["lc"]:
            break
        cum += a
        if cum <= n72:
            done += 1
        else:
            break
    return done

def main():
    data = load()
    # --- per-seed table -----------------------------------------------------
    full_rows = {r: {g: data[r][g]["lc"] for g in GAMES} for r in RUNS}
    per_seed_sigma = {r: sum(full_rows[r].values()) for r in RUNS}
    per_game_max = {g: max(full_rows[r][g] for r in RUNS) for g in GAMES}
    sigma_max = sum(per_game_max.values())
    shortfall = {r: sigma_max - per_seed_sigma[r] for r in RUNS}

    thr = {}
    for rho in RHOS:
        thr[str(rho)] = {r: {g: throttled_lc(data[r][g], rho) for g in GAMES} for r in RUNS}
    null_adj = {str(rho): {g: thr[str(rho)]["w0_eval_s1"][g] for g in GAMES} for rho in RHOS}

    table = {
        "runs": RUNS,
        "games": GAMES,
        "game_ids": {g: data["w0_eval_s1"][g]["game_id"] for g in GAMES},
        "full_budget_lc": full_rows,
        "per_seed_sigma_full_budget": per_seed_sigma,
        "per_game_max_27B": per_game_max,
        "sigma_27B_max": sigma_max,
        "per_seed_shortfall_vs_max": shortfall,
        "N_27B": {r: {g: data[r][g]["N"] for g in GAMES} for r in RUNS},
        "actions_per_level": {r: {g: data[r][g]["actions_per_level"] for g in GAMES} for r in RUNS},
        "throttled_lc_by_rho": thr,
        "null_adj_by_rho_W0_anchored": null_adj,
        "sigma_null_adj": {str(rho): sum(null_adj[str(rho)].values()) for rho in RHOS},
    }
    out = ROOT / "runs/a17_repair"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(table, open(out / "per_seed_table.json", "w"), indent=1)

    # --- gate evaluators ----------------------------------------------------
    def sigma_of_maxrows(rows_drawn, games=GAMES):
        return sum(max(row[g] for row in rows_drawn) for g in games)

    def enumerate_p(rows, k, bar, shift=0, cap=None):
        """Exact P(Sigma(per-game MAX over k row-draws w/ repl) >= bar), row-wise."""
        vals = []
        for combo in itertools.product(rows, repeat=k):
            drawn = [{g: min(row[g] + shift, cap[g]) if cap else row[g] + shift for g in GAMES}
                     for row in combo]
            vals.append(sigma_of_maxrows(drawn) >= bar)
        return sum(vals) / len(vals)

    def enumerate_p_indep(rows, k, bar, shift=0, cap=None):
        """Independent-per-game draws (conservative): each game's k draws iid from its 4 values."""
        per_game_vals = {g: [min(r[g] + shift, cap[g]) if cap else r[g] + shift for r in rows]
                         for g in GAMES}
        # enumerate per-game max distribution over k iid draws from 4 values
        dists = {}
        for g in GAMES:
            counts = {}
            for combo in itertools.product(per_game_vals[g], repeat=k):
                m = max(combo)
                counts[m] = counts.get(m, 0) + 1
            tot = sum(counts.values())
            dists[g] = {v: c / tot for v, c in counts.items()}
        # convolve
        acc = {0: 1.0}
        for g in GAMES:
            nxt = {}
            for s, p in acc.items():
                for v, q in dists[g].items():
                    nxt[s + v] = nxt.get(s + v, 0.0) + p * q
            acc = nxt
        return sum(p for s, p in acc.items() if s >= bar)

    caps = {g: data["w0_eval_s1"][g]["number_of_levels"] for g in GAMES}
    results = {"margin": MARGIN, "capability_bar": sigma_max + 2, "schemes": {}}

    # Parity regime (capability prong, Sigma >= 8), full-budget rows
    rows_full = [full_rows[r] for r in RUNS]
    par = {}
    for k in (1, 2):
        par[f"k={k}"] = {
            "P_GO_null_rowwise": enumerate_p(rows_full, k, sigma_max + 2, 0, caps),
            "P_GO_null_indep": enumerate_p_indep(rows_full, k, sigma_max + 2, 0, caps),
            "P_NOGO_shift1_rowwise": 1 - enumerate_p(rows_full, k, sigma_max + 2, 1, caps),
            "P_NOGO_shift1_indep": 1 - enumerate_p_indep(rows_full, k, sigma_max + 2, 1, caps),
        }
    results["schemes"]["parity_regime_capability_prong_bar8"] = par

    # Throttled regime (branch 2), rho anchors
    for rho in ANCHORS:
        rows_thr = [thr[str(rho)][r] for r in RUNS]
        bar = sum(null_adj[str(rho)].values()) + MARGIN
        d = {"bar_sigma_nulladj_plus_margin": bar}
        for k in (1, 2):
            d[f"k={k}"] = {
                "P_falseGO_null_rowwise": enumerate_p(rows_thr, k, bar, 0, caps),
                "P_falseGO_null_indep": enumerate_p_indep(rows_thr, k, bar, 0, caps),
                "P_falseNOGO_shift1_rowwise": 1 - enumerate_p(rows_thr, k, bar, 1, caps),
                "P_falseNOGO_shift1_indep": 1 - enumerate_p_indep(rows_thr, k, bar, 1, caps),
            }
        # weaker alternative: +1 level on exactly ONE game (each game equally likely)
        weak = {}
        for k in (1, 2):
            ps = []
            for gshift in GAMES:
                cap1 = caps
                # rows with +1 only on gshift
                rows_w = [{g: min(row[g] + (1 if g == gshift else 0), cap1[g]) for g in GAMES}
                          for row in rows_thr]
                ps.append(enumerate_p(rows_w, k, bar, 0, None))
            weak[f"k={k}"] = {"P_GO_by_shifted_game": dict(zip(GAMES, ps)),
                              "P_GO_avg": sum(ps) / len(ps)}
        d["weak_alt_plus1_single_game"] = weak
        results["schemes"][f"throttled_regime_rho={rho}"] = d

    # MC bootstrap cross-check (row-wise, 100k) at rho=2.5, shift=+1, k=1
    rng = random.Random(20260722)
    rows_thr = [thr["2.5"][r] for r in RUNS]
    bar = sum(null_adj["2.5"].values()) + MARGIN
    B = 100_000
    hits = 0
    for _ in range(B):
        row = rng.choice(rows_thr)
        s = sum(min(row[g] + 1, caps[g]) for g in GAMES)
        hits += (s >= bar)
    results["mc_check_rho2.5_shift1_k1_P_GO"] = hits / B
    results["mc_B"] = B

    json.dump(results, open(out / "false_nogo_bootstrap.json", "w"), indent=1)

    # console summary
    print("per-seed Sigma (full budget):", per_seed_sigma)
    print("per-game 27B MAX:", per_game_max, "SigmaMAX =", sigma_max, "shortfalls:", shortfall)
    for rho in ANCHORS:
        print(f"rho={rho}: null_adj per game =", null_adj[str(rho)],
              "Sigma =", sum(null_adj[str(rho)].values()),
              "| per-seed throttled rows:", {r: thr[str(rho)][r] for r in RUNS})
    print(json.dumps(results, indent=1))

if __name__ == "__main__":
    main()
