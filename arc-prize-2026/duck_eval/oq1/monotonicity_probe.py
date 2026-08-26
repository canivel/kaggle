#!/usr/bin/env python
"""
OQ-1 MONOTONICITY PROBE  --  panel round 26, directive #3.

Runs EXACTLY the design sealed in
    learnings/war_room/oq1_monotonicity_prereg_2026-08-16.md
    sha256 at seal: b989f2a2b836d5d65694cc23d0c05486c9a8d7c5c21b38c968746d98f983ad16

Question (prereg SS1): is the agent's action-selection criterion monotone in what scores?
  x_t = n_g(sigma(a_t))   revealed selection weight of the symbol executed at step t
  y_t = c_t               discounted forward credit toward a levels_completed increment
  tau_g = Kendall tau_b(x, y) ;  T = sum_g N_g tau_g / sum_g N_g
Nulls (conjunctive): N1 uniform within-game shuffle, N2 circular rotation, N3 score-anchor placebo.

FREE / CPU-only.  Reads only; writes only to runs/oq1_monotonicity/.

Usage:
    PYTHONPATH=duck_eval/taaf_bundle/src/tufa-arc-agi-framework/src \
        uv run --with imageio --with scipy python duck_eval/oq1/monotonicity_probe.py --selftest
    ... python duck_eval/oq1/monotonicity_probe.py --run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
from scipy.stats import kendalltau

# ---------------------------------------------------------------- sealed constants
GAMMA = 0.98                 # prereg SS3.2
COARSE = 8                   # prereg SS3.1  ACTION6@(x//8, y//8)
B_PERM = 10_000              # prereg SS4
MASTER_SEED = 20260816       # prereg SS4
ALPHA = 0.05                 # prereg SS5
BAND_Z = 1.0                 # prereg SS5 indistinguishability band
MIN_ALPHABET = 3             # prereg SS2 E2
MIN_STEPS = 30               # prereg SS2 E3
GAMMA_ROBUST = (0.95, 0.99)  # prereg SS3.2 descriptive
POWER_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)  # prereg SS8.4
POWER_R = 2000

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUN_DIR = os.path.join(REPO, "runs", "a22_v2_seed1")
OUT_DIR = os.path.join(REPO, "runs", "oq1_monotonicity")
PREREG = os.path.join(REPO, "learnings", "war_room", "oq1_monotonicity_prereg_2026-08-16.md")


# ---------------------------------------------------------------- core primitives
def symbol(action) -> str:
    """prereg SS3.1 -- PRIMARY coarsened action symbol."""
    name = action.id.name
    if name != "ACTION6":
        return name
    d = getattr(action, "data", None) or {}
    x, y = d.get("x"), d.get("y")
    if x is None or y is None:          # ACTION6 without coords (should not occur; be explicit)
        return "ACTION6@NA"
    return f"ACTION6@{int(x) // COARSE},{int(y) // COARSE}"


def symbol_idonly(action) -> str:
    """prereg SS3.1 -- ROBUSTNESS alphabet (descriptive only)."""
    return action.id.name


def credit_vector(n_steps: int, scoring_steps, gamma: float = GAMMA) -> np.ndarray:
    """prereg SS3.2: c_t = max{gamma^(e-t) : e in S, e >= t}, else 0. Steps are 1-based -> index 0..n-1."""
    c = np.zeros(n_steps, dtype=np.float64)
    if not scoring_steps:
        return c
    s = sorted(scoring_steps)
    # walk backwards: nearest future scoring step
    nxt = None
    j = len(s) - 1
    for t in range(n_steps, 0, -1):
        while j >= 0 and s[j] >= t:
            nxt = s[j]
            j -= 1
        # nxt is the smallest scoring step >= t seen so far
        if nxt is not None and nxt >= t:
            c[t - 1] = gamma ** (nxt - t)
    return c


def tau_b(x: np.ndarray, y: np.ndarray) -> float:
    t = kendalltau(x, y, variant="b")
    v = t.statistic if hasattr(t, "statistic") else t[0]
    return 0.0 if not np.isfinite(v) else float(v)


def weights_from_labels(labels: np.ndarray, n_sym: int) -> np.ndarray:
    """x_t = n_g(symbol at t): map integer symbol labels -> their occurrence count."""
    counts = np.bincount(labels, minlength=n_sym)
    return counts[labels].astype(np.float64)


# ---------------------------------------------------------------- data loading
def load_games():
    sys.path.insert(0, os.path.join(REPO, "duck_eval", "taaf_bundle", "src",
                                    "tufa-arc-agi-framework", "src"))
    with open(os.path.join(RUN_DIR, "intermediate_states.pkl"), "rb") as f:
        states = pickle.load(f)
    with open(os.path.join(RUN_DIR, "benchmark.json")) as f:
        bench = json.load(f)
    runs = bench["game_runs"]
    assert len(states) == len(runs) == 25, "expected 25 games"

    games = []
    for i, (G, r) in enumerate(zip(states, runs)):
        gid = r["game_id"]
        # cross-check pickle order against benchmark order (prereg SS0 item 2)
        assert len(G) == len(r["history"]) + 1, f"{gid}: state/history length mismatch"
        syms, syms_id, lcs = [], [], []
        for t in range(1, len(G)):
            pa = G[t].previous_action
            assert pa is not None, f"{gid}: null action at step {t}"
            syms.append(symbol(pa))
            syms_id.append(symbol_idonly(pa))
            lcs.append(int(G[t].levels_completed or 0))
        prev = int(G[0].levels_completed or 0)
        scoring = []
        for t, v in enumerate(lcs, start=1):
            if v > prev:
                scoring.append(t)
            prev = v
        assert lcs[-1] == r["levels_completed"], f"{gid}: final lc mismatch"
        games.append(dict(idx=i, game_id=gid, n=len(syms), syms=syms,
                          syms_id=syms_id, scoring=scoring, levels=lcs[-1]))
    return games


def split_games(games):
    """prereg SS6: game_ids sorted ascending; HELD-OUT = even ranks, DEV = odd ranks."""
    order = sorted(games, key=lambda g: g["game_id"])
    held = [g for k, g in enumerate(order) if k % 2 == 0]
    dev = [g for k, g in enumerate(order) if k % 2 == 1]
    return dev, held


def eligible(games, sym_key="syms"):
    """prereg SS2: E1 >=1 scoring event, E2 |A|>=3, E3 N>=30."""
    out = []
    for g in games:
        if not g["scoring"]:
            continue
        if len(set(g[sym_key])) < MIN_ALPHABET:
            continue
        if g["n"] < MIN_STEPS:
            continue
        out.append(g)
    return out


def prepare(g, gamma=GAMMA, sym_key="syms"):
    syms = g[sym_key]
    uniq = sorted(set(syms))
    lut = {s: k for k, s in enumerate(uniq)}
    labels = np.array([lut[s] for s in syms], dtype=np.int64)
    c = credit_vector(g["n"], g["scoring"], gamma)
    return labels, len(uniq), c


# ---------------------------------------------------------------- statistic + nulls
def T_stat(prepped) -> tuple[float, dict]:
    num = 0.0
    den = 0
    per = {}
    for gid, (labels, n_sym, c) in prepped.items():
        x = weights_from_labels(labels, n_sym)
        t = tau_b(x, c)
        per[gid] = t
        num += len(labels) * t
        den += len(labels)
    return (num / den if den else float("nan")), per


def null_distribution(prepped, kind: str, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keys = list(prepped)
    Ns = {k: len(prepped[k][0]) for k in keys}
    den = sum(Ns.values())
    out = np.empty(B, dtype=np.float64)
    for b in range(B):
        num = 0.0
        for k in keys:
            labels, n_sym, c = prepped[k]
            n = Ns[k]
            if kind == "N1":                     # uniform within-game shuffle
                lab = rng.permutation(labels)
                x = weights_from_labels(lab, n_sym)
                yy = c
            elif kind == "N2":                   # circular rotation
                off = int(rng.integers(0, n))
                lab = np.roll(labels, off)
                x = weights_from_labels(lab, n_sym)
                yy = c
            elif kind == "N3":                   # score-anchor placebo
                k_s = int((c == 1.0).sum()) or 1
                anchors = rng.choice(n, size=k_s, replace=False) + 1
                yy = credit_vector(n, list(anchors))
                x = weights_from_labels(labels, n_sym)
            else:
                raise ValueError(kind)
            num += n * tau_b(x, yy)
        out[b] = num / den
    return out


def perm_p(T_obs: float, null: np.ndarray) -> tuple[float, float]:
    mu, sd = float(null.mean()), float(null.std(ddof=1))
    dev = abs(T_obs - mu)
    p = (1.0 + int((np.abs(null - mu) >= dev - 1e-15).sum())) / (len(null) + 1.0)
    z = (T_obs - mu) / sd if sd > 0 else 0.0
    return p, z


def verdict(T_obs: float, res: dict) -> tuple[str, str]:
    """prereg SS5 decision table -- conjunctive over N1,N2,N3."""
    ps = [res[k]["p"] for k in ("N1", "N2", "N3")]
    zs = [res[k]["z"] for k in ("N1", "N2", "N3")]
    sig = [p <= ALPHA for p in ps]
    if all(sig) and all(z > 0 for z in zs):
        return "MONOTONE", "p<=0.05 and T>mean(null) under all three nulls"
    if all(sig) and all(z < 0 for z in zs):
        return "ANTI-MONOTONE", "p<=0.05 and T<mean(null) under all three nulls"
    if all(abs(z) < BAND_Z for z in zs):
        return "SCORE-BLIND", f"|z| < {BAND_Z} under all three nulls"
    if any(sig) and not all(sig):
        bad = [k for k in ("N1", "N2", "N3") if res[k]["p"] > ALPHA]
        return "INDETERMINATE-CONFOUNDED", f"nulls disagree; non-significant under {','.join(bad)}"
    return "INDETERMINATE", "1.0 <= |z| < 1.96 on at least one null; no null reaches alpha"


# ---------------------------------------------------------------- power (prereg SS8.4)
def power_curve(prepped, crit: dict, lambdas=POWER_LAMBDAS, R=POWER_R, seed=MASTER_SEED + 99):
    """Credit-tilt simulation. P(s) ∝ n_g(s)·exp(lambda · cbar_g(s)); credit vectors held fixed.

    Approximation, declared: critical values are taken from the OBSERVED-marginals nulls; tilting
    perturbs marginals slightly, so these power figures are approximate (prereg SS8.4).
    """
    rng = np.random.default_rng(seed)
    keys = list(prepped)
    den = sum(len(prepped[k][0]) for k in keys)
    base = {}
    for k in keys:
        labels, n_sym, c = prepped[k]
        counts = np.bincount(labels, minlength=n_sym).astype(np.float64)
        cbar = np.array([c[labels == s].mean() if (labels == s).any() else 0.0
                         for s in range(n_sym)])
        base[k] = (counts, cbar, c, n_sym)

    rows = []
    for lam in lambdas:
        hits = 0
        Ts = np.empty(R)
        for r in range(R):
            num = 0.0
            for k in keys:
                counts, cbar, c, n_sym = base[k]
                w = counts * np.exp(lam * cbar)
                w = w / w.sum()
                n = len(c)
                lab = rng.choice(n_sym, size=n, replace=True, p=w)
                x = weights_from_labels(lab, n_sym)
                num += n * tau_b(x, c)
            Ts[r] = num / den
        # conjunctive rule: must exceed the upper critical value of ALL three nulls
        hits = int(np.all([Ts > crit[kk] for kk in ("N1", "N2", "N3")], axis=0).sum())
        rows.append(dict(lam=float(lam), power=hits / R, mean_T=float(Ts.mean()),
                         sd_T=float(Ts.std(ddof=1))))
    return rows


# ---------------------------------------------------------------- selftest
def selftest() -> int:
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}")
        ok = ok and bool(cond)

    print("SELFTEST 1 -- credit_vector")
    c = credit_vector(6, [4], gamma=0.5)
    exp = np.array([0.5 ** 3, 0.5 ** 2, 0.5 ** 1, 1.0, 0.0, 0.0])
    check("forward-only discounted credit", np.allclose(c, exp), f"{c}")
    c2 = credit_vector(5, [], gamma=0.9)
    check("no scoring event -> all zero", np.allclose(c2, 0))
    c3 = credit_vector(6, [2, 5], gamma=0.5)
    check("two events, nearest-future wins",
          np.allclose(c3, [0.5, 1.0, 0.25, 0.5, 1.0, 0.0]), f"{c3}")

    print("SELFTEST 2 -- tau_b vs scipy on random ties-heavy input")
    rng = np.random.default_rng(7)
    a = rng.integers(0, 4, 200).astype(float)
    b = rng.random(200).round(2)
    check("tau_b matches scipy", abs(tau_b(a, b) - kendalltau(a, b, variant="b").statistic) < 1e-12)

    print("SELFTEST 3 -- weights_from_labels")
    lab = np.array([0, 0, 0, 1, 2, 2])
    check("x_t = count of its own symbol",
          np.array_equal(weights_from_labels(lab, 3), [3, 3, 3, 1, 2, 2]))

    print("SELFTEST 4 -- positive control (criterion IS monotone)")
    # symbol A used 60x, densely packed just before the win at step 70; symbol B used 20x, far away.
    n = 100
    syms = ["B"] * 20 + ["A"] * 60 + ["B"] * 20
    lut = {"A": 0, "B": 1}
    labels = np.array([lut[s] for s in syms])
    c = credit_vector(n, [70])
    x = weights_from_labels(labels, 2)
    t_pos = tau_b(x, c)
    check("high-mass symbol sits near the score -> tau > 0", t_pos > 0.2, f"tau={t_pos:.3f}")

    print("SELFTEST 5 -- negative control (criterion is ANTI-monotone)")
    syms = ["A"] * 60 + ["B"] * 20 + ["A"] * 20   # B (rare) sits at the win
    labels = np.array([lut[s] for s in syms])
    c = credit_vector(n, [70])
    t_neg = tau_b(weights_from_labels(labels, 2), c)
    check("rare symbol sits near the score -> tau < 0", t_neg < -0.05, f"tau={t_neg:.3f}")

    print("SELFTEST 6 -- null calibration (random criterion, all three nulls)")
    rng = np.random.default_rng(11)
    prepped = {}
    for g in range(3):
        n = 150
        labels = rng.integers(0, 5, n)
        c = credit_vector(n, sorted(rng.choice(n, 2, replace=False) + 1))
        prepped[f"sim{g}"] = (labels, 5, c)
    T_obs, _ = T_stat(prepped)
    for kind in ("N1", "N2", "N3"):
        nd = null_distribution(prepped, kind, 400, MASTER_SEED)
        p, z = perm_p(T_obs, nd)
        check(f"{kind}: random criterion is not significant", p > ALPHA, f"p={p:.3f} z={z:+.2f}")

    print("SELFTEST 7 -- split rule reproduces the sealed membership")
    fake = [dict(game_id=f"{c}{i}", scoring=[], n=0, syms=[], syms_id=[], levels=0, idx=i)
            for i, c in enumerate("abcdefghijklmnopqrstuvwxy")]
    dev, held = split_games(fake)
    check("13 held-out / 12 dev", len(held) == 13 and len(dev) == 12)
    check("held-out = even ranks", held[0]["game_id"] == "a0" and held[1]["game_id"] == "c2")

    print("SELFTEST 8 -- verdict table")
    mk = lambda p, z: {k: dict(p=p, z=z) for k in ("N1", "N2", "N3")}
    check("MONOTONE", verdict(1.0, mk(0.01, 3.0))[0] == "MONOTONE")
    check("ANTI-MONOTONE", verdict(-1.0, mk(0.01, -3.0))[0] == "ANTI-MONOTONE")
    check("SCORE-BLIND", verdict(0.0, mk(0.9, 0.3))[0] == "SCORE-BLIND")
    check("INDETERMINATE", verdict(0.0, mk(0.2, 1.4))[0] == "INDETERMINATE")
    r = mk(0.01, 3.0); r["N3"] = dict(p=0.4, z=0.8)
    check("null disagreement -> CONFOUNDED", verdict(1.0, r)[0] == "INDETERMINATE-CONFOUNDED")

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ---------------------------------------------------------------- run
def analyse(split_name, games, gamma=GAMMA, sym_key="syms", B=B_PERM, do_power=False):
    elig = eligible(games, sym_key)
    prepped = {g["game_id"]: prepare(g, gamma, sym_key) for g in elig}
    T_obs, per = T_stat(prepped)
    res = {}
    crit = {}
    for i, kind in enumerate(("N1", "N2", "N3")):
        nd = null_distribution(prepped, kind, B, MASTER_SEED + 1000 * (i + 1))
        p, z = perm_p(T_obs, nd)
        res[kind] = dict(p=p, z=z, null_mean=float(nd.mean()), null_sd=float(nd.std(ddof=1)),
                         crit_hi=float(np.quantile(nd, 1 - ALPHA / 2)),
                         crit_lo=float(np.quantile(nd, ALPHA / 2)))
        crit[kind] = res[kind]["crit_hi"]
    v, why = verdict(T_obs, res)
    out = dict(split=split_name, gamma=gamma, alphabet=sym_key, B=B,
               n_games_eligible=len(elig),
               games=[g["game_id"] for g in elig],
               n_steps=int(sum(g["n"] for g in elig)),
               n_scoring_events=int(sum(len(g["scoring"]) for g in elig)),
               T_obs=float(T_obs), per_game_tau=per, nulls=res,
               verdict=v, verdict_reason=why)
    if do_power:
        out["power"] = power_curve(prepped, crit)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("-B", type=int, default=B_PERM)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.run:
        ap.error("pass --selftest or --run")

    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    games = load_games()
    dev, held = split_games(games)
    print(f"loaded {len(games)} games; DEV={len(dev)} HELD-OUT={len(held)}")
    print(f"DEV eligible     : {[g['game_id'] for g in eligible(dev)]}")
    print(f"HELD-OUT eligible: {[g['game_id'] for g in eligible(held)]}")

    out = dict(
        prereg=os.path.relpath(PREREG, REPO).replace("\\", "/"),
        prereg_sha256_at_seal=hashlib.sha256(open(PREREG, "rb").read()).hexdigest(),
        source_run="runs/a22_v2_seed1",
        seed_master=MASTER_SEED, gamma=GAMMA, coarse=COARSE, alpha=ALPHA, band_z=BAND_Z,
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    print("\n--- DEV (calibration only, NON-verdict-carrying) ---")
    out["dev"] = analyse("dev", dev, B=a.B)
    print(json.dumps({k: out["dev"][k] for k in
                      ("T_obs", "verdict", "n_games_eligible", "n_steps")}, indent=2))

    print("\n--- HELD-OUT (PRIMARY) ---")
    out["held_out"] = analyse("held_out", held, B=a.B, do_power=True)
    print(json.dumps({k: out["held_out"][k] for k in
                      ("T_obs", "verdict", "verdict_reason", "n_games_eligible",
                       "n_steps", "n_scoring_events")}, indent=2))
    for k in ("N1", "N2", "N3"):
        r = out["held_out"]["nulls"][k]
        print(f"  {k}: T={out['held_out']['T_obs']:+.4f} null_mean={r['null_mean']:+.4f} "
              f"sd={r['null_sd']:.4f} z={r['z']:+.3f} p={r['p']:.4f}")
    print("  power (credit-tilt, conjunctive rule):")
    for row in out["held_out"]["power"]:
        print(f"    lambda={row['lam']:>4}  T={row['mean_T']:+.4f}  power={row['power']:.3f}")

    print("\n--- ROBUSTNESS (descriptive only) ---")
    rob = []
    for gm in GAMMA_ROBUST:
        r = analyse("held_out", held, gamma=gm, B=2000)
        rob.append(dict(kind=f"gamma={gm}", T_obs=r["T_obs"],
                        z={k: r["nulls"][k]["z"] for k in ("N1", "N2", "N3")},
                        p={k: r["nulls"][k]["p"] for k in ("N1", "N2", "N3")},
                        verdict=r["verdict"], n_games=r["n_games_eligible"]))
    r = analyse("held_out", held, sym_key="syms_id", B=2000)
    rob.append(dict(kind="alphabet=id-only", T_obs=r["T_obs"],
                    z={k: r["nulls"][k]["z"] for k in ("N1", "N2", "N3")},
                    p={k: r["nulls"][k]["p"] for k in ("N1", "N2", "N3")},
                    verdict=r["verdict"], n_games=r["n_games_eligible"]))
    out["robustness"] = rob
    for x in rob:
        print(f"  {x['kind']:>20}: T={x['T_obs']:+.4f} n={x['n_games']} "
              f"z={ {k: round(v,2) for k,v in x['z'].items()} } -> {x['verdict']}")

    out["elapsed_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {os.path.join(OUT_DIR, 'results.json')}  ({out['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
