import json, statistics
from collections import defaultdict
from itertools import combinations

d = json.load(open('f:/kaggle/arc-prize-2026/runs/null10/merged_null_benchmark.json'))
print(f"runs: {len(d)}")

# --- 1. Version-suffix consistency per game prefix (ME-NEW-2) ---
suff = defaultdict(set)
for r in d:
    pre, s = r['game_id'].split('-', 1)
    suff[pre].add(s)
print(f"\ngames: {len(suff)}")
unstable = {g: v for g, v in suff.items() if len(v) > 1}
print(f"version-unstable across the 10 seeds: {len(unstable)}: { {g: len(v) for g, v in unstable.items()} }")

# per-game lc vector
bygame = defaultdict(list)
for r in d:
    pre = r['game_id'].split('-', 1)[0]
    bygame[pre].append(r)

flips, dead, grinders = [], [], []
for g, runs in bygame.items():
    lcs = [r['levels_completed'] for r in runs]
    n_pos = sum(1 for x in lcs if x >= 1)
    if n_pos == 0: dead.append(g)
    elif n_pos == len(lcs): grinders.append(g)
    else: flips.append(g)
print(f"dead {len(dead)} {sorted(dead)}")
print(f"always>=1 {len(grinders)} {sorted(grinders)}")
print(f"flip {len(flips)} {sorted(flips)}")

# flip count restricted to version-matched rows: within each flip game, largest same-suffix subset
print("\nflip games version check:")
vm_flips = 0
for g in sorted(flips):
    runs = bygame[g]
    bysuf = defaultdict(list)
    for r in runs:
        bysuf[r['game_id'].split('-',1)[1]].append(r['levels_completed'])
    # is the flip present within a single version?
    intra = any(len(set(1 if x>=1 else 0 for x in v)) > 1 for v in bysuf.values())
    if intra: vm_flips += 1
    print(f"  {g}: suffixes={len(bysuf)} lcs_by_suffix={dict((k[:6],v) for k,v in bysuf.items())} intra_version_flip={intra}")
print(f"flip games with flip WITHIN a single version: {vm_flips}/{len(flips)}")

# --- 2. Empirical depth discount (RL-A / ME-NEW-2 / PS-N4) ---
# good runs: lc>=1. For run with N total actions, value fraction achieved by action N-t
def clears_by_action(r, t):
    apl = r['actions_per_level']; lc = r['levels_completed']
    cum = 0; k = 0
    for i in range(lc):
        cum += apl[i]
        if cum <= t: k += 1
        else: break
    return k

good = [r for r in d if r['levels_completed'] >= 1]
print(f"\ngood runs: {len(good)}")
for t in (90, 120, 150):
    fracs = []
    for r in good:
        N = len(r['history'])
        rem = max(N - t, 0)
        v = clears_by_action(r, rem) / r['levels_completed']
        fracs.append(v)
    print(f"discount(t={t}): mean value fraction in first N-{t} actions = {statistics.mean(fracs):.3f} (median {statistics.median(fracs):.3f})")

# FP rates: good runs with first clear after t
for t in (90, 120, 150):
    fp = sum(1 for r in good if r['actions_per_level'][0] > t)
    print(f"FP(t={t}): {fp}/{len(good)} = {fp/len(good):.3f}")

# --- 3. EV model: trigger x cap, discount sensitivity ---
# per-game: p = good rate, V = mean game score of good seeds (levels/number_of_levels)
# a run stuck (lc==0) at trigger t restarts; each restart samples good mode w.p. p achieving disc*V
# FP: good run with first clear > t restarts, loses its value, recovers p*disc*V
pg = {}
for g, runs in bygame.items():
    goods = [r for r in runs if r['levels_completed'] >= 1]
    p = len(goods)/len(runs)
    V = statistics.mean(r['levels_completed']/r['number_of_levels'] for r in goods) if goods else 0.0
    pg[g] = (p, V)

def ev(t, cap, disc):
    total = 0.0
    for g, runs in bygame.items():
        p, V = pg[g]
        for r in runs:
            first = r['actions_per_level'][0] if r['levels_completed'] >= 1 else None
            if first is None:
                # bad run: restarts fire, gains sum over cap restarts
                gain = sum(((1-p)**(k-1)) * p * disc * V for k in range(1, cap+1))
                total += gain
            elif first > t:
                # FP: loses own value (levels), recovers via restarts
                own = r['levels_completed']/r['number_of_levels']
                rec = sum(((1-p)**(k-1)) * p * disc * V for k in range(1, cap+1))
                total += rec - own
    return total / len(d) * 25  # per-run avg -> per-25-game-draw game pts... total/250 runs *25 games = per-draw pts
print("\nEV table (game-points per 25-game draw; local score units)")
print("disc \\ (t,cap):", [(t,c) for t in (90,120,150) for c in (1,2)])
for disc in (0.2, 0.4, 0.6):
    row = [f"{ev(t,c,disc):+.3f}" for t in (90,120,150) for c in (1,2)]
    print(f"  {disc}: {row}")

# --- 4. R2 honest null bound (RL-M2 residual / ME-M4) ---
q_seed = 3/10  # rule-of-three 95% UB per-seed
def p_game(q): return 3*q*q*(1-q) + q**3  # >=2/3 seeds
pgm = p_game(q_seed)
p2of3 = 3*pgm*pgm*(1-pgm) + pgm**3
print(f"\nR2 honest bound: per-seed UB q={q_seed}, P(game passes >=2/3)={pgm:.3f}, P(>=2 of 3 games)={p2of3:.3f}")
