# Headroom analysis — null10 true-mean, no-context levers (2026-07-13)

Data: `runs/null10/merged_null_benchmark.json` (10 seeds × 25 games, RHAE recomputed: triangular weights, 115 cap → mean **1.78** ± 0.61 across seeds, cap-100 variant 1.62; seed means span **0.79–2.93**). Tufa 20-pass reference: 1.60. Scripts in scratchpad (`headroom.py`, `death.py`, `levers.py`).

## 1. Distribution and bimodality (our 10 seeds)

- **5 games are dead in all 10 seeds** (dc22, g50t, m0r0, tr87, wa30): 0 levels ever, yet they consume **20.2% of all generated tokens** (3.07M/15.2M) for exactly 0 RHAE.
- **4 games are fully reliable** (lp85, r11l, sb26, su15): lc≥1 in 10/10 seeds, but **never clear level 2** in any seed.
- **8 flip games** and their stabilize-at-good-mode value (good-mode mean − overall mean, ÷25 = mean impact): **ft09 +0.198** (p_good 0.7), **tn36 +0.139** (0.4), **sk48 +0.115** (0.1), cn04 +0.065 (0.5), ar25 +0.059 (0.7), ls20 +0.057 (0.3), sc25 +0.056 (0.1), ka59 +0.029 (0.4). Total stabilization ceiling **+0.82 mean** (1.78→2.60); oracle best-seed-per-game bound +3.12.
- Score is concentrated: **ft09 alone is 26% of all points** (game mean 11.56, sd 9.93, lc 0–3). ft09 variance is most of the seed-mean variance.

## 2. Where good runs die: STUCK, not out of time

- 225/250 runs end in the last 5% of the ~7,900s shared wall (only 3 end before 6,000s) — the harness runs everything to the global wall; "gave_up" = wall exhaustion.
- But of 126 good runs (lc≥1), only **9 leveled up in the final 15%** of wall (genuinely out-of-time while progressing). Median **59% of a good run's tokens are burned on the one level it never finishes** (82/126 runs >50%; sb26 92%, su15 86%, lp85/r11l 65%). Median attempt depth on the failed level is ~3× its base actions.
- Verdict: extra raw budget buys little; the binding constraint is the level-2 wall, then compute waste.

## 3. Config levers (no context injection), ranked

| # | Lever | Expected gain (mean) | Risk | Cost |
|---|---|---|---|---|
| 1 | **Deprioritize lc==0 games after ~120 actions** (scheduler weight, game-agnostic): frees the 20% dead-game compute + bad-mode runs (124/250 runs are lc==0 and per gating doc never level) toward progressing games | +0.10–0.30 (soft: only 9/126 good runs were budget-limited, but deep grinders at 3× base benefit) | Low (p90 time-to-first-level = 94 actions → <10% FP at 120) | Low |
| 2 | **Restart policy: fresh episode if lc==0 at action 90** (the pre-registered v2 detector). By action 90 a bad-mode run has burned 61% of tokens; a restart leaves ~39% budget (median first level = 32 actions, so still reachable, depth discounted ~0.4). EV over the 8 flip games Σ(1−p)·p·good_mean·0.4 ≈ 3.2 game pts | **+0.13** | Medium (kills the ~10% late-bloomers; restarting at 40 is cheaper (36% tokens) but ~40% FP) | Low–Med |
| 3 | **Per-game wall/token reallocation** from reliable-but-capped games (sb26/su15/lp85 waste 65–92% of tokens grinding level 2) to high-variance ft09/tn36/vc33 — effectively multi-attempt on ft09, whose p_good=0.7 and +0.198 stabilization dominates | +0.05–0.20 | Medium (per-game budgets are private-set-unsafe unless keyed on observed lc, not game id) | Med |
| 4 | Temperature / n-parallel-threads | unquantifiable from null10 (no variation across seeds) — needs a dedicated A/B | ? | Low |

## 4. "+1 level on the 3 closest-but-fail games"

Closest = reliable grinders that clear level 1 in 10/10 seeds then park on level 2: **sb26, su15, lp85** (w=2, W=36/45/36). At observed median completed-level efficiency 0.83: 4.61+3.69+4.61 = 12.9 game pts → **+0.52 mean (1.78→2.29, +29%)**; even at crawl efficiency 0.25 it's +0.16. This is the single largest identified headroom, but it is a capability gap (level-2 mechanics), not a config knob — config levers 1–2 above are worth ~+0.25 combined; cracking any one reliable game's level 2 beats both.
