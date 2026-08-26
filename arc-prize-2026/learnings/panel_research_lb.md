# LB Dynamics — ARC-AGI-3 (snapshot 2026-07-06)

## Headline numbers (1,626 teams)
| Metric | Value |
|---|---|
| Leader | Mathurin Ache **1.56** (2026-07-05) |
| Top-10 cutoff | **1.35** |
| Top-20 cutoff | **1.28** |
| Top-50 cutoff | **1.16** |
| Median / Mean | 0.22 / 0.31 |
| Our best (0.27) | Sits in the single largest exact-score cluster (58 teams at 0.27) — the public-baseline band |

## Distribution
- 0.1–0.3 bins hold 782 teams (48%) — public-notebook territory. Exact-score clusters: 0.15 (63), 0.27 (58), 0.23 (53), 0.22 (50), 0.17 (53).
- High band (≥1.0): **125 teams**. Bins: 1.0→48, 1.1→34, 1.2→24, 1.3→13, 1.4→5, 1.5→1.
- No dominant exact score above 1.0 (max cluster: 9 teams at 1.01) — the high band is many small variations of the open-sourced Milestone-#1 code, not one copy-paste notebook. Copiers immediately fork and tweak.

## Timeline — diffusion speed after June 30 open-source
Last-submission dates of teams ≥1.0: Jun 30: 1 → Jul 1: 3 → Jul 2: 4 → Jul 3: 9 → Jul 4: 6 → **Jul 5: 49 → Jul 6: 53**. 82% of the ≥1.0 cohort landed in the last two days. Teams ≥1.16 (top-50 line): 54, of which 44 in the last two days. **The 1.2 band populated in ~5 days.** Code diffusion here is near-instant once milestone winners open-source.

## Extrapolation
- **Milestone #2 (Sep 30):** the 1.56-class code is now everyone's floor; 12 weeks of iteration on top of it. Given the 5-day diffusion rate and 125 teams already ≥1.0, expect the top-10 line at roughly **1.7–2.0** and the leader at 2.0+. Podium (paid) requires beating hundreds of forks with something original.
- **Final (Nov 2):** second open-source wave after Sep 30 resets the floor again. Public-LB top-10 plausibly **~2.0–2.4**.
- **Private-eval caveat (decisive):** private set ≈55 games vs 25 public (AERA claim). Per-game hardcoded heuristics — which is what most 1.0–1.3 forks are — should compress hard on unseen games. Final Top Score money is decided there, so the effective bar for a top-5 finish is a *general* agent averaging ≥1.5 levels/game on games it has never seen. Public-LB extrapolations above are upper bounds; generalization-first architecture (per feedback_arc_generalization_first) is the only path that survives the 55-game eval.

## Prize structure (arcprize.org/competitions/2026/arc-agi-3)
- **Grand Prize $700K**: first agent to 100%.
- **Top Score $75K**: 1st $40K, 2nd $15K, 3rd $10K, 4th $5K, 5th $5K → **only top-5 paid at final**.
- **Milestones $75K**: #1 (Jun 30) and #2 (Sep 30), each 1st $25K / 2nd $10K / 3rd $2.5K → **only top-3 paid per milestone**.
- Eligibility requires open-sourcing (CC0/MIT-0), which is exactly what drives the diffusion dynamics above.

## Implication
Top-10 public rank is achievable by riding the open-source wave, but money requires top-3 (milestone) or top-5 (final, on ~55 private games). Chasing the public 1.2 band is table stakes, not strategy.
