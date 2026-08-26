# Position Analysis — 2026-07-13 (Milestone-2 horizon: Sep 30, ~80 windows left)

## 1. Where we stand (LB snapshot 2026-07-12, 1708 teams)

| Cutoff | 07-06 | 07-12 | climb/day |
|---|---|---|---|
| #1 | 1.56 | 1.61 | +0.008 |
| top-10 | 1.35 | 1.46 | +0.018 |
| top-20 | — | 1.37 | ~+0.02 |
| top-50 | — | 1.26 | — |
| top-100 | 1.04 | 1.17 | +0.022 |

**Us: rank 187 at 1.02** (best of 5 frozen-fork draws {0.82, 0.89, 0.93, 0.95, 1.02}, mean 0.922, σ̂ 0.074). The post-Tufa code-diffusion surge is still running. Linear extrapolation to Sep 30 is absurd (top-10 → 2.9); assuming S-curve decay as the fork-and-tweak crowd saturates, projected Sep-30 cutoffs: **top-10 ≈ 1.7–2.0, top-20 ≈ 1.6–1.8, top-50 ≈ 1.5–1.6, top-100 ≈ 1.35–1.5.**

## 2. P(top-N by redraws alone) ≈ zero

Per-draw tail vs today's (not Sep-30) cutoffs, N(0.922, 0.074), 80 draws:

- top-100 (1.17): z=3.4, p≈3.6e-4/draw → **~2.8% over 80 draws** — and the cutoff outruns us.
- top-50 (1.26): z=4.6 → ~2e-4 total.
- top-20 (1.37) / top-10 (1.46): z=6.1 / 7.3 → **~0**.

Expected best-of-80 redraws ≈ 0.922 + 2.47σ ≈ **1.10** — below today's top-100. A 1.35 draw needs +0.43, 4× our observed total spread. **Variance cannot save us; only a true-mean lift can.**

## 3. Required true-mean lift (difficulty ratio 0.55)

Local vanilla null = 1.69; official-set mean = 0.922; +X local → +0.55X official.

| Target official draw | Official Δ | Local Δ needed | Local null must reach |
|---|---|---|---|
| 1.35 median (top-20 today) | +0.43 | **+0.78** | **2.47** (+46%) |
| 1.26 mean (1.35 as +1.2σ best-of-10) | +0.34 | +0.62 | 2.31 (+37%) |
| 1.6–1.8 (Sep-30 top-20 proj.) | +0.7–0.9 | +1.25–1.6 | 2.9–3.3 |

So the minimum interesting local A/B effect is **≈+0.6 to +0.8 total levels-equivalent on the null**, compounded across merged variants. Substrate v1's Δ+0.169 (p=0.31) was an order of magnitude short even if real.

## 4. Submission policy: stop redrawing, fund the gates

- σ̂ is characterized (5 draws, χ²-CI [0.044, 0.213]); draw #6 completes the panel band, then **frozen-fork redraws have near-zero information and near-zero rank EV** (§2).
- Allocate windows: **~85% gated-variant screens/confirmations** (v2 substrate screen v5 pending, then hybrid duck+BFS, per-game router, gated exec-WM), **~1 redraw/week** only as drift sentinel + queue-never-empty filler (standing rule).
- Kaggle keeps best score, so a failed variant submission costs nothing on the LB — the only real cost is the window. That asymmetry favors spending windows on variants, not redraws.

**Bottom line:** we are ~rank 187 and drifting backward at fixed skill. Milestone-2 top-20 requires roughly **doubling local null (1.69 → ~2.9)** via 2-3 stacked, permutation-gated wins. Every window from draw #7 onward should test a candidate lift.
