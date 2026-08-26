# Gap Forensics: Tufa 1.21/1.6 vs our fork {0.82, 0.89} — 2026-07-09

## TL;DR — the catch
There are two catches, and neither is our config. **(1) Measurement basis:** 1.6 is public-train (their 25-game local set); the official Kaggle set is 110 different games and is ~27–35% harder for this harness — Tufa's own official draws were {0.77, 1.21, 1.30-retracted}. **(2) Selection/luck:** the notebook we forked carries Tufa's own admission (cell 0): *"this notebook is a more readable version of the notebook that scored our milestone-winning 1.21; unfortunately, we haven't had the same lucky result with this one."* 1.21 was a right-tail draw and Kaggle LB reports max-of-draws. Our {0.82, 0.89} brackets their own 0.77 official draw and matches milestone 3rd place (0.86, mbmmurad). The repro is faithful.

## A. Apples-to-apples (Monte Carlo from score.json, 500 game_runs)
Their 20 complete passes on the 25 public games: mean 1.6002, sd 0.4591, min 0.999, max 2.382 — a single pass on the *public* set already dips to 1.0. MC single-draw on 110 games (game+seed bootstrap from their empirical per-game distributions), calibrated to candidate official means:

| Official-mean assumption | sd (110g draw) | pct(0.82) | pct(0.89) | P(max of 10 draws ≥1.5) |
|---|---|---|---|---|
| 1.21 (their milestone = the mean) | 0.213 | 2.2% | 5.5% | 0.62 |
| 1.093 (mean of their 3 official draws) | 0.192 | 6.7% | 14.3% | 0.22 |
| 1.00 (1.21 was lucky, per their note) | 0.176 | 15.1% | 27.7% | 0.05 |

Uncalibrated (official set assumed as easy as public): our draws would be <0.2 percentile — decisively rejecting "same difficulty" and confirming the basis gap. Under the defensible calibrations (their 1.21 being a selected lucky draw), our draws sit at roughly the **7th–28th percentile**: below center, jointly ~1–8% under mean≈1.05 — mildly unlucky, not anomalous, and consistent with Tufa's own 0.77.

## B. Config diff (run_config.json vs fork bundle)
- **Model: IDENTICAL.** `vrfai/Qwen3.6-27B-FP8`; the fork attaches `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` — the only Qwen snapshot dataset that exists (`kaggle datasets list`). Only wheelhouse v3 exists; no v4+, no bigger snapshot. Model-delta theory: **dead**.
- Example run (SLURM, 2 GPUs): 64 effective concurrency, 45 min/game, analyzer_timeout 120s, 20 passes. Kaggle bundle: concurrency 28, 7920s (132 min)/game, analyzer_timeout 900s, 1 pass. These are **Tufa's own Kaggle scalings** inside their shared bundle — not deltas we introduced.
- Caveats found: example run is June-2 code (commit dd50b55, benchmark label `0-history-turns`) vs bundle June-12 (ARC3-Inference aa69123 **DIRTY**). The variance dataset comes from a slightly older build than the notebook, and we forked the "readable rewrite," not the literal milestone kernel (`taaf-duck-harness-kaggle`, which they mark "not recommended"). Any readable-vs-original delta is unmeasured, but Tufa's own reruns of the readable one also underperformed 1.21.

## C. Per-game structure
Top-5 contributors = 57% of the 1.60: **ft09** 10.28 (26% of total alone), **r11l** 3.71, **vc33** 3.34, **sb26** 2.83, **lp85** 2.72. ft09-like outliers being rarer in the 110-game official mix mechanically explains 1.6→~1.1. Bimodality: 10/25 games bimodal (cd82, cn04, dc22, g50t, ls20, m0r0, re86, s5i5, sc25, sk48); 14 games score >40% of tries; tr87 and wa30 never score; g50t/m0r0/tr87/wa30 essentially never leave level 1. sb26/tn36 (animation-blind per writeup) are in both sets; they still score via level-1 grinding.

## D. LB toppers
Current LB wall: 1.56 / 1.56 / 1.56 / 1.48 / 1.46… — no 1.9 crowd. The recent-kernels feed is duck forks (`arc3 duck v7/v8/v9`, `taaf-duck-harness-grammar`, multiple straight copies). Under mean 1.21, P(max of 10 resubmits ≥ 1.5) ≈ 0.62 — the 1.4–1.56 wall is exactly what order statistics predicts from duck forks resubmitted daily, **no config change required**. No newer driessmit1 datasets exist for them to have swapped in.

## E. Verdict (ranked by evidence)
1. **Measurement-basis difference — CONFIRMED, primary.** 1.6 (25 public games) vs official 110-game set; their official band is 0.77–1.30. Comparing our Kaggle draws to 1.6 is a category error.
2. **Pure variance + LB max-of-draws selection — CONFIRMED, secondary.** 1.21 is a lucky draw by the authors' own admission in the forked notebook; our two draws are 7th–28th percentile of their implied single-draw distribution and bracket their 0.77.
3. **Config/model delta — REJECTED.** Same model, same wheelhouse, same served config; Kaggle-side settings are Tufa's own.
4. **Something else — minor residual:** we run the "readable" rewrite, not the literal 1.21 kernel; example-run code is 4 weeks older than the bundle. Unquantifiable, small.

**Change nothing in the fork.** To bank an LB number, resubmitting the identical fork ~8–10 more times has ~50–60% chance of a ≥1.4 draw — but that is luck-chasing (feedback_arc_generalization_first). Real EV: spend the compute on the Phase-1 differentiators already scoped in tufa_writeup_review_2026-07-08.md — harness-side exploration/dedup injection, vLLM prefix caching (more turns per game inside 9h), frame-diff animation summaries (sb26/tn36), and structured curated memory. Treat {0.82, 0.89} as repro-consistent per the P0 interpretation rule (draw in 0.77–0.9 = within Tufa's published variance band; skip the bisect).
