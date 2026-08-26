# Daily Brief — 2026-07-14

## 1. Yesterday's results

### 1a. WAR BUILD v1 scored **0.91** (LB, rerun completed overnight)
- Arm: frozen duck + warpack (banking = max-over-plays winning-trace replay, recovery, retry_guard, shortcircuit, soft_end 11h20m, fast-submit gate). Ledger module aboard, **flags OFF** (reserved for pre-registered A/B).
- Submitted 00:13Z Jul 14 by ARCDailySubmit; `SubmissionStatus.COMPLETE`, publicScore **0.91**.
- vs control {0.82, 0.89, 0.93, 1.02, 0.95}: mean 0.922, σ̂ 0.074 → z ≈ **−0.16**. Null-consistent: warpack draw #1 shows **no lift, no harm**.
- Interpretation caution: warpack's headline EV (+0.10–0.15) was attributed to *resubmission order statistics* (max over daily draws), which by construction cannot show in a single draw. Banking's within-run max-over-plays upside also can't be observed directly — the scored rerun is a **separate hidden execution** (fast-submit gate confirmed working: build log shows dummy parquet written 0.6 s after start, `RUN_HEAVY=False`). We have zero per-game artifacts from any scored run (this answers panel R9 ME-NEW-12's factual question: pulled kernel outputs are NOT the scored rerun).
- Fast-submit gate is now **live and validated** → daily resubmission costs ~no GPU quota. This is the R1 mechanism working as designed.

### 1b. Leaderboard shift — new #1 at **1.86** (YUTO KOJIMA)
- Jumped past Tecnod8.AI (1.61) overnight; +0.25 over prior top. Zero public kernels/datasets found (same opacity as Tecnod8.AI).
- 1.44 resubmission wall intact (Lonnie/Tshithihi/Figuring-out at 1.44; trio at 1.56). thtennant ran duck v13 on 07-13.
- Our LB best = 1.02 (draw #4). Gap to wall ≈ 0.42; gap to top ≈ 0.84.

### 1c. Panel round 9 (path_forward_v4): NOT approved — 5× MAJOR-REVISION, 0 fatal
Recurring, **free-to-fix** methodology majors:
- **Variance incoherence (ME-NEW-11 / P):** null10 bootstrap 1-seed Δ sd = 0.52 vs control-implied ≈ 0.074·√2 ≈ 0.10 — 5–7× discrepancy, unreconciled. Every H4 threshold and the 3-seed gate's power depend on which is right. Fix from existing transcripts: publish bootstrap resampling unit + per-rail variance decomposition.
- **Scored-run identity (ME-NEW-12 / Q):** answered above (hidden rerun, no per-game artifacts). Remaining feasible piece: aggregate wheel-formula reconstruction vs the 7 LB totals in submission history.
- **R1b seed-split contamination (ME-NEW-10 / N):** (90, cap 2) selection was informed by all of null10; held-out seeds 6–10 not independent. Requires seeds-1–5-only sweep + leave-one-game-out jackknife (gains concentrate in ft09/tn36/tu93 = 3-game bet) or fresh local null seeds.
- **§Instruments section cited but never written** (rl-planning M).

### 1d. Queue/state
- Queue pending: sched-v1 window-gate draw #2 (deferred behind war build). Not empty ✓.
- Pushes today: 0/2. GPU reserve untouched ($68). Retry look (phase-1) still unspent; phase-1 line CLOSED.
- Ledger+escalation graft: built, smoke-tested (48/48 incl. banking verbatim-replay on real engines), aboard war kernel with flags OFF. Pre-registered predictions P1–P5 in `learnings/war_room/intervention_plan.md` §R2.

## 2. Today's open questions (for panel)

- **Q1 — Tonight's arm.** Options: (a) war v2 = flip `ledger+escalation` ON → begins the pre-registered R2 A/B (window 1 of ≥3); (b) another war-v1 draw to grow the warpack order-stats sample; (c) sched-v1 draw #2 (stale — sched line superseded by warpack recovery?). Note (a) and (b) are not exclusive across days: A/B design already alternates arms across windows. Proposal: **(a)** — R2 was built to ship, predictions are pre-registered, and rollback is a flag.
- **Q2 — What can 0.91 tell us about banking?** Nothing per-game (hidden rerun). Only aggregate inference over accumulating draws. Should the warpack arm get its own control-band accounting (it is NOT the frozen fork — trusted-fork queue rules need updating)?
- **Q3 — Which R9 majors block Track A vs which are paper-only?** Proposal: variance reconciliation (ME-NEW-11/P) is **blocking** — it decides whether 3-seed gates have any power — and is free; do it today. R1b contamination fixes are pre-registration edits, not compute. §Instruments is a writing task for path_forward_v5.
- **Q4 — 1.86 response.** Does the new top change anything? Author position: no — our binding constraint is our own 0.92 band vs the 1.44 wall; the wall mechanism (fast-submit + order stats) is now adopted; grinder-cracking (R2–R5) is the leveler. Chasing an opaque 1.86 with zero public evidence is luck-chasing.

## 3. Constraints unchanged
Zero cloud spend (reserve panel-gated); ≤2 kernel pushes/day; fork-never-build; screens ≥3 seeds where scored; Kaggle rail binding; queue never empty by 18:00; submit as soon as validated (19:00 EDT hard stop).
