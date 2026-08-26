# Author Response — Round 2 → v3

**From:** Proposal Architect (claude-fable-5) · **Re:** winning_solution_v3.md · **Date:** 2026-07-06
All dispositions below are implemented in v3; section references are to v3.

---

## Methodology reviewer

**N1 (MAJOR) — "~75 paired observations" is pseudoreplication; permutation must randomize at game level.** ACCEPTED IN FULL. The exchangeable unit is now the game. Gate statistic = mean over dev-18 of per-game mean deltas (seeds averaged within game, never pooled as observations); significance from an exact game-level sign-flip permutation test (2¹⁸ patterns). The "~75 observations" claim is explicitly retracted in the Statistical Instrument section. Power/MDE is recomputed at n = 18. *(Answers Q1: sign-flip on per-game mean deltas, n = 18, dev support.)*

**N2 (MAJOR) — identical-build null undefined for the 3-vs-3-seed statistic.** ACCEPTED. Exact recipe pre-registered: all disjoint 3/3 ordered splits of the 8–10 frozen-baseline seeds (8 seeds → 280 pairs). Because the splits are mutually dependent, this null is demoted from primary threshold to **calibration check**: the sign-flip test applied to null splits must realize ≈ nominal α, with a hierarchical-bootstrap CI reported on that realized rate. The sign-flip test is the primary gate; the split-null validates it and supplies variance components. *(Answers Q2.)*

**N3 (MAJOR) — no family-wise control, no MDE.** ACCEPTED. All looks enumerated ex ante: Phase-1 gate + 1 retry, Phase-2 gate + 2 scaffold retries, voting (1), Phase-3 (1), reserve (1) = **8 looks**; per-look α = 0.0125 (98.75th percentile), family-wise false-merge ≤ 0.10. Hyperparameter screening (e.g., trigger-N) runs at 1 seed and only the selected config faces a gate; the ablation matrix is estimation and takes no looks. MDE at 80% power (α = 0.0125, n = 18, 3 seeds/arm) is a mandatory Phase-0 output, with pre-registered plausible-effect bands per component and a formula-executed rule: raise seeds 3→5→8 until MDE ≤ band lower bound; if N = 8 still fails, MDE is printed beside every decision and kills where MDE > plausible effect are labeled "underpowered kill" (one Nov revisit). Seed count is thus a Phase-0 output, not the constant 3. *(Answers Q3, Q4.)*

**Winner's curse residual (confirmation n=2 < selection n=3).** ACCEPTED. Confirmation seed count = selection seed count (≥3, scales with MDE-driven raises); a confirmation below the gate threshold reverts the merge. *(Answers Q5.)*

**N4 (MAJOR) — worst-case-over-7-games selection is extreme-value noise and burns the holdout.** ACCEPTED. Selection statistic changed to holdout-7 **mean** RHAE at 3 fresh seeds; near-ties (<1 SE) broken by CVaR over the 2 worst games. Min-over-games is dropped. Holdout consumption at selection is acknowledged; the untouched generality estimate is now a **vault of 5 procedural-variant games, generated and hash-frozen by Aug 1, executed exactly once for the Nov-2 report** and never used for any decision. *(Answers Q6; Q7 is mooted by dropping the min statistic, though its sampling SD will still be reported from the null for the record.)*

**MINOR — σ̂ CI.** ACCEPTED. χ² CI on σ̂ reported; every σ̂-based criterion evaluated at both endpoints. The Phase-4 criterion is restated as mean ≥ frozen-fork mean + 2σ̂/√n_draws — i.e., 2 standard errors of the mean, the intended severity (v2's 2σ̂-on-single-draw-scale wording was an error, not an intention).

**MINOR — attenuation-blind rank correlation.** ACCEPTED. Disattenuated using per-game reliabilities from the seed null and the LB σ experiment; pre-registered threshold: disattenuated ρ < 0.5 → local gating not used for submission selection.

---

## RL/planning reviewer

**N1 (MAJOR) — α controlled but not β; retries inflate α.** ACCEPTED, jointly with methodology-N3 (see above): 3-vs-3 split null as calibration (part 1), 8 enumerated looks at per-look α = 0.0125 (part 2), MDE at 80% power with the seed-raising rule. At $8–16/sweep the raises are budgeted (+≤$300 contingency). *(Answers your Q1–Q3.)*

**N2 (MAJOR) — frontier-return cost unbudgeted under RHAE.** ACCEPTED. v3 states the reset surface: the live API's only primitive is full-episode RESET (scorecard-counted; no save-state), so a frontier return costs a prefix replay of length d. Frontier selection is now cost-aware — score = novelty / (1 + return_cost), with expected return cost computed from Phase-1 logs — and the archive ablation must report the probe/return/progress action-count decomposition beside its RHAE delta. *(Answers Q4; the measured mean return cost on the deepest public games is a Phase-1 log output.)*

**N3 (MAJOR) — node cap arithmetically vacuous without an action abstraction.** ACCEPTED. The planner's abstraction is pre-registered: ≤5 movement/rotate primitives plus ACTION6 restricted to ≤16 salience-tier segmentation objects → branching factor b ≤ 21, giving depth ~4+ at 2×10⁵ nodes uniform and deeper with dedup. Phase 0(c) logs realized branching factor, node counts, and plan-depth distribution on the pilot games; the 22-sims re-scoring reports plan-length distribution; plan-transfer thresholds are interpreted **per plan-length bucket (1–3 / 4–6 / 7+)**, not at the 5-step Class-A horizon. *(Answers Q5, Q6: transfer is reported bucketed, gated on the aggregate conjunction with buckets shown.)*

**N4 (MINOR) — audit sampling too thin.** ACCEPTED. 3 prefixes × 3 replays at varied depths per game; aliasing scan treated as necessary-not-sufficient; any post-audit runtime contradiction auto-disables the archive for that game. *(Answers Q7.)*

**N5 (MINOR) — null/dev support mismatch.** ACCEPTED. The null is run on dev-18, the identical support the gates use; holdout-7 gets 3-seed variance estimates for selection-time use only.

---

## LLM-agents reviewer

**N1 (MAJOR) — pilot runs a different numeric model than the kernel.** ACCEPTED. The Kaggle SKU is named in the Evidence Base: **RTX PRO 6000 (Blackwell, 96 GB, native FP8)**, per duck's `GPU_NAME_PATTERNS`, log-confirmed Phase 0 hour 1. Calibration anchor: 3 pilot games re-run as a synthesis battery on the RTX PRO 6000 inside the existing 6 h/wk smoke budget. Pre-registered acceptance: |kernel − local Class-A rate| ≤ 15 pp; if exceeded, Phase-2 entry is decided on the kernel-anchored numbers. All turn caps are token-denominated (duck's p90 tokens/turn), never wall-clock, on both platforms. *(Answers Q1.)*

**N2 (MAJOR) — pilot doesn't test self-collected data or the joint budget.** ACCEPTED. (a) ≥2 of the 10 pilot games run **closed-loop**: the agent collects its own transitions via scripted `explore()` on the A40 — no curated recorded histories. (b) A pre-registered per-game token allocation is a Phase-2 entry artifact: explore 30% / synthesize+verify+refactor 40% / plan 5% / act 25% of T_game (duck's measured per-game tokens, extracted Phase 0a). The pilot reports **tokens-to-first-Class-A per game** against that allocation, and Phase-2 entry requires ≥1 closed-loop pass. *(Answers Q2.)*

**Partial-resolution residual — context budget doesn't close for Phase 2.** ACCEPTED. A second, Phase-2-entry context table for the synthesize/verify/refactor turn is now specified: prompt 4k + schemas 1.2k + current model source 3.5k + one template skeleton 2.5k (the missing line-item) + verify report 1k + transition window 6k + history 4k (first evicted) + generation headroom 8k ≈ 30.2k/32k. The model-source line is pinned to the **measured p75 of pilot-accepted model token lengths**; if p75 > 3.5k the table must be re-closed before entry; models >4k post-refactor are rejected (consistent with the MDL penalty). *(Answers Q3: the opus-sim/accepted-model length distribution is a Phase-0 measured output, and the 1.5k figure no longer load-bears.)*

**Partial-resolution residual — power/seed count.** ACCEPTED — see methodology-N3: variant seed count is a Phase-0 output via the MDE rule. *(Answers Q4.)*

**N3 (MINOR) — gates over 25 games conflict with holdout protocol.** ACCEPTED. All Phase-1–3 gates restated on dev-18: Phase-2 gate ≥6/18 (33% ≈ old 8/25), zone 3–5/18; 25-game counts are boundary-sweep reporting only. *(Answers Q5.)*

**N4 (MINOR) — scripted trigger unspecified.** ACCEPTED. Progress ≐ new deduped archive state OR score/level increment; N ∈ {5, 10, 20}, screened at 1 seed in Phase-1 week 1, with only the selected N facing the gate (one look). *(Answers Q6.)*

---

## Program-synthesis reviewer

**NEW-1 (MAJOR) — pilot regime mismatch (recorded data, dequant weights, open loop).** ACCEPTED on all three prongs: ≥2 closed-loop pilot games (own-policy data on the A40); 3-game synthesis battery on the actual Kaggle SKU with a ≤15 pp pre-registered discrepancy bound (entry decided on kernel-anchored numbers if tripped); tokens-to-first-Class-A against the pre-registered allocation covers the joint-budget question. *(Answers Q5.)*

**NEW-2 (MAJOR) — weak baselines, no margin.** ACCEPTED. **Lookup-with-identity-fallback** is added as the third scored baseline. Class-A is redefined as held-out exact-match ≥ max(identity, pure-lookup, lookup-with-identity-fallback) + **δ = 10 pp**, pre-registered; the absolute-50% threshold is dropped. *(Answers Q1.)*

**NEW-3 (MAJOR) — held-out split construction unspecified.** ACCEPTED exactly as prescribed: pilot split = temporal 70/30 (train earlier, verify later) with a changed-frame stratum reported separately; in-kernel definition = **prospective verification** on the next 30 live transitions after acceptance, with a demotion rule (prospective < accepted score − 15 pp), composing with replanning-on-contradiction. *(Answers Q2.)*

**MAJOR-5 residual — LOGO audit had no decision rule.** ACCEPTED. Phase-2 entry is evaluated on the **LOGO-scaffolded pilot numbers** — the only condition that models the private set. Full-template numbers are reported for diagnosis only. *(Answers Q4.)*

**NEW-4 (MINOR) — MDL parameters absent; branch-count evadable.** ACCEPTED. Penalty = λ·gzip-KB of full source **including data literals** (dict-literal lookup tables pay full freight), λ = 2 pp held-out exact-match per KB, pre-registered; train-vs-held-out gap reported as a memorization flag. Branch-count is dropped as the unit. *(Answers Q3.)*

**NEW-5 (MINOR) — plan-transfer omits goal attainment.** ACCEPTED. End-state-match rate and goal-attainment rate are reported separately (including for the 22 re-scored sims, by plan-length bucket); Phase 2 gates on the **conjunction**: end-state ≥40% AND goal-attainment ≥25%. *(Answers Q6.)*

**NEW-6 (MINOR) — pilot n unpinned.** ACCEPTED. n = 10 (your recommendation; the A40 is otherwise idle); entry bar 4/10 = 40%, consistent with the dev-18 gate of 6/18 ≈ 33%. *(Answers Q7.)*

---

## Systems reviewer

**N1 (MAJOR, borderline FATAL) — slot ledger contradicts the quota cap.** ACCEPTED. A "slot" is now typed and costed: class A = local A40 sweep (3 seeds × dev-18, 20–40 A40-h, $8–16); class B = Kaggle scored run (12 GPU-h, quota-counted); class C = smoke (≤1 h). Of the 81 slots, ~61 are class A — the quota confusion arose because v2 never said most slots are local. Class-B weekly ledger, with the **pessimistic branch (reruns quota-counted) as the default schedule**: wks 1–2 duck ×2 (+1 reserved retry); sigma in wks 3/5/7/9/11 (×5); dev-validation in wks 4/6/8/10 (×4); draws ×5–6 in wks 11–12.5 — total ≈ 17–18 ≤ 24 cap. This also answers your sigma-vs-validation interleaving question (Q4): alternate weeks, never both. Pre-registered triage if weeks are lost: voting loses Kaggle validation first, then Phase-1 substrate (local validation only), then Phase-3; sigma runs and Phase-4 draws are never cut. Optimistic branch simply doubles validation. *(Answers Q1, Q4.)*

**N2 (MAJOR) — Phase-3/4 schedule impossibility.** ACCEPTED — we pick and commit to your second option: **TTT ships only in the Nov-2 build, never Sep-30.** The milestone build freezes Sep 12–15 with Phase-0–2 components only; Phase 3 keeps its Aug 25–Sep 14 window but targets Nov 2, with its ablation + fresh-seed confirmation running Sep 15–Oct 15 on the A40 plus October Kaggle quota. The conflict no longer exists. *(Answers Q5.)*

**N3 (MAJOR) — SKU unnamed; wall-denominated pilot caps.** ACCEPTED. SKU named in the revision itself (RTX PRO 6000 Blackwell, FP8-native, from duck's `GPU_NAME_PATTERNS`; log-confirmed hour 1). All turn caps campaign-wide — including the pilot's — are denominated in tokens (duck's p90 tokens/turn) via the measured throughput ratio, never wall seconds. *(Answers Q2; the A40:Kaggle tokens/s ratio itself is a Phase-0 measurement.)*

**Residuals from your partial resolutions.** ACCEPTED: (i) the null is costed — 45–95 A40-h at planning parity, 2–3 concurrent spot pods, with a pre-registered fallback (8 seeds dev-18, 3-seed holdout variance) if measured parity doubles the estimate *(answers Q3)*; (ii) batch/turn parity — Phase 0 measures the A40's max sustainable vLLM concurrency at 32k and re-denominates per-turn token yield if ≪28 *(answers Q6)*; (iii) one in-kernel co-resident tokens/s smoke on the Kaggle SKU is required before any Phase-3 merge.

**N4 (MINOR) — zero week-1 margin.** ACCEPTED. Pre-registered taxonomy: infra-failure (ERROR/OOM/timeout/metadata, no score returned) = free retry that does not consume an attempt; score-failure counts. One attempt is reserved for week 2.

**N5 (MINOR) — preemption-exclusion bias.** ACCEPTED. Resume-and-include is the default; only non-resumable runs are excluded, with exclusion counts reported per sweep so denominators are auditable.

---

## Closing

Every round-2 MAJOR and MINOR across the five reviews has a concrete, pre-registered mechanism in v3; no objection is deferred to "Phase 0 will decide" unless the reviewer's own fix was a Phase-0 measurement, and in each such case the decision rule that consumes the measurement is written down now. The program-synthesis reviewer offered ACCEPT on receipt of a metric-spec addendum without a full round — v3's Phase-0(c) subsection plus this letter's NEW-2/3/4/5/6 dispositions constitute that addendum.
