# WINNING-SOLUTION PROPOSAL v4 — ARC-AGI-3 Campaign

**Author:** claude-fable-5 · **Date:** 2026-07-06 · **Revision of:** v3 · **Targets:** Milestone #2 (Sep 30, top-3 paid), Final (Nov 2, top-5 paid, ~55 private games)

---

# Thesis

The money is decided on the private ~55-game set, where per-game heuristics compress (AERA, arXiv:2605.25931). Duck harness as floor in week 1, then three capabilities, riskiest premise tested first: (1) a deterministic exploration substrate (scripted-first `explore()`, cost-aware frontier archive, gated per-game on a determinism/aliasing audit); (2) a runtime Rodionov verify-refactor world-model loop whose load-bearing premise — Qwen 3.6 27B synthesizing Class-A transition models under runtime caps — is a Phase-0 pilot **run under the Phase-2 context regime**; (3) a TTT value net, entry-gated offline, **Nov-2 build only**. Every merge passes a game-level sign-flip gate with family-wise error control, a pre-registered MDE, and a shrinkage-detection confirmation.

# Evidence base

1. **Duck = 1.21% Kaggle RHAE**, Qwen 3.6 27B FP8, vLLM, 28 threads, one `python` tool, segmentation object graphs, persisted world model; 25 games × 2.2 h in ~12 h wall — token-bound. Weaknesses: irreproducibility, level-transition amnesia, no systematic exploration.
2. **Kaggle scored kernels run on RTX PRO 6000 (Blackwell, 96 GB, native FP8)** — per duck's `GPU_NAME_PATTERNS`; confirmed from logs Phase 0 hour 1. The A40 (Ampere, no FP8) pilot thus runs a numerically different model (FP8-dequant→BF16); Phase 0c anchors this.
3. Public-LB noise: two identical-code pairs (0.43/0.22, 0.33/0.24); σ is measured, not assumed.
4. v35 at structural ceiling (0.25). Rodionov loop survives its leakage audit (58.12%, GPT-5.5 — evidence for the loop, not 27B). Rudakov exploration: 3rd on Preview private LB, LLM-free.
5. Live assets: 25 opus sims (build-methodology evidence only), BFS planner, GraphExplorer, RHAE-correct harness, daemon. Process rules: fork-never-build, preflight, byte-matched metadata, smoke test.

# Statistical instrument (pre-registered)

**Exchangeable unit = game.** Gate statistic: mean over dev games of **per-game mean deltas** (deltas averaged over seeds within game). Significance: **exact game-level sign-flip permutation test** (2¹⁸ patterns on dev-18; min p ≈ 4×10⁻⁶). Seeds are never pooled as observations.

**Null instrument, calibration, failure branch.** Phase 0 runs the frozen duck build at 8–10 seeds on **dev-18** (the gate support). The identical-build null for the 3-vs-3-seed statistic is **all 280 unordered disjoint 3/3 splits** of 8 baseline seeds (560 ordered; we evaluate the 280 unordered, sign-symmetrized). The splits are mutually dependent, so this null is a **calibration check**: the sign-flip test applied to all 280 must reject at ≈ nominal α = 0.0125, judged by a **hierarchical-bootstrap 95% CI** (calibration-test α = 0.05) on the realized rejection rate. **Pre-registered failure branch:** (i) CI lower bound **above** nominal (anti-conservative) → all gates run at the **empirical percentile of the split-null** (critical value set so split-null rejection = 0.0125), its dependence-induced imprecision reported beside every gate decision; (ii) realized rate **below** nominal → sign-flip proceeds, conservative; (iii) CI covers nominal → proceed. No mid-July discretion.

**Family-wise error control.** Gate looks enumerated: Phase-1 gate + 1 retry (2), Phase-2 gate + 2 scaffold retries (3), voting (1), Phase-3 (1), reserve (1) = **8 looks**. Per-look α = 0.0125; gate-level family-wise ≤ 0.10; with the confirmation stage below, **merge-level family-wise false-merge ≈ 8 × 0.0125 × 0.13 ≈ 0.013**. Hyperparameter screens run at 1 seed; only the selected configuration faces the gate — one look. The ablation matrix is estimation; no looks.

**Confirmation (winner's-curse shrinkage detection, not re-test).** Every gate winner re-runs on fresh seeds, N = selection N (≥3). **Criterion: confirmation delta ≥ 50% of the gate critical value c** (normal approximation to the sign-flip null: c ≈ 2.24·SE); below c/2 reverts the merge. Operating characteristics: at true effect = MDE (≈ 3.08·SE), gate power 0.80, confirmation power Φ(3.08 − 1.12) ≈ 0.975, **joint ≈ 0.78**; under the null, P(pass both) ≈ 0.0125 × 0.131 ≈ 0.0016 per look. The confirmation detects selection-inflated deltas; a full-threshold re-test would cost 0.80² = 0.64 end-to-end power, rejected.

**Power / MDE (end-to-end).** Phase 0 publishes the **MDE at 80% gate power ⇒ ~78% joint power** (α = 0.0125, n = 18, 3 seeds/arm) from the measured variance decomposition; **the seed-raising rule and kill labels are calibrated to the 78% joint number**. Plausible-effect bands: substrate +0.10–0.25 RHAE, world model +0.15–0.40, voting +0.05–0.15, TTT +0.05–0.15. **Rule (formula-executed):** if MDE exceeds a band lower bound, raise seeds/arm 3→5→8 until MDE ≤ bound (≤ ~$130/sweep). If N = 8 still fails, the gate runs at N = 8 with MDE printed beside the decision; a kill with MDE > plausible effect is an **"underpowered kill"**, eligible for one Nov-window revisit.

**Regressions:** per-game BH-FDR q = 0.1.

**LB σ.** 4–6 byte-identical frozen-fork submissions; σ̂ reported **with its χ² CI** (df 3–5 → ~[0.6σ̂, 2.9σ̂]); every σ̂-based criterion evaluated at both endpoints. Cross-environment: per-game rank correlation local-vs-Kaggle, **disattenuated** via per-game reliabilities; disattenuated ρ < 0.5 → local gating not used for submission selection.

**Splits.** dev-18 / holdout-7 (phase-boundary sweeps + final selection) / **vault: 5 procedural-variant games from public-game mechanics, hash-frozen by Aug 1, run at ≥3 seeds** for the Nov-2 report (extra seeds consume no decisions), never used for any decision. **Claim scope: the vault, reported as mean ± CI, bounds catastrophic overfitting; it does not estimate private-set performance** — its frame is biased toward mechanics our pipeline targets. All gates stated on dev-18; 25-game numbers are reporting only.

**Per-game binary aggregation.** Wherever a count threshold rests on a per-game binary (≥6/18, zone 3–5/18, kill <3/18), the binary is **majority over the 3 gate seeds (≥2/3)**; the pilot publishes the seed-to-seed sampling SD of the count, reported beside every zone/kill decision.

# The plan (Jul 7 – Sep 30)

**Cadence:** daemon queue never empty; weekly depth check.

### Phase 0 — Instrument + kill-questions (Jul 7–20)
**(a) Duck fork verbatim**, fresh slug, byte-matched metadata — 2 scored attempts, one reserved for week 2. **Taxonomy: infra-failure (ERROR/OOM/timeout/metadata) = free retry; score-failure counts.** Verify whether scored reruns draw from interactive quota. From logs: GPU SKU, per-game tokens **T_game**, tokens/turn distribution.
**(b) Local null:** 8–10 seeds × dev-18 on RunPod A40 — ~45–95 A40-h at planning parity, 2–3 spot pods; fallback if token parity doubles cost: 8 seeds dev-18, 3-seed variance holdout-7. Measure tokens/s parity and max vLLM concurrency at 32k on the A40; if ≪ 28, per-turn token yield re-denominated. **All turn caps token-denominated (duck's p90 tokens/turn), never wall-clock.**
**(c) Qwen-27B synthesis pilot — n = 10 games.** Arms: 3 scaffold variants (fill-in-skeleton / free-form / diff-refactor) on recorded histories, leave-one-game-out scaffolding. **The deciding LOGO prompts conform to the Phase-2 runtime context table: transition evidence capped at 6k tokens, selected by the runtime's own policy — changed-frame transitions first, then most-recent unchanged-frame, evicted oldest-first — plus the 1k verify-report slot.** The unconstrained-history (≤32k) number is run once per scaffold, **reported separately as an upper bound only; it decides nothing.** (6k ≈ 40–50 transitions at duck's ~120–150 tokens/transition encoding — measured Phase 0; the 30-transition prospective verification streams, never held in context.)
**Entry counting rule:** a pilot game counts as Class-A only under its **LOGO-selected single scaffold** (selected on the other 9 games) — one scaffold, one shot per game, no best-of-three. The runtime ships **one** scaffold (the overall LOGO winner) under the 40% synthesis allocation; a router winner re-closes the allocation table before entry.
Plus **≥2 games closed-loop** (agent self-collects transitions via scripted `explore()` on the A40), reporting **tokens-to-first-Class-A per game** against the allocation: explore 30% / synthesize+verify+refactor 40% / plan 5% / act 25% of T_game. **Pre-registered 1/2 interpretation:** 1 pass / 1 fail satisfies the ≥1-closed-loop entry requirement but sets a **closed-loop-thin flag**: tokens-to-first-Class-A is re-measured under the gated Phase-1 `explore()` before the mid-Phase-2 zone decision; a violating re-measure drops Phase 2 to verify-only regardless of dev count.
**Quantization anchor (redefined):** 3 pilot games re-run **on the RTX PRO 6000 itself** inside the 6 h/wk smoke budget as a **battery of 5 independent synthesis attempts per game** (fixed prompts, generation seeds 0–4, LOGO-selected scaffold), paired with the identical local A40 battery. **Anchor statistic: mean over the 3 games of the paired per-game difference in held-out exact-match (pp) — continuous.** Acceptance: |mean paired diff| ≤ 10 pp, **valid only if 10 pp ≥ 2·SE of the paired difference** (SE from the 15 attempt pairs); otherwise the battery expands 5→10 attempts/game; still noisier → anchor unmeasurable and **Phase-2 entry decided on kernel-anchored numbers** (same as a tripped anchor). Binary Class-A flips are a **reported side flag, never the criterion**.
**Class-A (one definition):** held-out exact-match on 5-step open-loop rollouts ≥ max(identity-frame, pure-lookup, lookup-with-identity-fallback) + 10 pp. Pilot split: temporal 70/30, changed-frame stratum reported separately. Acceptance score = held-out exact-match (pp) − λ·gzip-KB of full source incl. data literals, λ = 2 pp/KB; train-vs-held-out gap = memorization flag. **Generation headroom is measured, not assumed:** the pilot logs per-scaffold generation-length distributions (incl. thinking tokens); the Phase-2 headroom line is pinned to the **measured p90 per selected scaffold**, and **truncation-rate at that headroom is a scaffold-selection criterion** alongside acceptance score.
**Planner telemetry:** abstraction pre-registered — ≤5 movement/rotate primitives plus ACTION6 on ≤16 salience-tier objects, b ≤ 21. Phase 0 logs realized branching, node counts, plan-depth distribution; 22 opus sims re-scored on end-state-match and goal-attainment separately, by plan-length bucket (1–3 / 4–6 / 7+).
**(d) Determinism/aliasing audit:** 3 prefixes × 3 replays at varied depths per game; aliasing scan necessary-not-sufficient; any post-audit runtime contradiction auto-disables the archive per game.
**Exit gates:** Kaggle ≥0.9 within 2 scored attempts; null + calibration verdict + MDE published; pilot 6k-regime Class-A, closed-loop, anchor, planner numbers recorded; Phase 1–3 gates instantiated.

### Phase 1 — Exploration substrate (Jul 21–Aug 3)
**Build:** scripted `explore()` trigger — progress ≐ new deduped archive state OR score/level increment; N ∈ {5, 10, 20} screened at 1 seed, selected N faces the gate. LLM self-routing is the A/B variant. **Frontier economics:** the only reset primitive is full-episode RESET (scorecard-counted; no save-state), so a frontier return costs a prefix replay of length d; frontier score = novelty / (1 + return_cost); the ablation reports the probe / return / progress decomposition beside its RHAE delta. `bfs_solve()`: 10 s / 2×10⁵ nodes, anytime, over the pre-registered abstraction. Phase-1 context table (~29.5k/32k) stands.
**Gate:** sign-flip on dev-18 at α = 0.0125, reported with/without the 14 clone-history games; confirmation at N = selection seeds against c/2.

### Phase 2 — Runtime executable world models (Aug 4–24; entry conditional)
**Entry gate:** **≥4/10 pilot games Class-A on 6k-regime LOGO numbers under the per-game LOGO-selected single scaffold** (kernel-anchored if the anchor tripped or was unmeasurable), **including ≥1 closed-loop game** (thin-flag rule applies). **Entry artifact — Phase-2 context table (synthesize/verify/refactor turn, worst case):** prompt 4k, schemas 1.2k, model source 3.5k (p75 of pilot-accepted lengths; measured p75 > 3.5k re-closes the table before entry), template skeleton 2.5k, verify report 1k, transition window 6k, history 4k (evicted first), **generation headroom = measured pilot p90 for the shipped scaffold (planning number 8k; re-closed to measurement)** ≈ 30.2k/32k. Models > 4k tokens post-refactor rejected (MDL-consistent).
**Build:** synthesize → verify → refactor → plan, replanning-on-contradiction. **In-kernel verification is prospective:** the next 30 live transitions after acceptance; prospective exact-match < accepted score − 15 pp demotes the model.
**Gate (dev-denominated):** Class-A (majority over 3 seeds per game) on **≥6/18 dev**, and on those games **end-state plan-transfer ≥40% AND goal-attainment ≥25%** at realized plan depths (bucket-reported); paired sign-flip at α = 0.0125; count's seed-sampling SD reported. **Zone 3–5/18:** verify-only + contradiction signal, no BFS-through-model; scaffold retries within the 8-look budget. **<3/18:** synthesis killed permanently.

### Phase 3 — TTT (Aug 25–Sep 14) — **Nov-2 build only**
TTT never ships in the Sep-30 build. Offline entry gate unchanged (Spearman ρ ≥ 0.3, ≥30 informative events/game, else cancelled pre-slot). Co-residency: A40 measurement plus one in-kernel co-resident tokens/s smoke on the Kaggle SKU pre-merge; tax enters the 15% rule ex ante. Ablation + confirmation Sep 15–Oct 15. JEPA-XXS: one clean trial; one clean ERROR → dead.

### Phase 4 — Hardening & milestone (Sep 8–30)
**Milestone build = Phase-0–2 only. Freeze Sep 12–15.** Ablation matrix (estimation, MDE-derived seed counts). 5–6 scored draws at ≤2/wk. **Selection rule (also picks the Nov-2 submission): K ≤ 3 pre-registered candidates** (frozen milestone build ± two gated variants) on **highest holdout-7 mean RHAE at 3 fresh seeds; near-ties (<1 SE) broken by CVaR over the 2 worst games**. Phase 0 publishes **P(select the truly better config | true Δ = 0.05, 0.10)** for K = 2, 3 from its variance components; P(correct | 0.10) < 0.7 cuts K to 2. The holdout is consumed as a canary; the vault-5 (≥3 seeds, mean ± CI, catastrophic-overfitting bound only) is the untouched Nov-2 report. **Success criterion: mean of draws ≥ frozen-fork mean + 2σ̂·√(1/n_draws + 1/n_frozen)** — SE of the difference, both means noisy — at both endpoints of σ̂'s χ² CI; floor: mean ≥ duck-fork mean.

**Oct 1–Nov 2:** open-source-wave absorption drill; Phase-3 confirmation; vault run.

# Slot ledger & resources

**Slot classes:** **A** = local A40 sweep (3 seeds × dev-18, 20–40 A40-h, $8–16); **B** = Kaggle scored run (12 GPU-h, quota-counted); **C** = Kaggle smoke (≤1 h). Of 81 slots: ~61 A, ~18 B, ~2 C batches. **Class-B ledger (pessimistic branch — reruns quota-counted — default):** wks 1–2 duck ×2 (+1 reserved retry); wks 3,5,7,9,11 sigma ×5; wks 4,6,8,10 dev-validation ×4; wks 11–12.5 draws ×5–6. Total ≈ 17–18 ≤ 24. **Triage:** voting loses Kaggle validation first, then Phase-1 substrate (local-only), then Phase-3; **sigma runs and Phase-4 draws never cut.** Optimistic branch (reruns quota-free, tested Phase 0): validation doubles.
**RunPod A40:** ~$1,900 through Sep 30, +≤$300 seed raises; ≤$150/wk. Preemption: resume-and-include; non-resumable runs excluded, counts reported. **3080:** CNN prototyping, offline TTT gate. **Opus:** ~5M tokens/wk (KAOS, opus-4-8).

# Kill criteria (pre-registered)

- **P0:** duck <0.8 after 2 scored attempts + 4-day bisect (dead Jul 24) → CPU-track pivot. Pilot <4/10 on 6k-regime LOGO under LOGO-selected scaffolds → Phase 2 pre-killed; residual = duck + substrate + voting.
- **P1:** gate + one retry fail (2 looks) → scripted-fallback-only; still regresses → pure duck.
- **P2:** zone rules above; kills at MDE > plausible effect carry the underpowered-kill label (joint-power).
- **P3:** offline gate fails → cancelled pre-slot; JEPA one clean ERROR → dead.
- **Global:** any component costing >15% of the measured per-game token budget without a passed gate is removed.

# Risks

1. **27B synthesis fails (high; tested Jul 7–20 under the runtime context regime, closed-loop and on-SKU arms).** Residual band 1.2–1.5 — milestone-competitive, not thesis-fulfilling. Accepted.
2. **Quantization gap A40↔Blackwell** — bounded by the 3-game, 5-attempt paired continuous anchor; unmeasurable anchor → kernel-anchored entry.
3. **LB σ exceeds local signal** — local-first gating; disattenuated transfer check first.
4. **Sep 30 open-source wave (certain)** — compose-with-anything assets + absorption drill.
5. **Kernel/env fragility (proven)** — fork-never-build, preflight, smoke, infra-retry taxonomy.

# Change log from v3

- **[llm-NN1]** Deciding LOGO prompts **conform to the Phase-2 runtime context regime**: transition evidence capped at 6k under the runtime's own selection/eviction policy + 1k verify-report slot; ≤32k-history number reported separately as upper bound only. Entry gate, P0 kill, Risk 1 restated on 6k-regime numbers.
- **[llm-NN2]** Anchor redefined on **continuous paired per-game held-out exact-match (pp)**: battery = 5 attempts/game × 3 games, local vs on-SKU; acceptance |mean paired diff| ≤ 10 pp, valid only if ≥ 2·SE; noise branch 5→10 attempts, else unmeasurable → kernel-anchored entry. Binary Class-A flips = side flag only.
- **[meth-N1]** Calibration-failure branch: 280 unordered disjoint splits (fixes v3's "280 ordered"), hierarchical-bootstrap 95% CI (calibration α = 0.05); CI above nominal → gates at split-null empirical percentile, dependence-imprecision reported; below → sign-flip proceeds as conservative.
- **[meth-N2]** Option (b) adopted: confirmation = **delta ≥ 50% of gate critical value** (shrinkage detection, not re-test). Joint power at MDE ≈ **0.78** (vs 0.64 full-threshold); per-look false-merge ≈ 0.0016; merge-level family-wise ≈ 0.013. Seed-raising rule and underpowered-kill label recalibrated to the joint number.
- **[llm-NN3]** Entry counting: per-game Class-A under the **LOGO-selected single scaffold**; no best-of-three; runtime ships one scaffold; router winner re-closes the allocation table.
- **[llm-NN4]** Generation headroom = **measured p90** per scaffold (8k = planning number); **truncation-rate is a scaffold-selection criterion**.
- **[llm-NN5]** 1/2 closed-loop: entry met + **thin flag**; re-measure under gated Phase-1 `explore()` before the zone decision; allocation violation → verify-only zone.
- **[llm-Q6]** 6k ≈ 40–50 transitions (measured Phase 0); the 30-transition prospective verify streams outside context.
- **[meth-N3]** Phase-4 criterion = SE of the difference: **2σ̂·√(1/n_draws + 1/n_frozen)**.
- **[meth-N4]** Vault at **≥3 seeds**, mean ± CI; claim = **bounds catastrophic overfitting, not private-set estimate**.
- **[meth-N5]** Final selection **K ≤ 3**; Phase 0 publishes P(select truly better | Δ = 0.05, 0.10); P(correct | 0.10) < 0.7 cuts K to 2.
- **[meth-N6]** Per-game Class-A binary = **majority over 3 seeds**; count thresholds use it; seed-sampling SD of the count reported beside every zone/kill decision.

---
*Sources: panel_research_*.md; reviews in learnings/panel/round1–3/.*

---

# Post-panel amendments (methodology round-4 residuals, adopted verbatim)

**A1 [N7 — anchor SE at game level].** The quantization-anchor statistic is the mean of the 3 per-game mean paired differences, SE = SD(per-game means)/√3, critical value from t(df=2). If the acceptance criterion is unsatisfiable at df=2 (expected), the pre-registration defaults to **kernel-anchored Phase-2 entry** (the RTX-PRO-6000 battery numbers decide), stated now — not discovered mid-July.

**A2 [N8 — equivalence test].** Anchor acceptance = TOST at α=0.05: the 90% CI of the game-level mean paired difference must lie entirely within ±10 pp. The 5→10 attempt expansion handles precision failure.

**A3 [N9 — conservative-branch MDE].** In calibration branch (ii) (realized α below nominal), the MDE is recomputed at the realized rejection rate and printed beside every kill decision; kills against the optimistic MDE are labeled underpowered.

**A4 [N10 — calibration check power].** Phase 0 publishes the calibration check's minimum detectable miscalibration (smallest realized α whose CI excludes nominal with 80% probability, from the same hierarchical bootstrap) beside the calibration verdict. A "pass" certifies only what this number says it certifies.

# Panel provenance

- Process: 4-round adversarial review, 5 independent reviewer personas (RL/planning, LLM-agents, program-synthesis, methodology, systems) on claude-fable-5 via KAOS (agent_sdk provider). Pass bar pre-registered: ≥4/5 ACCEPT + 0 FATAL. No one-shot pass permitted.
- Round 1 (v1): 0/5 accept — 5 FATAL, scores 4-5/10.
- Round 2 (v2): 0/5 accept — 0 FATAL, ~18 MAJOR, scores 6-7/10. All round-1 FATALs resolved.
- Round 3 (v3): 3/5 accept (8, 9, 8) — 4 MAJOR remaining, scores 7-9/10.
- Round 4 (v4): llm-agents flipped to ACCEPT (9/10). **Final: 4/5 ACCEPT, 0 FATAL — PASSED.** Methodology dissent recorded at 9/10 with 4 residual objections (N7-N10), all adopted as amendments A1-A4 above.
- Full record: learnings/panel/round{1,2,3,4}/ (25 reviews + author responses).
