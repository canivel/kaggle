# WINNING-SOLUTION PROPOSAL v3 — ARC-AGI-3 Campaign

**Author:** claude-fable-5 · **Date:** 2026-07-06 · **Revision of:** v2 after 5× MAJOR-REVISION · **Targets:** Milestone #2 (Sep 30, top-3 paid), Final (Nov 2, top-5 paid, ~55 private games)

---

# Thesis

The money is decided on the private ~55-game set, where per-game heuristics compress (AERA, arXiv:2605.25931). Duck harness as floor in week 1, then three capabilities, riskiest premise tested before any slots commit: (1) a deterministic exploration substrate (scripted-first `explore()`, cost-aware frontier archive, gated per-game on a determinism/aliasing audit); (2) a runtime Rodionov verify-refactor world-model loop whose load-bearing premise — Qwen 3.6 27B synthesizing Class-A transition models under runtime caps — is a Phase-0 pilot; (3) a TTT value net, entry-gated offline, **shipping only in the Nov-2 build**. Every merge passes a game-level sign-flip permutation gate with family-wise error control and a pre-registered minimum detectable effect.

# Evidence base

1. **Duck = 1.21% Kaggle RHAE**, Qwen 3.6 27B FP8, vLLM, 28 threads, one `python` tool, segmentation object graphs, persisted world model; 25 games × 2.2 h in ~12 h wall — token-bound. Weaknesses: irreproducibility, level-transition amnesia, no systematic exploration.
2. **Kaggle scored kernels run on RTX PRO 6000 (Blackwell, 96 GB, native FP8)** — asserted by duck's `GPU_NAME_PATTERNS`; confirmed from kernel logs Phase 0 hour 1. The A40 (Ampere, no FP8) pilot thus runs a numerically different model (FP8-dequant→BF16, AWQ-int8 fallback); Phase 0c anchors against this.
3. Public-LB noise: two identical-code pairs (0.43/0.22, 0.33/0.24) — point estimates; σ is measured, not assumed.
4. v35 at structural ceiling (0.25). Rodionov loop survives its leakage audit (58.12%, GPT-5.5 — evidence for the loop, not 27B). Rudakov exploration: 3rd on Preview private LB, LLM-free.
5. Live assets: 25 opus sims (build-methodology evidence only), BFS planner, GraphExplorer, RHAE-correct harness, daemon. Hard process rules unchanged: fork-never-build, preflight, byte-matched metadata, smoke test.

# Statistical instrument (pre-registered)

**Exchangeable unit = game.** The gate statistic is the mean over dev games of **per-game mean deltas** (each game's delta averaged over its seeds). Significance: **exact game-level sign-flip permutation test** (2¹⁸ = 262,144 patterns on dev-18; min attainable p ≈ 4×10⁻⁶). Seeds are never pooled as observations; v2's "~75 paired observations" is retracted as pseudoreplication.

**Null instrument and resampling recipe.** Phase 0 runs the frozen duck build at 8–10 seeds on **dev-18** (the gate support). The identical-build null for the actual 3-vs-3-seed statistic is built from **all disjoint 3/3 seed splits** of the baseline seeds (8 seeds → 280 ordered disjoint pairs). These splits are mutually dependent, so this null serves as a **calibration check** — the sign-flip test applied to null splits must reject ≈ nominal α, with a hierarchical-bootstrap CI on that realized rate — not as the primary threshold. The sign-flip test is primary; the split-null validates it and supplies variance components for power.

**Family-wise error control.** All gate looks are enumerated now: Phase-1 gate + 1 retry (2), Phase-2 gate + 2 scaffold retries (3), voting (1), Phase-3 (1), reserve (1) = **8 looks**. Per-look α = 0.0125 (98.75th percentile of the sign-flip null), family-wise false-merge probability ≤ 0.10. Hyperparameter screening inside a component (e.g., trigger-N sweep) runs at 1 seed and only the selected configuration faces the gate — one look. The ablation matrix is estimation, not gating, and takes no looks.

**Power / MDE.** Phase 0 publishes the **minimum detectable effect at 80% power** (α = 0.0125, n = 18 games, 3 seeds/arm) from the measured variance decomposition. Pre-registered plausible-effect bands: substrate +0.10–0.25 RHAE, world model +0.15–0.40, voting +0.05–0.15, TTT +0.05–0.15. **Rule (formula-executed, no discretion):** if MDE exceeds a component's band lower bound, raise seeds/arm 3→5→8 until MDE ≤ bound (≤ ~$130/sweep at 8 seeds, inside budget). If N = 8 still fails, the gate runs at N = 8 with MDE printed beside the decision; a kill where MDE > plausible effect is labeled **"underpowered kill"**, eligible for one Nov-window revisit.

**Winner's-curse correction.** Every selected winner is re-run on **fresh seeds equal in number to the selection arm (N ≥ 3)**; the confirmation number is the reported number. A confirmation contradicting the gate (delta below the gate threshold) reverts the merge.

**Regressions:** per-game checks at BH-FDR q = 0.1 (unchanged).

**LB σ.** 4–6 byte-identical frozen-fork submissions; σ̂ reported **with its χ² CI** (df 3–5 → roughly [0.6σ̂, 2.9σ̂]); every σ̂-based criterion evaluated at both endpoints. Cross-environment check: per-game rank correlation local-vs-Kaggle, **disattenuated** via per-game reliabilities from the seed null and LB σ experiment; disattenuated ρ < 0.5 → local gating not used for submission selection.

**Splits.** dev-18 / holdout-7 (phase-boundary sweeps + final selection) / **vault: 5 procedural-variant games generated from public-game mechanics, hash-frozen by Aug 1, executed exactly once for the Nov-2 generality report** and never used for any decision. All gates are stated on dev-18; 25-game numbers are reporting only.

# The plan (Jul 7 – Sep 30)

**Cadence invariant:** daemon queue never empty; weekly queue-depth check.

### Phase 0 — Instrument + kill-questions (Jul 7–20)
**(a) Duck fork verbatim**, fresh slug, byte-matched metadata — 2 scored attempts, one reserved for week 2. **Pre-registered taxonomy: infra-failure (ERROR/OOM/timeout/metadata — no score) = free retry; score-failure counts.** Verify whether scored reruns draw from the interactive quota. Extract from logs: GPU SKU confirmation, realized per-game tokens **T_game**, tokens/turn distribution.
**(b) Local null:** 8–10 seeds × dev-18 on RunPod A40 — **~45–95 A40-h at planning parity, 2–3 concurrent spot pods**; if measured token parity doubles the cost, fall back (pre-registered) to 8 seeds on dev-18, 3-seed variance on holdout-7. Measure tokens/s parity and **max sustainable vLLM concurrency at 32k on the 48 GB A40**; if ≪ 28, per-turn token yield re-denominated. **All turn caps everywhere are token-denominated (duck's p90 tokens/turn), never wall-clock.**
**(c) Qwen-27B synthesis pilot — n = 10 games (pre-registered).** Arms: 3 scaffold variants (fill-in-skeleton / free-form / diff-refactor) on recorded histories with **leave-one-game-out scaffolding — LOGO numbers are the deciding numbers**; plus **≥2 games closed-loop** (agent collects its own transitions via scripted `explore()` on the A40), reporting **tokens-to-first-Class-A per game** against the pre-registered allocation: explore 30% / synthesize+verify+refactor 40% / plan 5% / act 25% of T_game. **Quantization anchor:** 3 pilot games re-run as a synthesis battery **on the RTX PRO 6000 itself** inside the 6 h/wk smoke budget; pre-registered acceptance |kernel − local Class-A rate| ≤ 15 pp — if exceeded, Phase-2 entry is decided on kernel-anchored numbers.
**Class-A (one definition):** held-out exact-match on 5-step open-loop rollouts **≥ max(identity-frame, pure-lookup, lookup-with-identity-fallback) + 10 pp** (absolute-50% dropped). Pilot split: **temporal 70/30 (train earlier, verify later), changed-frame stratum reported separately.** Acceptance score = held-out exact-match (pp) − λ·gzip-KB of full source **including data literals**, λ = 2 pp/KB; train-vs-held-out gap reported as a memorization flag.
**Planner telemetry:** planner action abstraction pre-registered — ≤5 movement/rotate primitives plus ACTION6 on ≤16 salience-tier segmentation objects, **branching factor b ≤ 21** (depth ~4+ at the 2×10⁵-node cap, deeper with dedup). Phase 0 logs realized branching factor, node counts, plan-depth distribution on pilot games; the 22 opus sims are re-scored reporting **end-state-match and goal-attainment rates separately, by plan-length bucket (1–3 / 4–6 / 7+)**.
**(d) Determinism/aliasing audit:** **3 prefixes × 3 replays at varied depths per game**; the aliasing scan is treated as necessary-not-sufficient; **any post-audit runtime contradiction auto-disables the archive for that game**.
**Exit gates:** Kaggle ≥0.9 within 2 scored attempts; null + MDE published; pilot Class-A, closed-loop, anchor, planner-telemetry numbers recorded; Phase 1–3 gates numerically instantiated.

### Phase 1 — Exploration substrate (Jul 21–Aug 3)
**Build:** scripted `explore()` trigger — **progress ≐ new deduped archive state OR score/level increment; N ∈ {5, 10, 20} screened at 1 seed, selected N faces the gate**. LLM self-routing is the A/B variant. **Frontier economics:** the live API's only reset primitive is full-episode RESET (scorecard-counted; no save-state), so a frontier return costs a prefix replay of length d; frontier score = **novelty / (1 + return_cost)**; the archive ablation reports the **probe / return / progress action decomposition** beside its RHAE delta. `bfs_solve()`: 10 s / 2×10⁵ nodes, anytime, over the pre-registered abstraction. Phase-1 context table (~29.5k/32k) stands.
**Gate:** sign-flip on dev-18 at α = 0.0125, reported with/without the 14 clone-history games, fresh-seed confirmed at N = selection seeds.

### Phase 2 — Runtime executable world models (Aug 4–24; entry conditional)
**Entry gate:** **≥4/10 pilot games Class-A on LOGO numbers** (kernel-anchored if the anchor tripped), across the 3 scaffolds, **including ≥1 closed-loop game**. **Entry artifact — Phase-2 context table (synthesize/verify/refactor turn, worst case):** prompt 4k, schemas 1.2k, current model source 3.5k (**budgeted at the p75 of pilot-accepted model token lengths; if measured p75 > 3.5k the table is re-closed before entry**), one template skeleton 2.5k, verify report 1k, transition window 6k, history 4k (evicted first), generation headroom 8k ≈ 30.2k/32k. Models > 4k tokens post-refactor are rejected (MDL-consistent).
**Build:** synthesize → verify → refactor → plan, replanning-on-contradiction default. **In-kernel verification is prospective:** the next 30 live transitions after acceptance; prospective exact-match < accepted score − 15 pp demotes the model.
**Gate (dev-denominated):** Class-A on **≥6/18 dev** and, on those games, **end-state plan-transfer ≥40% AND goal-attainment ≥25%**, interpreted at realized plan depths (bucket-reported); paired sign-flip delta at α = 0.0125. **Zone 3–5/18 (pre-registered):** verify-only + contradiction signal, no BFS-through-model; scaffold retries within the 8-look family budget. **<3/18:** synthesis killed permanently.

### Phase 3 — TTT (Aug 25–Sep 14) — **Nov-2 build only**
**Committed resolution of the schedule conflict: TTT never ships in the Sep-30 milestone build.** Offline entry gate unchanged (Spearman ρ ≥ 0.3, ≥30 informative events/game, else cancelled before any slot). Co-residency: A40 measurement **plus one in-kernel co-resident tokens/s smoke on the Kaggle SKU before any merge**; tax enters the 15% rule ex ante. Phase-3 ablation + fresh-seed confirmation run Sep 15–Oct 15 (local A40 + October Kaggle quota). JEPA-XXS: one clean pre-registered trial; one clean ERROR → dead.

### Phase 4 — Hardening & milestone (Sep 8–30)
**Milestone build = Phase-0–2 components only. Freeze Sep 12–15.** Ablation matrix (estimation, MDE-derived seed counts); expected score from the measured joint effect. 5–6 scored draws at ≤2/wk. **Selection rule (also picks the Nov-2 submission): highest holdout-7 mean RHAE at 3 fresh seeds; near-ties (<1 SE) broken by CVaR over the 2 worst games.** Min-over-games is dropped as an extreme-value statistic. The holdout is thereby consumed as a canary — acknowledged; **the vault-5 is the untouched generality estimate**. **Success criterion: mean of draws ≥ frozen-fork mean + 2σ̂/√n_draws** (SE of the mean — the intended severity), evaluated at **both endpoints of σ̂'s χ² CI**; floor: mean ≥ duck-fork mean.

**Oct 1–Nov 2:** open-source-wave absorption (5-day drill); Phase-3 confirmation; vault run.

# Slot ledger & resources

**Slot classes:** **A** = local A40 sweep (3 seeds × dev-18, 20–40 A40-h, $8–16); **B** = Kaggle scored run (12 GPU-h, quota-counted); **C** = Kaggle smoke (≤1 h). Of 81 slots: ~61 class A, ~18 class B, ~2 class-C batches. **Class-B weekly ledger (pessimistic branch — reruns quota-counted — is the default schedule):** wks 1–2 duck ×2 (+1 reserved retry); wks 3,5,7,9,11 sigma ×5; wks 4,6,8,10 dev-validation ×4; wks 11–12.5 draws ×5–6. Total ≈ 17–18 ≤ 24 (12 wks × 2/wk cap). **Triage if weeks are lost:** voting loses Kaggle validation first, then Phase-1 substrate (local-only validation), then Phase-3; **sigma runs and Phase-4 draws are never cut.** Optimistic branch (reruns quota-free, tested Phase 0): validation doubles.
**RunPod A40:** ~$1,900 through Sep 30, +≤$300 for MDE-driven seed raises; ≤$150/wk. **Preemption: resume-and-include default; only non-resumable runs excluded, exclusion counts reported per sweep.** **3080:** CNN prototyping, offline TTT gate. **Opus:** ~5M tokens/wk (KAOS, opus-4-8).

# Kill criteria (pre-registered)

- **P0:** duck <0.8 after 2 scored attempts + 4-day bisect (dead Jul 24) → CPU-track pivot. Pilot <4/10 LOGO across all scaffolds → Phase 2 pre-killed; residual = duck + substrate + voting (an honestly well-run fork).
- **P1:** gate + one retry fail (2 looks) → scripted-fallback-only; still regresses → pure duck.
- **P2:** zone rules above; kills at MDE > plausible effect carry the underpowered-kill label.
- **P3:** offline gate fails → cancelled pre-slot; JEPA one clean ERROR → dead.
- **Global:** any component costing >15% of the measured per-game token budget without passing its gate is removed.

# Risks

1. **27B synthesis fails (high; tested Jul 7–20, closed-loop and on-SKU arms).** Residual band 1.2–1.5 — milestone-competitive, not thesis-fulfilling. Accepted.
2. **Quantization gap A40↔Blackwell** — bounded by the 3-game kernel anchor.
3. **LB σ exceeds local signal** — local-first gating; disattenuated transfer check first.
4. **Sep 30 open-source wave (certain)** — compose-with-anything assets + absorption drill.
5. **Kernel/env fragility (proven)** — fork-never-build, preflight, smoke, infra-retry taxonomy.

# Change log from v2

- **[meth-N1, rl-N1(1)]** Exchangeable unit = game; exact sign-flip on per-game mean deltas, n = 18; "~75 paired observations" retracted as pseudoreplication; power redone at n = 18.
- **[meth-N2]** Resampling recipe: disjoint 3/3 splits of the baseline seeds as a dependence-acknowledged calibration check (bootstrap CI on realized α); sign-flip primary.
- **[meth-N3, rl-N1(2), llm-Q4]** 8 gate looks enumerated; per-look α = 0.0125, family ≤ 0.10; MDE at 80% power pre-registered; seed raising 3→5→8 by formula; underpowered-kill label; seed count is a Phase-0 output.
- **[meth-PR curse]** Confirmation seeds = selection seeds (≥3); contradiction reverts the merge.
- **[meth-N4]** Selection = holdout-7 **mean** at 3 fresh seeds (CVaR tiebreak); min-over-games dropped; **vault-5 procedural games hash-frozen Aug 1** as the untouched Nov-2 generality report.
- **[meth-m1]** σ̂ χ² CI reported; criterion restated as 2σ̂/√n_draws (intended severity), evaluated at both endpoints.
- **[meth-m2]** Transfer check disattenuated; threshold ρ < 0.5 pre-registered.
- **[llm-N1, ps-NEW-1b, sys-N3]** Kaggle SKU named (RTX PRO 6000 Blackwell, FP8-native, per duck's GPU_NAME_PATTERNS); 3-game on-SKU synthesis anchor in smoke budget; ≤15 pp pre-registered discrepancy; **all turn caps token-denominated (duck p90 tokens/turn)**.
- **[llm-N2, ps-NEW-1a/c]** ≥2 closed-loop pilot games; token allocation 30/40/5/25 of T_game as entry artifact; tokens-to-first-Class-A reported; entry requires ≥1 closed-loop pass.
- **[ps-NEW-2]** Lookup-with-identity-fallback as third scored baseline; Class-A = exact-match ≥ max(baselines) + 10 pp; absolute 50% dropped.
- **[ps-NEW-3]** Pilot split temporal 70/30 + changed-frame stratum; in-kernel = prospective verification on next 30 live transitions, with demotion rule.
- **[ps-M5 residual]** Phase-2 entry decided on LOGO-scaffolded numbers.
- **[ps-NEW-4]** MDL = λ·gzip-KB incl. data literals, λ = 2 pp/KB; train-vs-held-out gap reported.
- **[ps-NEW-5]** End-state-match and goal-attainment reported separately; gate on conjunction (≥40% / ≥25%).
- **[ps-NEW-6]** Pilot n = 10; bar 4/10 (40%), consistent with dev gate 6/18 (33%).
- **[llm-N3]** All gates restated on dev-18 (≥6/18; zone 3–5/18); 25-game counts reporting only.
- **[llm-N4]** "No-progress" defined; N ∈ {5,10,20} screened at 1 seed, one gate look.
- **[llm-PR context]** Phase-2-entry context table for the synth/verify/refactor turn, template line-item, model-source budget pinned to measured p75 of accepted lengths.
- **[rl-N2]** Reset primitive stated (full-episode RESET only); frontier score novelty/(1+return_cost); probe/return/progress decomposition in the ablation.
- **[rl-N3]** Planner abstraction pre-registered (≤5 primitives + ≤16 objects, b ≤ 21); Phase-0 telemetry (branching, nodes, depths); plan-transfer per depth bucket.
- **[rl-N4]** Audit = 3 prefixes × 3 replays, varied depths; aliasing scan necessary-not-sufficient; runtime contradiction auto-disables archive per game.
- **[rl-N5]** Null run on dev-18 — the gates' support.
- **[sys-N1]** Slot classes in GPU-hours; weekly class-B ledger under both quota branches (pessimistic default, 17–18 ≤ 24); triage order pre-registered.
- **[sys-N2]** Committed: **TTT ships only in the Nov-2 build**; Sep-30 freeze covers Phase-0–2 only; Phase-3 confirmation Sep 15–Oct 15.
- **[sys-N4]** Infra-failure vs score-failure taxonomy; one attempt reserved for week 2.
- **[sys-N5]** Resume-and-include default; exclusion counts reported.
- **[sys-PR residuals]** Null costed (45–95 A40-h, 2–3 pods, fallback rule); in-kernel co-resident tokens/s smoke on the SKU before any Phase-3 merge; A40 max concurrency measured, per-turn yield re-denominated.

---
*Sources: panel_research_{literature,winners,ourstack,lb}.md; reviews in learnings/panel/round1/ and round2/.*
