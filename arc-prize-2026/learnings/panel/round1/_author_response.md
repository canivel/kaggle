# Author Response — Round 1 (v1 → v2)

**Author:** claude-fable-5 · **Date:** 2026-07-06 · **Revised proposal:** `learnings/winning_solution_v2.md`

Disposition key: **RESOLVED-BY-CHANGE** (v2 changed), **REBUTTED** (with evidence), **ACKNOWLEDGED-AS-RISK** (accepted, why tolerable). Section references are to v2.

---

## RL-Planning reviewer

**[rl-F1] 3-seed instrument underpowered for every gate.** RESOLVED-BY-CHANGE. v2 §Measurement protocol: Phase 0 runs the frozen duck baseline at 8–10 seeds and publishes the per-game/aggregate null; every gate is redefined as "paired per-game delta > 90th percentile of the identical-build null" via paired permutation test over ~75 paired observations (25 games × 3 seeds), with fresh-seed confirmation after selection. Point thresholds (+3/+4/+2 levels) are deleted; numeric gates are instantiated *from* the measured null at Phase-0 exit. Answer to Q1: unknown today — which is exactly why no gate is numeric until the null exists.

**[rl-M1] Dedup archive assumes determinism + Markov-sufficient observations.** RESOLVED-BY-CHANGE. Phase 0(d): pre-registered determinism/aliasing audit — replay identical action prefixes twice per game and diff frames; scan logs for identical segmentation hashes preceding divergent transitions. The archive is enabled per-game only where both checks pass (answer to Q3: never run; now scheduled and gating). We do not claim reset-to-state; "return-to-frontier" is graph routing to a frontier-adjacent state, and on games failing the audit the archive is off entirely.

**[rl-M2] "BFS closes 14 games" contradicts baseline; clone-coupled.** RESOLVED-BY-CHANGE (the reviewer's "likely truth" is correct and v2 says so). `bfs_solve()` is a generic forward planner over a supplied transition function; the 14-game result was through hand-built clones, offline. v2 Thesis retracts it as private-set evidence, redraws the dependency (BFS value on unseen games is 100% conditional on Phase-2 synthesis), and reports Phase-1 deltas with the 14 clone-history games excluded. Answer to Q2: clone claim, not live-API; live click spaces are not claimed BFS-tractable without a model.

**[rl-M3] Replay accuracy is the wrong target; planner will exploit model error.** RESOLVED-BY-CHANGE. Class-A is redefined as ≥50% exact-match on held-out 5-step open-loop rollouts; Phase 2 is additionally gated on plan-transfer rate ≥40% (fraction of model-found plans realizing their predicted end state live); replanning-on-contradiction is the default control loop, not a fallback. Answer to Q4: plan-transfer was never measured on the 22 sims — Phase 0 re-scores them under both new metrics before Phase 2 exists.

**[rl-M4] TTT net has no plausible training signal.** RESOLVED-BY-CHANGE. Phase 3 now has an offline entry gate: CNN trained on logged Phase-1/2 trajectories must reach Spearman ρ ≥0.3 against realized progress AND logs must show ≥30 informative progress events/game on the target sparse games, else Phase 3 is cancelled before any slot is spent. Answer to Q5: unknown until Phase-1/2 logs exist; the gate makes the unknown decisive rather than hopeful.

**[rl-M5] Additive compounding (+3+4+2 → 1.6) unjustified; components attack the same bottleneck.** RESOLVED-BY-CHANGE. Sub-additivity conceded (all three convert wasted probes into directed ones and share wall/token budget). Phase-4 expectation now comes from the measured joint ablation effect; ≥1.6 is demoted to a reported aspiration; a pre-registered "still worth submitting" floor (mean ≥ frozen-fork mean) exists so the freeze isn't anchored to optimism.

**[rl-m1] Restart voting may cost RHAE.** RESOLVED-BY-CHANGE. Voting ablations are RHAE-denominated and include a no-restart arm. Answer to Q6: RHAE is efficiency-sensitive; the ablation measures the actual cost rather than us asserting one.

**[rl-m2] P2 4–7/25 zone undefined.** RESOLVED-BY-CHANGE. Pre-registered: 4–7 → verify-only + contradiction signal (no BFS-through-model), one retry across remaining scaffold variants; <4 after that → permanent kill. Answer to Q7 is now in the plan text.

---

## LLM-Agents reviewer

**[llm-F1] Both differentiators collapse to one untested dependency (27B synthesis).** RESOLVED-BY-CHANGE. Conceded — this was v1's central structural flaw. v2: (a) the Qwen-27B synthesis pilot moves to Phase 0 (A40, recorded transitions from 5–10 public games, runtime-realistic token/turn caps, Class-A rate under the new held-out metric); (b) Phase-2 entry is conditional on the pilot; (c) `bfs_solve()`'s dependency is stated in the Thesis; (d) the residual system if synthesis dies (duck + substrate + voting ≈ well-run fork, 1.2–1.5 band) is named in P0/Risk 1 instead of hidden. Answer to Q1: on a private game `bfs_solve()` operates on the Phase-2 runtime model or nothing; dependency graph redrawn accordingly. Answer to Q2: unmeasured — which is why it is now the first experiment of the campaign.

**[llm-M1] Rodionov de-risking is a category error; fix costs one afternoon.** RESOLVED-BY-CHANGE. Exactly the fix adopted: the afternoon experiment is Phase 0(c), with three pre-registered scaffold variants so a failure indicts the model, not one prompt.

**[llm-M2] LLM-routed tool delegation is the weakest pattern presented as default.** RESOLVED-BY-CHANGE. Inverted: scripted `explore()` trigger after N no-progress turns is the primary architecture; LLM self-routing is the A/B variant. Tool-distraction regressions are covered by the FDR-controlled per-game regression test against the null (which replaces the underpowered 2/3-seeds clause).

**[llm-M3] No context/token accounting for the 32k window.** RESOLVED-BY-CHANGE. Phase-1 entry artifact: worst-case per-turn budget table (duck prompt ~4k, tool schemas ~1.2k, WM blocks ~1.5k, archive frontier summary ~0.8k top-k only, transition window ~4k, history ~12k, thinking ~5k ≈ 29.5k/32k) with a fixed eviction order (history → templates → transition window; WM blocks never evicted). Numbers are estimates to be verified against real transcripts in Phase 1 — but the budget and eviction policy are now specified, not "truncation and caps."

**[llm-M4] Criteria pre-committed before variance known.** RESOLVED-BY-CHANGE. Same machinery as rl-F1/meth-M1: gates instantiated from the Phase-0 null; seed counts derived from the null, funded by the (re-costed) A40 budget.

**[llm-m1] Notes-carried restarts propagate confabulation.** RESOLVED-BY-CHANGE. Clean-restart vs. notes-carried is a pre-registered ablation arm in Phase 2; no variance-reduction credit is claimed for voting a priori (that criterion was deleted per meth-F2).

**[llm-m2] Local A40 ≠ Kaggle environment.** RESOLVED-BY-CHANGE. Two checks in §Measurement protocol: token-parity budgeting (tokens/s measured on both platforms; local per-game budget set in tokens, not minutes) and a cross-environment rank correlation of per-game levels (frozen build, local 3× vs. Kaggle draws) before local gating is trusted for submission selection. Answer to Q5 (BFS vs. duck segmentation): the adapter from duck's segmentation graph to our planner's state encoding is an explicit Phase-1 work item inside the 14 slots.

**[llm-Q6] Post-Sep-30 edge after forced open-sourcing.** ACKNOWLEDGED-AS-RISK, partially addressed. v2's October sketch names the composable assets (archive, measurement harness, TTT, protocol) plus holdout/procedural generality work; a full October plan is deliberately deferred until the Sep-30 wave's content is known — planning it in detail now would be planning against unknown code. We accept the residual risk that five weeks is short for a post-release edge.

---

## Program-Synthesis reviewer

**[ps-M1] "De-risked at build-quality level" is a category error.** RESOLVED-BY-CHANGE. Identical to llm-M1: Phase-0 pilot on the A40 with runtime caps; Phase-2 entry conditional. The 22/24 figure is relabeled in v2 Evidence as a build-methodology result, not a runtime-capability result.

**[ps-M2] `verify()` on replayed history is Goodhart-able (lookup-table convergence).** RESOLVED-BY-CHANGE. All three demanded mechanisms adopted: held-out transition split (refactor loop never sees it), scored degenerate baselines (identity-frame and pure-lookup must both be beaten), and a mechanical code-size/branch-count penalty in the acceptance rule (MDL enforced by the acceptor, not a prompt).

**[ps-M3] "State-exact" undefined at horizon; unlinked to planning utility.** RESOLVED-BY-CHANGE. Defined: exact-match on held-out 5-step open-loop rollouts. Phase 2 gates on plan-transfer rate ≥40% — the quantity that produces levels — with the existing 22 sims re-scored under both metrics in Phase 0 (answer to Q1/Q3: one-step replay historically; never plan-transfer; both now measured before Phase 2). Identity/lookup scoring under the new metric is part of that re-score.

**[ps-M4] BFS path contradicts zero-game-specific thesis; Phase-1 gains mis-attributed.** RESOLVED-BY-CHANGE. See rl-M2/llm-F1: claim retracted, dependency explicit, and Phase-1 deltas reported with and without the 14 clone-history games so the exploration substrate is credited only with what it earns. Answer to Q2: runtime-synthesized model or nothing.

**[ps-M5] Template leakage into generality claim unaudited.** RESOLVED-BY-CHANGE. The exact experiment demanded — leave-one-game-out scaffolding on the public set — is inside the Phase-0 pilot; a sharp drop reclassifies templates as content-carrying and they are stripped to grid utilities/skeletons. The leakage audit ships as a written artifact (answer to Q5: current templates contain mechanics abstractions; LOGO decides whether that survives; yes, we will publish the ablation).

**[ps-m1] Class-A vs. 50%-state-exact inconsistency; no levels→RHAE conversion.** RESOLVED-BY-CHANGE. One definition everywhere (Class-A ≡ the new held-out-rollout metric; answer to Q7: same threshold now). All gates are denominated in RHAE via eval_harness (which is RHAE-correct), removing the unconverted "levels" bookkeeping; the Phase-4 criterion is expressed directly in RHAE against the frozen-fork mean.

**[ps-m2] P2 kill conflates model ceiling with harness quality.** RESOLVED-BY-CHANGE. Three structurally distinct scaffold variants (fill-in-skeleton / free-form / diff-based refactor) are pre-registered; no "27B ceiling" verdict is permitted until all three fail. Answer to Q6 = rl-F1 answer: measured in Phase 0, gates derived from it.

---

## Methodology reviewer

**[meth-F1] LB noise "≥0.21" derived from n=2 and used as a constant.** RESOLVED-BY-CHANGE. Answer to Q6: literally two identical-code pairs (0.43/0.22 and 0.33/0.24 — the second was unreported in v1; still inadequate). v2 demotes the figure to point estimates and adds the demanded experiment: 4–6 submissions of the byte-identical frozen duck fork during Phases 1–2 (using filler slots v1 wasted on v35 redraws), yielding empirical σ with a CI before the LB is treated as either unusable or a known lottery.

**[meth-F2] Variance-reduction gate undecidable at n=3 (F(2,2)).** RESOLVED-BY-CHANGE. The "std reduced ≥30%" criterion is deleted as a gate. Voting is now gated on the paired per-game RHAE permutation test (~75 paired observations); variance is tracked descriptively only.

**[meth-M1] No power analysis; seed budget arbitrary.** RESOLVED-BY-CHANGE. The reviewer's prescription is adopted verbatim: 8–10-seed frozen-build null in Phase 0, gates = "delta > 90th percentile of identical-build null," paired per-game permutation tests instead of aggregate point thresholds, seed counts per gate derived from the null (funded — the re-costed A40 budget covers 10+ seeds at decision points; answer to Q2: no justification existed; fixed).

**[meth-M2] Sequential selection → compounding winner's curse.** RESOLVED-BY-CHANGE. Fresh-seed confirmation runs after every selection event, including the Sep freeze (both final configs re-run on seeds never used for selection; confirmation numbers are the reported numbers; shrinkage budgeted). Answer to Q3: we do not analytically correct — we re-measure on unseen seeds, which is the cleaner fix at this scale.

**[meth-M3] Adaptive overfitting to the 25 public games.** RESOLVED-BY-CHANGE. 18-dev/7-holdout split from day one; holdout touched only at phase boundaries (one sweep each); dev/holdout gap reported per phase as the overfitting canary. Residual ACKNOWLEDGED-AS-RISK: 7 games is a small holdout and the public set is heuristic-solvable (AERA), so the canary is noisy — procedural variants supplement it in October. Answer to Q5: no Preview-era local-vs-private correlation is computable from our data (we lacked this harness then); the holdout gap is the best available proxy and we say so.

**[meth-M4] "Best draw ≥1.6" measures luck; max-of-N doesn't transfer.** RESOLVED-BY-CHANGE. Success criterion is now the mean of Phase-4 draws with CI vs. frozen-fork mean + 2σ̂; best-draw is reported, not gated. Final-submission selection rule pre-registered (higher fresh-seed holdout worst-case per-game RHAE; same rule for Nov 2 — answer to Q4). Partial REBUTTAL retained: milestone rank is decided on the *public* LB, so multiple draws of a frozen build still raise expected public rank even though they say nothing about the system — v2 keeps the draws but stops scoring itself on them.

**[meth-m1] 1.7–2.0 extrapolation has no model.** RESOLVED-BY-CHANGE. Labeled a guess in v2 Evidence §2; no gate or freeze decision depends on it.

**[meth-m2] Regression clause is a multiplicity trap.** RESOLVED-BY-CHANGE. Replaced with per-game paired tests under Benjamini–Hochberg FDR at q=0.1, exactly as prescribed.

**[meth-Q7] False-kill probability of P2 at true rate 6/25.** RESOLVED-BY-CHANGE. Under the old rule, high — a fair point. The new 4–7 zone rule means a true 6/25 lands in verify-only-plus-retry rather than death, and the kill requires failure across all three scaffold variants, which bounds single-run false kills.

---

## Systems reviewer

**[sys-F1] Kaggle GPU quota arithmetic violated in Phases 0 and 4.** RESOLVED-BY-CHANGE. Conceded; the arithmetic did not close. v2: Phase 0 cut to 2 duck attempts (24 h < 30 h/wk); freeze moved to Sep 12–15 so 5–6 draws fit at the hard cap of 2 scored runs/wk; whether scored reruns draw from the interactive quota is verified empirically in Phase 0 (per option (a) in the objection) and the plan fits even in the worst case. The LB-sigma submissions are also scheduled at ≤2/wk inside the same cap.

**[sys-M1] A40 ≠ Kaggle operating point (wall-clock ratio, no FP8 on Ampere).** RESOLVED-BY-CHANGE. Phase 0 measures tokens/s on both platforms (A40 runs the FP8 checkpoint via weight-only dequant; pre-registered AWQ-int8 fallback if throughput is unusable); local per-game budgets set to token parity; sweeps re-costed at 20–40 A40-h (not v1's 8), budget revised to ≈$1,900 through Sep 30. The v1 "8 GPU-h/sweep" figure conflated our old CPU-stack harness timing with duck-eval; corrected. Ablation-matrix seed counts are sized after the parity measurement so the 40 h/wk constraint is checked with real numbers, not assumed ones.

**[sys-M2] 3 seeds can't support the gates; variance criterion is a coin flip.** RESOLVED-BY-CHANGE. Same resolution as meth-F2/meth-M1 (criterion deleted; null-calibrated permutation gates; seed counts derived in Phase 0).

**[sys-M3] "GPU idle between vLLM batches" is not how vLLM works.** RESOLVED-BY-CHANGE. The v1 sentence was wrong and is gone. Phase-3 entry includes the demanded one-day measurement: vLLM tokens/s at gpu_memory_utilization 0.90 vs. 0.75 and CNN train-step latency co-resident vs. solo; the measured throughput tax is entered into the 15%-of-budget kill rule ex ante (answer to Q5: utilization setting is an output of that measurement, not a guess).

**[sys-M4] Per-run wall-clock accounting missing (55 game-hours vs. 12 h).** RESOLVED-BY-CHANGE, with the answer already in evidence: duck runs 28 concurrent game threads against one vLLM server, so 25 × 2.2 h compresses into ~12 h wall and the system is token-bound (panel_research_winners.md §Compute profile — v1 failed to surface this; v2 Evidence §1 states it). Answer to Q6: ~28-way concurrency, single GPU, shared prefix cache. Phase 0 still extracts the *realized* per-game wall/tokens from our own runs and re-denominates all budget-relative criteria in (tokens, wall).

**[sys-m1] Spot preemption corrupts sweeps.** RESOLVED-BY-CHANGE. Per-game checkpoint/resume mandatory in eval_harness before the first sweep; instance type logged per sweep; preempted runs excluded from paired comparisons.

**[sys-m2] BFS in the 60 s tool loop needs numeric caps.** RESOLVED-BY-CHANGE. Phase-1 spec: 10 s wall / 2×10⁵ nodes per call, anytime best-partial return — consistent with feedback_arc_long_bfs_mcts (caps on *tool calls inside a turn*, distinct from that memory's warning against capping total search budget across a long solve; the anytime design preserves the latter).

---

## Summary of dispositions

| Review | FATAL | MAJOR | MINOR | Resolved | Rebutted | Acknowledged |
|---|---|---|---|---|---|---|
| rl-planning | 1 | 5 | 2 | 8 | 0 | 0 |
| llm-agents | 1 | 4 | 2 | 7 | 0 | 1 (Q6 October depth) |
| prog-synthesis | 0 | 5 | 2 | 7 | 0 | 0 |
| methodology | 2 | 4 | 2 | 8 | 1 partial (max-of-N public-rank value) | 1 (holdout size) |
| systems | 1 | 4 | 2 | 7 | 0 | 0 |

No objection is left unaddressed. The two acknowledgments are residual-risk statements, not deferrals: October planning depth (blocked on unknown Sep-30 releases) and holdout-size noise (bounded by the 25-game universe; supplemented by procedural variants).
