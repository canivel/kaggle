## Summary (2 sentences)

v2 is a substantially honest revision: the determinism audit, the retraction of the clone-coupled BFS claim, the held-out rollout + plan-transfer metric, and the offline TTT gate all land essentially as demanded, and the risk-ordering (27B pilot in week 1) is the right correction. What remains broken is the statistical spine's *power* side — the gates now control false merges (α≈0.1 per look, itself inflated by pre-registered retries) but nowhere commit to a minimum detectable effect at 3 treatment seeds — plus two search-substrate gaps the revision opened or left open: the RHAE cost of frontier returns in an episodic API, and a BFS node cap that yields depth ≤2 under any click-sized branching factor unless an unstated action abstraction exists.

## Objections

### Resolution of prior objections

**[FATAL, r1] 3-seed instrument underpowered — PARTIALLY-RESOLVED.** The 8–10-seed frozen-baseline null, paired permutation tests, BH-FDR regressions, fresh-seed confirmation, and "gates numerically instantiated from the null" address the false-positive half of my objection and kill the v1 multiplicity trap. But my fix explicitly demanded a power calculation and minimum-detectable-effect (MDE) per gate *with a rule for raising seed counts*, and v2 does not commit to either: treatment arms still run at 3 seeds (75 paired obs), and "instantiated from the null" fixes only the threshold, not the sensitivity. See new objection N1 for the residual, which is MAJOR.

**[MAJOR, r1] Determinism/Markov assumptions behind the archive — RESOLVED.** Phase 0(d) is the audit I specified (double prefix replay + hash-precedes-divergence scan), gating the archive per-game. A residual cost issue the audit does not touch is raised as N2; the aliasing-audit sampling weakness is N4 (minor).

**[MAJOR, r1] "BFS owns 14 games" clone-coupled — RESOLVED.** The claim is retracted, `bfs_solve()` is correctly recharacterized as a planner over a supplied transition function whose private-set value is fully conditional on Phase 2, and Phase-1 deltas are reported with/without the 14 clone-history games. The unanswered live branching-factor question resurfaces as N3.

**[MAJOR, r1] Replay accuracy is the wrong planning target — RESOLVED.** Class-A is now held-out 5-step open-loop rollout exact-match against scored degenerate baselines; plan-transfer ≥40% gates Phase 2; replanning-on-contradiction is the default control loop; the 22 sims are re-scored for plan-transfer in Phase 0. This is exactly the fix.

**[MAJOR, r1] TTT value net has no training signal — RESOLVED.** Offline entry gate (Spearman ρ≥0.3, ≥30 informative events/game on target sparse games) before any slot, else Phase 3 cancelled, plus co-residency tax measured ex ante. ρ≥0.3 is a permissive bar, but it is pre-registered and offline, which was the point.

**[MAJOR, r1] Additive compounding — RESOLVED.** Summed marginals removed; Phase-4 expectation from measured joint ablation effect; ≥1.6 demoted to aspiration; "still worth submitting" floor pre-registered.

**[MINOR, r1] Voting under RHAE — RESOLVED.** RHAE-denominated with no-restart and clean-vs-notes arms; variance-reduction de-gated.

**[MINOR, r1] P2 4–7 zone — RESOLVED.** Pre-registered: verify-only + contradiction signal, no BFS-through-model, scaffold retries, permanent kill below 4.

### New objections

**[MAJOR] N1: Gates control α but not β — no MDE commitment, and pre-registered retries inflate the stated α** — Two specific defects. (1) Exchangeability/power: the null must be constructed as *3-seed-vs-3-seed* paired deltas of the identical build (splits of the 8–10 baseline seeds), not from pooled 8–10-seed variance, or the 90th-percentile threshold is miscalibrated for the actual 3-seed test statistic; and Phase 0 must report the MDE at ~80% power for that statistic, with a pre-registered rule "if MDE exceeds the plausible component effect, raise treatment seeds to N" — at $8–16/sweep this is affordable and there is still no excuse. (2) The retry rules ("fails its permutation gate twice consecutively," "retry with remaining scaffolds") are multiple looks at α=0.10 each; two looks push the false-merge probability toward ~0.19 per component. Either drop the threshold per look (e.g., 95th percentile) or pre-register the family-wise budget across all looks in all phases. Without this, the instrument is honest about what it rejects and silent about what it can see — a gate that cannot detect a real +0.2 RHAE component will kill good ideas and pass its budget to noise.

**[MAJOR] N2: Frontier-return cost under RHAE is unbudgeted — the archive can pass determinism and still lose the gate structurally** — In a live episodic API with no reset-to-state, returning to a frontier means replaying an action prefix of length O(depth), and RHAE penalizes actions; a Go-Explore-style loop that revisits k frontiers at mean depth d spends k·d actions producing zero levels, exactly the failure mode an efficiency-weighted metric punishes. The determinism audit establishes returns are *possible*, not that they are *affordable*. **Fix:** state whether the API offers any level/episode reset primitive; compute expected return-cost per frontier visit from Phase-1 logs; make frontier selection cost-aware (novelty ÷ return-cost, not novelty alone); and report the archive ablation's action-count decomposition (probe vs. return vs. progress actions) alongside the RHAE delta.

**[MAJOR] N3: The BFS node cap and the click action space are arithmetically incompatible without a stated action abstraction** — With arbitrary clicks the primitive branching factor is O(10³–10⁴); at 2×10⁵ nodes, uniform BFS reaches depth <2, which makes `bfs_solve()` decorative. The plan mentions salience-tiered click prioritization for *exploration* but never specifies the abstract action set fed to the *planner* over synthesized models. **Fix:** pre-register the planner's action abstraction (e.g., one action per segmentation object + movement primitives), and have Phase 0(c) report realized branching factor, node counts, and plan depths on the pilot games — this is one afternoon of logging and it determines whether Phase 2's plan-transfer gate is even reachable. Relatedly, state realized plan lengths so the ≥40% plan-transfer threshold is interpreted at the depths BFS actually produces, not at the 5-step Class-A horizon.

**[MINOR] N4: The determinism audit's sampling is too thin to catch low-probability stochasticity** — One prefix replayed twice per game misses stochastic branches with p<~0.5 per trajectory and time/frame-count-dependent state. Use ≥3 prefixes × 3 replays at varied depths per game, and treat the aliasing scan as necessary-not-sufficient (it only sees visited data); log any post-audit contradiction as an automatic per-game archive disable at runtime.

**[MINOR] N5: Null/dev-set mismatch** — The 8–10-seed null is described on 25 games but Phase-1/2 gates are evaluated on dev-18; the permutation null and FDR must be computed on the identical 18-game support, or the threshold is again miscalibrated. One sentence fixes this; say it.

## Questions for the authors

1. What is the pre-registered rule if the Phase-0 variance decomposition shows the 3-seed MDE exceeds the plausible per-component effect — raise seeds to what N, at what cost, decided by whom?
2. How exactly is the identical-build null constructed — 3-vs-3 seed splits matching the test statistic, or pooled 8–10-seed variance? On 25 games or dev-18?
3. What is the total number of gate "looks" (including retries and scaffold variants) across Phases 1–3, and the implied family-wise false-merge probability at 90th-percentile thresholds?
4. Does the live API offer any reset/restart primitive short of full-episode restart, and what is the measured mean action cost of one frontier return on the deepest public games?
5. What abstract action set does `bfs_solve()` search over, and what branching factor / plan depth did the 22 offline sims exhibit when the plan-transfer re-scoring is run in Phase 0?
6. What are realized BFS plan lengths, and is the 40% plan-transfer threshold conditioned on plan length or aggregated over all plans?
7. For N4: how many prefixes/replays per game will the determinism audit actually run, and what is the runtime policy when a post-audit contradiction is observed?

## What I cannot judge

Kaggle kernel/quota engineering (rerun-quota semantics, metadata byte-matching, slug hygiene, the 30 h/wk arithmetic); leaderboard diffusion sociology; whether Qwen 3.6 27B's code synthesis can clear the pilot bar (I judge the pilot's *design*, which is now sound, not the LLM's ceiling); FP8-on-Ampere dequant and AWQ fallback feasibility; dollar budgeting; competition rules on milestone eligibility and the Sep 30 open-source dynamics; provenance of cited arXiv results, taken at face value from the panel research files.

## Verdict: MAJOR-REVISION

## Score: 7/10

Six of my eight round-1 objections are cleanly resolved, including the retraction I did not expect to get, and the revision's reflex — demote claims to measurements, gate entries offline — is the right one. But the plan's spine is still the instrument, and the instrument still has a stated false-positive rate and an *unstated* detection floor; combined with the unbudgeted frontier-return cost and a planner whose node cap is arithmetically vacuous without an action abstraction, three MAJORs stand between this and execution. All three are one-to-two-day fixes inside the existing Phase 0; fix them in v3 and I expect to accept.