## Summary (2 sentences)

v3 delivers every fix my round-2 review demanded, in several places more carefully than I demanded it: the sign-flip test with game as the exchangeable unit and the retraction of the 75-observation pseudoreplication, the 8-look enumeration at α = 0.0125, the published MDE with a formula-executed seed-escalation rule, the cost-aware frontier score with a stated RESET-only reset primitive and the probe/return/progress decomposition, and a pre-registered planner abstraction whose arithmetic (21⁴ ≈ 1.94×10⁵ ≈ the node cap) actually closes. What remains are three genuinely minor gaps — a seed-escalation rule blind to the between-game variance floor, an unchecked completeness assumption in the ≤16-object salience abstraction, and possible screening-seed contamination of gate arms — none of which invalidates the plan and all of which fit inside the existing Phase 0.

## Objections

### Resolution of prior objections

**[MAJOR, r2] N1: Gates control α but not β; retries inflate α — RESOLVED.** Both defects are fixed and the fix to defect (1) is better than my prescription: the 3-vs-3 split-null is correctly demoted to a dependence-acknowledged calibration check (280 splits of 8 seeds are not independent draws, and the proposal says so), with the exact sign-flip test primary; MDE at 80% power is published from the Phase-0 variance decomposition; the seed-raising rule is formula-executed with a cost cap and the "underpowered kill" label removes the silent-detection-floor problem. Defect (2) is fixed by enumerating all 8 looks at per-look α = 0.0125, family-wise ≤ 0.10, with hyperparameter screening correctly collapsed to one look. Residuals are N6 and N8 below, both MINOR.

**[MAJOR, r2] N2: Frontier-return cost under RHAE unbudgeted — RESOLVED.** The reset primitive is now stated (full-episode RESET only, scorecard-counted — i.e., returns cost a full prefix replay of length d and RESET itself is penalized), frontier selection is novelty/(1 + return_cost) as demanded, and the archive ablation reports the probe/return/progress action decomposition beside its RHAE delta. Whether the economics leave the archive any value is now an empirical question the gate is equipped to answer — which is all I asked.

**[MAJOR, r2] N3: Node cap incompatible with click action space absent an abstraction — RESOLVED.** The abstraction is pre-registered (≤5 movement/rotate primitives + ACTION6 on ≤16 salience-tier objects, b ≤ 21), and the arithmetic is honest: 21⁴ ≈ 194k ≈ the 2×10⁵ cap, so "depth ~4+, deeper with dedup" is a defensible claim rather than a decorative one. Phase-0 telemetry (realized branching, nodes, depth distribution) and depth-bucketed plan-transfer with end-state-match and goal-attainment reported separately are exactly the fix. The completeness of that abstraction is a residual, N7.

**[MINOR, r2] N4: Determinism audit sampling too thin — RESOLVED.** 3 prefixes × 3 replays at varied depths; aliasing scan necessary-not-sufficient; runtime contradiction auto-disables the archive per game. Verbatim.

**[MINOR, r2] N5: Null/dev-set mismatch — RESOLVED.** Null on dev-18, the gates' support; 25-game numbers reporting only. The one sentence I asked for.

### New objections

**[MINOR] N6: The seed-escalation rule cannot buy back between-game variance, and the plausible-effect bands it keys on are unevidenced** — Raising seeds 3→5→8 shrinks only the within-game seed-noise component of Var(mean per-game delta); the between-game heterogeneity component σ²_game/18 is invariant to seed count, so the MDE has a floor at N = ∞ that no $130 sweep can breach. **Fix:** have the Phase-0 variance decomposition publish this floor explicitly, and short-circuit the escalation (skip 5 and 8, go straight to the underpowered-kill regime) whenever floor > band lower bound — this saves money and prevents the formula from performatively burning sweeps. Separately, the bands themselves (substrate +0.10–0.25, world model +0.15–0.40, etc.) drive a "no discretion" rule but are stated without provenance; anchor them to something (Rudakov's exploration delta, duck's measured level-transition losses) or label them explicitly as priors so the underpowered-kill label is interpretable.

**[MINOR] N7: The ≤16-object salience truncation is an untested completeness assumption that can make the Phase-2 gate misattribute planner failure to the world model** — BFS is complete over the abstract action set, but if the goal-relevant click target falls below the salience-16 cut, goal-attainment fails with a perfect Class-A model, and the conjunction gate (≥40% / ≥25%) kills or zones synthesis for a planner defect. **Fix (one afternoon, inside Phase 0c):** for each pilot game with recorded histories, compute the fraction of known winning action sequences expressible in the pre-registered abstraction; if expressibility < ~90% on solved games, expand the tier or add a coverage-triggered fallback before the Phase-2 gate is interpreted. The separate end-state-match vs goal-attainment reporting will partially reveal this failure mode post hoc, but the check should run ex ante.

**[MINOR] N8: Possible screening-seed contamination of gate arms** — Hyperparameter screening at 1 seed is fine as one look, but if that screening seed is reused as one of the 3 gate seeds for the selected configuration, selection optimism leaks into the gate statistic and the sign-flip test's calibration on that arm is compromised. **Fix:** one sentence — gate seeds are drawn disjoint from all screening seeds. Cheap and closes the last selection-contamination path in an otherwise clean instrument.

## Questions for the authors

1. Will the Phase-0 variance decomposition report the seed-invariant (between-game) MDE floor at N = ∞, and does the escalation formula short-circuit when that floor exceeds a component's band lower bound?
2. What evidence, if any, sets the lower bounds of the pre-registered plausible-effect bands — or are they priors, in which case will they be labeled as such wherever the underpowered-kill label is invoked?
3. Are the 1-seed screening seeds excluded from the 3-seed gate arms of the selected configuration?
4. On the pilot games with recorded histories, what fraction of known winning action sequences is expressible in the ≤5-primitive + top-16-object abstraction, and what is the pre-registered response if that fraction is low?
5. For the depth-bucketed plan-transfer gate: if realized plans concentrate in the 1–3 bucket, is the ≥40%/≥25% conjunction evaluated per bucket, or aggregated — and is a gate pass on 1–3-step plans alone considered sufficient for Phase-2 merge?
6. In the frontier decomposition, what mean return-cost (actions per frontier visit) do you expect to tolerate before the archive is judged structurally unaffordable, or is that left entirely to the RHAE gate?

## What I cannot judge

Kaggle kernel/quota engineering (rerun-quota semantics, slot ledger arithmetic, byte-matched metadata, the 24-slot cap); whether Qwen 3.6 27B can clear the pilot bar — I judge the pilot's design, which is sound, not the model's ceiling; FP8-Blackwell vs A40 dequant/AWQ numerics and the ≤15 pp anchor's realism; dollar budgeting and RunPod spot economics; competition-rules questions (milestone eligibility, Sep-30 open-source dynamics, whether vault-5 procedural variants are a faithful proxy for the private set's distribution — I can judge that it is *untouched*, not that it is *representative*); provenance of cited arXiv results, taken at face value from the panel research files.

## Verdict: ACCEPT

## Score: 8/10

All three of my round-2 MAJORs are fixed as specified — in the null-construction case, more carefully than specified — and the revision's consistent reflex across three rounds (retract claims, demote assertions to measurements, pre-register the response to every measurable outcome) is what a competition plan under uncertainty should look like. The three new MINORs (variance-floor short-circuit, abstraction expressibility check, seed disjointness) are each a sentence-to-an-afternoon inside the existing Phase 0 and should be folded in before Jul 7 execution; none warrants holding the plan. Residual risk is now where it belongs: in the 27B pilot's empirical outcome, not in the instrument that will measure it.