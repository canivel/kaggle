## Summary (2 sentences)
v2 is a genuine revision, not a cosmetic one: every load-bearing fix I demanded — Phase-0 Qwen pilot before slot commitment, held-out k-step verification with degenerate baselines and a mechanical MDL penalty, plan-transfer gating, retraction of the "14 games" claim, scheduled leave-one-game-out audit, unified Class-A definition, and pre-registered scaffold variants — appears with concrete mechanisms attached. What remains within my domain is a layer of under-specified acceptance-rule details (margin over baselines, held-out split construction, MDL coefficient, pilot ecological validity) that are individually small but sit exactly on the centerpiece's decision gate, so they must be pinned down in a metric-spec addendum before Phase 0 runs.

## Objections

### Resolution of prior objections

**[MAJOR-1] "De-risked at build-quality level" category error — RESOLVED (with residual, see NEW-1).** The Qwen-27B pilot is now Phase 0 (Jul 7–20), on the A40, under runtime-realistic caps, with Phase-2 entry explicitly conditional on it and 21 slots contingent rather than committed. The v1 evidence claim is honestly relabeled ("a build-methodology result, not a runtime-capability result"). This is exactly the fix requested.

**[MAJOR-2] `verify()` Goodhart-ability — RESOLVED (with residual, see NEW-2/NEW-3).** Held-out split, scored identity-frame and pure-lookup baselines, and an MDL penalty moved from prompt to acceptance rule. The mechanism class is right; the parameters are not yet specified (below).

**[MAJOR-3] Horizon-free metric, no planning-utility link — RESOLVED.** Class-A is now ≥50% exact-match on held-out 5-step open-loop rollouts; plan-transfer ≥40% is a co-equal Phase-2 gate; the 22 existing sims are scheduled for plan-transfer re-scoring in Phase 0. Both of my requested quantities (k-step rollout with k stated; plan-transfer as the gating quantity) are present.

**[MAJOR-4] BFS path contradiction — RESOLVED.** `bfs_solve()` is explicitly a generic planner over a supplied transition function; the "14 games" claim is retracted as private-set evidence; Phase-1 deltas are reported with and without the 14 clone-history games. This is the cleanest resolution in the document.

**[MAJOR-5] Template leakage unaudited — PARTIALLY-RESOLVED.** The leave-one-game-out scaffolding experiment is scheduled inside the Phase-0 pilot and a written artifact is promised. But no decision rule is attached: what LOGO degradation (e.g., Class-A rate drops by more than X games, or held-out rollout accuracy drops >Y points) triggers what consequence (template library demoted to grid-utilities-only? Phase-2 gate re-based on LOGO numbers?). An audit with no pre-registered threshold and no action is a report, not a gate. Fix: pre-register that Phase-2 entry is evaluated on the *LOGO-scaffolded* pilot numbers, since those are the only ones that model the private-set condition.

**[MINOR-1] Metric inconsistency, levels→RHAE unstated — RESOLVED.** One Class-A definition everywhere; all gates RHAE-denominated; ≥1.6 demoted to a reported aspiration; Phase-4 success defined against measured σ̂.

**[MINOR-2] P2 kill conflates model ceiling with harness quality — RESOLVED.** Three pre-registered, structurally distinct scaffold variants (fill-in-skeleton / free-form / diff-refactor) before any ceiling verdict.

### New objections

**[MAJOR] NEW-1: The pilot measures a different regime than the one it green-lights.** Three mismatches: (a) pilot data is *recorded* transition histories from prior (duck/opus) exploration on public games, but the runtime loop must synthesize from transitions its own — likely worse — exploration policy collects, interleaved with planning; (b) the A40 runs a dequantized-FP8 or AWQ-int8 Qwen, a numerically different model from the Kaggle FP8 checkpoint, and synthesis quality is known to be quantization-sensitive at the "writes correct branchy code" margin; (c) the pilot is open-loop batch, not the closed loop where bad early models poison the data distribution. A pilot pass could authorize 21 slots on unearned evidence. Fix: run ≥2 of the pilot games in closed-loop mode (agent collects its own transitions on the A40), and spend part of the 6 h/wk Kaggle smoke budget on one synthesis-battery run on the actual Kaggle SKU to bound the quantization gap.

**[MAJOR] NEW-2: The acceptance rule's baselines are weak under the very metric that was fixed, and no margin is specified.** Pure-lookup *trivially* fails held-out 5-step rollouts (held-out states are absent from the table), so "beats pure-lookup" is nearly vacuous; the dangerous degenerate is lookup-with-identity-fallback (or nearest-seen-state fallback), which scores well on sparse-change ARC-like games and is exactly what a 27B under refactor pressure converges to. Simultaneously, on sparse-change games the identity baseline itself may exceed 50% on 5-step rollouts, making the absolute ≥50% threshold and the identity baseline jointly satisfiable by a useless model unless the required margin is stated. Fix: add lookup-with-identity-fallback as a third scored baseline, and define Class-A as exact-match ≥ max(all baselines) + δ with δ pre-registered per the pilot's measured baseline distribution — not an absolute 50%.

**[MAJOR] NEW-3: "Held-out split" construction is unspecified, and the specification matters twice.** If the split is i.i.d. random over the transition buffer, rare reward-relevant transitions are underrepresented in *both* splits and the metric still hides exactly the errors that break planning; and in-kernel, "transitions the refactor loop never saw" is operationally ambiguous when the loop runs online over a growing buffer. Fix: pre-register (a) pilot split = temporal (train on earlier, verify on later) plus a stratum of changed-frame transitions reported separately, and (b) the in-kernel definition = prospective verification (the next N live transitions after acceptance), which is also the honest online analogue and composes with replanning-on-contradiction.

**[MINOR] NEW-4: MDL penalty parameters are absent and branch-count is evadable.** A dict-literal lookup table is one "branch" and short code; branch-count alone doesn't penalize it. Specify the penalty as total source length including data literals (e.g., gzip'd bytes or AST node count) with a pre-registered coefficient trading penalty units against held-out exact-match points, and report the train-vs-held-out accuracy gap as a memorization flag.

**[MINOR] NEW-5: Plan-transfer as defined omits goal-attainment.** "Plans that realize their predicted end state live" verifies dynamics fidelity along the plan but not that the plan's *purpose* (level progress) was achieved — a correct dynamics model with a wrong win-condition predicate passes end-state matching and yields zero levels. Report both: end-state match rate and goal-attainment rate; gate Phase 2 on the latter or on their conjunction.

**[MINOR] NEW-6: Pilot threshold arithmetic is unpinned.** "≥4 piloted games" over an unspecified 5–10 game pilot is a 40–80% bar depending on n, which materially changes false-kill and false-pass probabilities relative to the Phase-2 target of 8/25 (32%). Pre-register n now (I recommend 10, since the A40 is otherwise idle in Phase 0) and state the bar as a fraction consistent with the 8/25 target.

## Questions for the authors
1. What margin δ over the strongest degenerate baseline defines Class-A, and will lookup-with-identity-fallback be added as a scored baseline? (NEW-2)
2. How exactly is the held-out split constructed in (a) the pilot and (b) the in-kernel loop — random, temporal, or stratified by frame-change? (NEW-3)
3. What is the MDL penalty's unit and coefficient, and does "code size" include data literals? (NEW-4)
4. Will Phase-2 entry be evaluated on the LOGO-scaffolded pilot numbers or the full-template numbers? If full-template: why, given the private set has no matching templates? (MAJOR-5 residual)
5. What fraction of the pilot runs closed-loop (own-policy data collection), and how will the FP8-vs-dequant/AWQ quantization gap be bounded before a pilot pass authorizes 21 slots? (NEW-1)
6. Plan-transfer: is a plan counted as transferred if the predicted end state is realized but no level/score progress results? Give the two rates separately for the 22 re-scored sims.
7. Pilot n: 5 or 10 games? State it before Jul 7.

## What I cannot judge
Unchanged from round 1: Kaggle kernel/metadata/quota mechanics and whether "2 scored runs/wk + sigma experiment + Phase-4 draws" actually fits (I defer to the systems reviewer); the LB-noise measurement design's statistical power (methodology reviewer); the Phase-3 value-net training specifics beyond its entry gate, which as written (offline ρ ≥0.3, ≥30 events/game, pre-slot) is acceptable; competitive/game-theoretic timing questions; and the veracity of the cited research files and the 58.12% Rodionov / Rudakov figures, which I continue to take on faith.

## Verdict: MAJOR-REVISION

## Score: 7/10

The revision is close: all seven prior objections are RESOLVED or PARTIALLY-RESOLVED with real mechanisms, and the plan's risk ordering is now correct. The remaining MAJORs (NEW-1/2/3 plus the LOGO decision rule) all concern the *parameters* of the Phase-0 pilot and its acceptance rule — the single gate on which the entire Phase-2 thesis rests — and can be fixed with a one-page pre-registered metric-spec addendum before Jul 7. I would move to ACCEPT on receipt of that addendum without another full panel round.