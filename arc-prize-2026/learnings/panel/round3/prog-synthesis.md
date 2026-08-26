## Summary (2 sentences)
v3 delivers precisely the metric-spec addendum I conditioned acceptance on: every one of my remaining round-2 objections — LOGO as the deciding numbers, the lookup-with-identity-fallback baseline with a pre-registered +10 pp margin, temporal split plus changed-frame stratum, prospective in-kernel verification with a demotion rule, gzip-KB MDL including data literals with a stated coefficient, goal-attainment as a co-equal gate, closed-loop pilot arms plus an on-SKU quantization anchor, and pilot n = 10 — is now resolved with numbers attached. What remains in my domain are two small, genuinely new coherence issues in the runtime acceptance/demotion machinery, neither of which blocks Phase 0 starting Jul 7.

## Objections

### Resolution of prior objections

**[MAJOR-5 residual] LOGO audit had no decision rule — RESOLVED.** "LOGO numbers are the deciding numbers" is stated in Phase 0c and the Phase-2 entry gate is explicitly "≥4/10 pilot games Class-A on LOGO numbers." This is exactly the fix I specified: the private-set condition (no matching templates) is now what the gate models.

**[NEW-1] Pilot regime mismatch (recorded histories, quantization, open-loop) — RESOLVED.** All three mismatches are addressed: ≥2 closed-loop pilot games with entry requiring ≥1 closed-loop pass; a 3-game synthesis battery on the actual RTX PRO 6000 with a pre-registered ≤15 pp discrepancy bound and a kernel-anchored fallback if it trips; token allocation (30/40/5/25 of measured T_game) and tokens-to-first-Class-A as entry artifacts. The anchor-trip consequence (decide on kernel numbers) is the right failure mode rather than a mere report.

**[NEW-2] Weak baselines, no margin — RESOLVED.** Lookup-with-identity-fallback is a scored baseline; Class-A = exact-match ≥ max(all three baselines) + 10 pp; the jointly-satisfiable absolute-50% threshold is dropped. A fixed pre-registered δ = 10 pp is an acceptable substitute for my "δ per measured baseline distribution" — arguably cleaner, since it cannot be tuned post hoc.

**[NEW-3] Held-out split unspecified — RESOLVED.** Pilot: temporal 70/30 with the changed-frame stratum reported separately. In-kernel: prospective verification on the next 30 live transitions with a demotion rule. Both halves match my requested specification; prospective verification composes with replanning-on-contradiction as intended.

**[NEW-4] MDL parameters absent — RESOLVED.** λ·gzip-KB of full source including data literals, λ = 2 pp/KB, train-vs-held-out gap reported as a memorization flag, plus a consistent 4k-token post-refactor hard cap in the Phase-2 context table. The dict-literal evasion is closed.

**[NEW-5] Plan-transfer omits goal-attainment — RESOLVED.** End-state-match and goal-attainment are reported separately, by plan-length bucket, for the 22 re-scored sims, and Phase 2 gates on the conjunction (≥40% / ≥25%).

**[NEW-6] Pilot arithmetic unpinned — RESOLVED.** n = 10 pre-registered before Jul 7; bar 4/10 (40%) stated as consistent with the dev gate 6/18 (33%).

### New objections

**[MINOR] N1: The in-kernel demotion threshold is incoherent with the acceptance margin.** A model is accepted at max(baselines) + 10 pp but demoted only when prospective exact-match falls below *accepted score − 15 pp* — so a model can run live, with BFS planning through it, while performing up to 5 pp *worse than lookup-with-identity-fallback* and never trip demotion. Fix: demote when prospective exact-match < max(baselines evaluated on the same prospective window) + δ, or tighten the buffer to accepted − 10 pp so demotion cannot admit sub-baseline models. This is one line in the kernel and should be pinned before Phase 2, not Phase 0.

**[MINOR] N2: The +10 pp margin has no minimum-sample-size guard.** On a temporal 30% held-out of a possibly short per-game transition history, the number of 5-step rollout windows can be small (tens), where a 10 pp margin is within one binomial SE and Class-A becomes a coin flip per game. Pre-register a minimum held-out window count (e.g., ≥30 rollout windows, matching the prospective-verification n) below which Class-A is "undetermined" and the game doesn't count toward the 4/10 bar in either direction; report window counts per pilot game.

**[MINOR] N3: Baseline rollout semantics unstated.** For 5-step open-loop rollouts the two lookup baselines need a defined composition rule when an intermediate predicted state is unseen (pure-lookup must emit *something* at steps 2–5). Presumably pure-lookup aborts-as-mismatch and the fallback variant substitutes identity, but state it — the max-of-baselines margin moves several pp depending on the convention, and it must be identical in the pilot and any kernel-anchored re-scoring.

## Questions for the authors (numbered)
1. Will the demotion rule be restated relative to baselines on the prospective window (or tightened to accepted − 10 pp), per N1? A yes suffices; no re-review needed.
2. What is the minimum held-out rollout-window count for a Class-A verdict to be valid, and how many windows do the 10 pilot games actually have? (N2)
3. State the lookup baselines' rollout composition rule (abort-as-mismatch vs identity-substitution at unseen intermediate states). (N3)
4. For the ≥2 closed-loop pilot games: if scripted `explore()` collects too few changed-frame transitions to populate the 30% held-out, does that game fall to "undetermined" under the N2 rule or count as a synthesis failure? Pre-register which.

## What I cannot judge
Unchanged: Kaggle kernel/metadata/quota mechanics and whether the class-B ledger (17–18 ≤ 24) actually holds (systems reviewer); the sign-flip/FWER/MDE machinery's correctness and the σ̂ χ² CI design (methodology reviewer); Phase-3 value-net training beyond its offline entry gate, which remains acceptable as written; the RTX PRO 6000 SKU assertion and the 58.12% Rodionov / Rudakov figures, taken on faith pending the Phase-0 hour-1 log confirmation; competitive timing of the Sep-30 open-source wave.

## Verdict: ACCEPT

## Score: 9/10

All seven of my round-2 objections are RESOLVED with concrete, pre-registered mechanisms, and the centerpiece's decision gate (Phase-0 pilot → Phase-2 entry) is now falsifiable end-to-end: LOGO-scaffolded, closed-loop-inclusive, quantization-anchored, baseline-margined, temporally split, MDL-penalized, and goal-attainment-gated. The three new MINORs (demotion-threshold coherence, margin sample-size guard, baseline rollout semantics) are one-paragraph fixes that can be committed in the Phase-0 pre-registration document without another panel round; I recommend the chair collect written answers to my four questions as a condition of the accept, not as a revision cycle.