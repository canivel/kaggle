## Summary (2 sentences)

The revision makes real systems progress — the v5 canary produced hard evidence (dataset-mount weights found, 337 s boot-to-serve, tool-call and vision roundtrips green, and a 1500 s action-count slice) and a v4 prereg with named gates now exists — but v6 was fired before the ρ_action kill threshold was named (NC-5 open), before anyone did the trivially available arithmetic on the v5 slice, and the panel-unanimous readiness-gate A/B remains unscheduled for another day. The v5 slice itself, read at face value, projects an action count far below the prereg's own G2 floor, and nobody in the document says so.

## Objections

**Prior-round resolution status (all six, in order):**

**[MAJOR → PARTIALLY-RESOLVED] Boristown readiness-gate allocated zero action** — Upgraded from "zero mention" to open question 3 with R21 directive #1 at 5/5, a proposed prereg shape (n=3–5 gated vs frozen, one-sided at the 1.47-anchor effect size), and an explicit slot-competition framing. But it is *still unscheduled* one full round after unanimous panel endorsement, and the brief frames it as tied with a sentinel draw whose disposition memo itself says "no lift channel." Partially resolved; see new objection N1 for the ranking the panel is asked to make.

**[MAJOR → PARTIALLY-RESOLVED] A17 72B feasibility asserted, never computed** — v5 delivered real data: GPU identified (RTX PRO 6000 Blackwell), boot-to-serve 337 s, fenced-recovery live, and a 1500 s slice with raw N per game. That is genuine progress. But the arithmetic I demanded *before v6 fires* — tok/s, tokens/action, implied ρ_action — was never done; v6 fired anyway under user order, hiding behind "MEASUREMENT ONLY, no interpretation at k=1." See new objection N2: the k=1 slice, even with wide error bars, already brackets the answer, and the bracket is bad.

**[MAJOR → PARTIALLY-RESOLVED] No pre-registered PASS/FAIL thresholds** — The 07-26 v4 prereg now exists with numeric gates (G1 recovery ≥ 0.95, G2 ≥ 100 executed actions, G3/G4). That is the structure I asked for. However NC-5 — the numeric ρ_action kill threshold Y — is explicitly undischarged, and v6 is *already in flight* while Y is unnamed. The sealed-walk protection only works if Y is fixed before v6's numbers are readable; the window to do that is hours, not a Sunday panel cycle.

**[MAJOR → PARTIALLY-RESOLVED] Quota and slot economics absent** — The brief now states v6 is a free kernel build consuming no scored slot, bounds it to one session (~2.5 GPU-h), and notes "push slot 1 of 2." But there is still no weekly GPU-hour ledger covering filler cadence + canary reruns + retry slots + the proposed A/B arms, and the filler-draw marginal-value point is only half-conceded (directive #5 "filler holds rank, does not climb" — good) without converting that into a cadence decision.

**[MINOR → RESOLVED] Default trajectory cannot reach gold, unstated** — Now stated plainly and quantified: P(single frozen draw ≥ 1.49) ≈ 2×10⁻⁴, "filler holds rank, it does not climb." Resolved.

**[MINOR → RESOLVED] Weekday auto-fire lacks resource cap** — NC-3 addressed: session cap + stall-kill + zero-action-abort bound the burn to one ~2.5 GPU-h session, and A17_WINDOW_S=7920 + ~340 s boot ≈ 2.3 h is consistent with that bound. Resolved (conditional on the stall-kill actually having a numeric timeout — state it).

**New objections:**

**[MAJOR] N1: The v5 slice already forecasts a G2 failure and the brief refuses to say it** — The 1500 s slice shows Σ N = 5 executed actions across 4 games (2+1+0+2). Even crediting the full 345 s boot against the slice (≈1155 s active), that is ≈15.6 actions/hr, projecting ≈33 actions over the 7920 s window — a factor of ~3 below the prereg's own G2 ≥ 100 floor, and implying ρ_action ≈ 480/33 ≈ 14–15 if 480 is the frozen-fork denominator. "No interpretation at k=1" is a defensible rule for *capability* claims; it is not a license to ignore a 3× throughput shortfall that simple division exposes. Actionable: publish this projection now, alongside the confidence band a Poisson model on N=5 gives (roughly 2–10 events → 13–66 projected actions, upper bound still below 100), so the sealed walk reads v6 against a stated prior instead of feigning surprise.

**[MAJOR] N2: Hardware parity between the canary kernel and the scored-submission environment is unverified** — The canary drew an RTX PRO 6000 Blackwell, which is not the historical Kaggle scoring tier; if scored submissions rerun on a different GPU class (or a different count), every v5/v6 number is measured on the wrong hardware and the 337 s boot, tok/s, and ρ_action do not transfer. This is load-bearing for the entire A17 rail. Actionable: grep the frozen fork's *scored* run logs (the 14 filler draws) for the GPU name string and confirm it matches the canary's; if the scoring tier differs, v6's output must be re-labeled as an upper bound and re-benched on the scoring tier before any promotion arithmetic.

**[MAJOR] N3: Name Y now — here is a concrete proposal so "the panel should propose Y" cannot slip another round** — Since the brief asks: set the kill rule as **ρ_action ≥ 4.8 ⇒ 72B route dead** (equivalently, projected full-window Σ N₇₂B < 100, i.e., G2 itself), with a secondary gate that projected actions/hr at t=1500 s of v6 must be ≥ 45 to continue the session (ties into the zero-action-abort machinery). Rationale: the frozen fork scores 0.97 mean at the 480-action denominator; a 5× action deficit cannot plausibly be overcome by per-action quality at the 72B tier when the external capability anchor (Opus 5, unconstrained API) is 30.2% — there is no evidence any model buys back a 5× cadence loss in this harness. If the panel prefers a different Y, fine — but it must be committed in writing before v6's log is pulled.

**[MAJOR] N4: Rank the readiness-gate A/B strictly above sentinel draw #2 and schedule it this week** — On slot economics the comparison is not close: the readiness gate has a confirmed external anchor at 1.47 (above our 14-draw max of 1.33, near gold cutoff 1.49), a ~10-line serving-side diff, a clean mechanistic story (cold-start actions eaten during weight load / CUDA-graph compile), and a testable secondary prediction (band variance shrinks, since s=0.138 is plausibly cold-start jitter); the sentinel has n=1 at 0.71, p≈0.07, and a disposition memo that says "no lift channel." Actionable: pre-register n=4 gated draws replacing the next 4 fillers (cost: zero incremental slots, since fillers were running anyway), one-sided test of mean shift ≥ +0.25 (the 1.47-anchor-implied effect), decision by ~Aug 2. Option (b) of open question 2 is correct; take it.

**[MINOR] N5: v5's "PASS" is a boot verdict, and the brief's headline conflates it with route viability** — "VERDICT: PASS (dataset-weights route ALIVE)" is accurate for the mount/serve/parser risks it discharged, but the section header will be read by future rounds as "A17 is on track" when the same section's own slice data points the other way (N1). Retitle or add one sentence: "PASS covers boot/serve/tooling only; throughput verdict pending v6 against pre-committed Y."

## Questions for the authors (numbered)

1. What decode tok/s and tokens/action did the v5 canary actually observe in the 1500 s slice? The log clearly contains enough to compute inter-action intervals — why were these not reported?
2. Is the scored-submission rerun environment the same GPU class (RTX PRO 6000 Blackwell, same count) as the interactive kernel that ran v5? Cite a log line from a scored draw.
3. In ρ_action = 480/Σ N₇₂B, confirm 480 is the frozen fork's measured full-window executed-action count (mean? min?) — and on which draw(s) was it measured?
4. What is the numeric stall-kill timeout and the zero-action-abort trigger for v6 (seconds without an executed action)?
5. Does the 4-game sequential schedule in v5/v6 amortize one vLLM boot across all games, or re-boot per game? If per-game, ~340 s × 4 ≈ 23 min of the 7920 s window is boot — was that counted in the window budget?
6. What does the "push slot 1 of 2" budget cover for the rest of this week if v6 needs a retry (seed 2 or a fixed config)?

## What I cannot judge

The statistical machinery around the sentinel arm (t-predictive p≈0.07, MK/CUSUM no-trend, sequential stopping boundaries) beyond sanity level — that is reviewers #2/#3 territory. The governance question of whether the user-ordered mid-week v6 fire violated the 07-27 restructure's spirit. The EWM Stage-1 latent-state audit content (open question 4) — I can only price its compute-slot conflict with A17, not its merits. Game-specific scoring mechanics of ARC-AGI-3 (whether 33 actions can score at all on some games).

## Verdict: MAJOR-REVISION

## Score: 5/10