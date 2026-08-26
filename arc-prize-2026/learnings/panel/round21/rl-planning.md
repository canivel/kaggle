## Summary (2 sentences)
The brief is a competently monitored status report on a stationary process, but as a *strategy* it allocates nearly all submission budget to a frozen arm whose characterized distribution (mean ≈0.97, s ≈0.14) has essentially zero probability of reaching the gold cutoff of 1.49, while the two known positive-EV moves — the boristown readiness-gate diff and the exploration arm — are respectively unscheduled and shelved on n=1. Operationally clean, decision-theoretically stalled: it monitors well and plans almost not at all.

## Objections

[MAJOR] Filler-dominant allocation has ~zero win probability and near-zero information value — Under the brief's own numbers (n=13, mean 0.974, s 0.143), a frozen-fork draw hits ≥1.49 at z ≈ 3.6, i.e. P ≈ 2×10⁻⁴ per draw; even matching our own 1.33 best is a ~2.5σ event. Simultaneously, the marginal information from draw #14 of an already MK/CUSUM-characterized stationary band is negligible — you are paying full submission slots for neither score nor information. A daily filler cadence needs an explicit value-of-information argument or a reduced monitoring cadence (e.g., 2/week) with freed slots reallocated to interventions. The brief contains no such argument.

[MAJOR] The single cheapest known intervention — boristown's vLLM readiness gate — is not scheduled — The ground truth states the *only real functional diff* between our artifact (band 0.82–1.33, max 1.33) and boristown's 1.47 is a readiness gate, with a fork-diff document already written (07-24). If eval-time model readiness is truncating early-episode actions, that is a mechanistic explanation for our entire band deficit, testable with a one-diff A/B draw. That this ~one-line, documented, externally-anchored intervention does not appear anywhere in the A17 pin, the panel questions, or the draw schedule is the largest planning omission in the document. Either present the ablation result or explain why it was rejected.

[MAJOR] Exploration arm shelved on n=1 with a statistically indefensible harm-pause — The sentinel arm was HARM-PAUSED after a single draw of 0.71 against a 0.80 threshold, when the frozen arm itself has s ≈ 0.15 and a realized minimum of 0.82. One draw at 0.71 is ~0.6σ below the frozen arm's own floor — this is indistinguishable from noise, and pausing on it is exactly the premature-abandonment failure mode any bandit analysis warns against. With ~98 draw days to Nov 2 and a +0.5 mean gap to gold, an exploration cadence of "n=1, shelved, open question #4 unanswered" guarantees the gap is closed only if A17 works. State a re-open criterion (e.g., pause only if mean of first 3 sentinel draws < 0.80) or justify the shelving with more than one sample.

[MAJOR] A17 has a deadline (Aug 3) but no pre-registered numeric kill/success criterion in this brief — The brief references "prereg gates G1–G4" and a "ρ_action denominator" but gives no threshold: what actions/sec or ρ_action value must the 72B-AWQ canary clear for A17 to survive, and what number kills it? A 72B VL model under Kaggle time limits is exactly the regime where throughput collapse silently converts a "capability" bet into fewer episode steps and a *lower* score; the unresolved Qwen2.5-VL tool-call format defect compounds this. Since A17 is designated the "single highest-priority build item" and the only live gap-closing bet, the panel cannot evaluate the plan without the gate numbers inline. Restate G1–G4 with thresholds in the brief itself.

[MINOR] Ledger provenance gap: n=13 arithmetic rests on an unrefreshed canonical file — The canonical ledger (refreshed 07-25) shows n=11; the brief's n=13 requires a 0.84 draw on 07-26 sourced only from "the 07-26 brief," not the live-API refresh path the team itself declares the sole edit authority. The arithmetic checks out (10.802 + 0.84 + 1.02 = 12.662; /13 = 0.974 ✓), but "validated interpretation" should not cite draws absent from the canonical source. Run the refresh before Sunday briefs, or mark unrefreshed draws as provisional.

[MINOR] Open questions #3 and #4 are abdications, not proposals — A strategy document should present a recommended answer with a decision rule ("war-v4 spec starts now iff it consumes zero GPU-h and no draw slots"; "sentinel draw #2 fires on date X unless condition Y"), then invite objection. Posing them open-ended to a Sundays-only panel bakes in a ≥7-day decision latency on the two questions most coupled to the remaining calendar.

## Questions for the authors (numbered)
1. What is your explicit probability estimate, per arm (frozen, A17, war-v4, sentinel), of producing a ≥1.49 draw before Nov 2, and what draw-budget allocation follows from those estimates?
2. Why has the boristown readiness-gate diff not been A/B'd, given the fork-diff doc identifies it as the only functional difference behind a 1.47 anchor? What blocks a one-diff submission this week?
3. What exact numeric thresholds define A17 canary PASS (v5) and gates G1–G4 (v6), and what ρ_action value kills the 72B line?
4. What is the re-open criterion for the sentinel arm, and what n was the 0.80 harm-pause threshold calibrated against?
5. Was the 07-26 draw (0.84) API-verified through the canonical refresh path, and if so why is the ledger file two days stale on a "validated" brief?
6. Under the Sundays-only restructure, what specific conditions (enumerate them) trigger weekday panel escalation for an A17 promotion decision landing mid-week?

## What I cannot judge
Kaggle infrastructure specifics (dataset-weights vs. model-mount routes, notebook GPU-h accounting); the internals of the Qwen2.5-VL tool-call defect and the fenced-recovery adapter; the KAOS dream/skill-promotion machinery and fingerprint taxonomy; the accuracy of the discussions/research sweep coverage. I take the API-verified ledger numbers as given per panel instructions.

## Verdict: MAJOR-REVISION

## Score: 4/10