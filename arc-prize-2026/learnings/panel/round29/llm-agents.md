## Summary (2 sentences)

This brief is unusually disciplined about evidence classes and correctly refuses to promote on n=1 draws, but its headline deliverable — the `bench` A/B as "the first free comparator of the campaign" — is oversold: the placebo-control claim rests on an unverified third-party assertion that all 13 grafts are text-only, the design has a hard statistical floor (permutation p ≥ 1/6 at 2v2) that makes even the promised "kill" verdict unlicensable, and the arm it would measure is not the config that drew 1.62. The cheapest, most informative action available tonight (reading the 1.62 run's own graft banner from the existing kernel log) is deferred while a 9-hour instrument run with confounded arms is promoted to the top of the handoff.

## Objections

**[MAJOR] The "placebo control" claim is load-bearing and unverified against your own flag list.** The clean-control argument depends entirely on tennant's V-doc sentence that *"every graft in the stack is INFORMATIONAL: none of them chooses an action."* But your own cell-12 dump ships `retry_guard`, `shortcircuit`, `undo`, `carryover`, and `untried` — names that on their face describe control-flow and state interventions, not prompt text. If even one of the 13 grafts alters action selection, retry behavior, or episode state, then arm B (suppress text only) is not a placebo and the A/B difference is uninterpretable. Actionable fix: before running bench, audit each of the 13 flags in the bundle source and confirm its effect surface is prompt-text-only; additionally diff the action traces of one A and one B replica — identical action sequences under different text would itself be diagnostic.

**[MAJOR] The kill instrument cannot kill at n=2/arm — quantify the gate or don't run it.** With 4 observations split 2v2, an exhaustive permutation test has C(4,2)=6 arrangements, so the minimum achievable p-value is ~0.167 — no outcome, however extreme, reaches conventional significance. Against a documented within-config spread of the 1.82-vs-0.00 class on byte-identical code, "kill only on a large gap" is not a decision rule, it's a vibe. The proposal pre-registers a *direction* ("this can only license stop") but no *threshold*. Actionable fix: pre-register a numeric rule before the run (e.g., "kill iff max(A) < min(B) AND the gap exceeds X, where X is derived from the 1.82/0.00 spread"), and state honestly that even that rule has p≈1/6 under the null — i.e., this is a smoke test, and the brief should stop calling it an instrument that can "settle" anything.

**[MAJOR] Instrument-audit ordering is inverted: the zero-cost log read is deferred while the 9-hour run is promoted.** Gap 5 admits the 13-grafts-live claim is [UNK] on the actual run because `install()` is blanket-guarded and silently falls back to stock. If the 1.62 draw was actually a stock run, both the ledger correction (Decision 3, which would enter "13 grafts" — a *different* wrong row) and the bench comparison lose their referent. The `TAAF_GRAFTS FEATURES={...}` banner is already sitting in the completed kernel's log and costs nothing to read. Actionable fix: pull the banner *first*, tonight, before the ledger edit and before bench; make Decision 3 conditional on it.

**[MAJOR] Stateful grafts plus fixed replica ordering confound arm with position.** Four replicas of one game in a single session, in the fixed order A0/B0/A1/B1, on a harness that includes `carryover`, `undo`, and `untried` (cross-game state by name) means arm A's replicas can leak state into subsequent replicas, and arm is confounded with session position (warm-up, cache, cumulative token/wall-clock depletion — the brief's own find #7 says the wall-clock cap binds everything). Reading the arm label from the artifact path prevents *label* drift; it does nothing about *order* and *state* confounds. Actionable fix: verify `make_replicas` gives each replica fresh graft state, and counterbalance or randomize replica order; if the harness cannot do either, say so in the pre-registration and downgrade the read accordingly.

**[MAJOR] Even a clean bench verdict answers the wrong comparison, on the wrong code.** Arm B reproduces "the v12 floor's prompt" inside the TV28 harness — that is not your certified field-floor config (mean 1.5413), so "A ≤ B on m0r0" licenses "grafts don't help within TV28," not "the TV28 arm loses to our floor," which is the actual portfolio decision. Worse, bench necessarily runs against the 08-29 republished bundle (unpinned slug, `composite.py` doubled to 24,798 B), so arm A is not even the agent that drew 1.62. Actionable fix: pin the bundle version *before* the bench run (not just before a redraw, as Handoff #2 has it), record which version bench measured, and rewrite find #1's claim from "whether to stop the TV28 arm" to the narrower question it can address.

**[MINOR] "Zero draws" is not "zero cost."** One ~9 h GPU session is a large fraction of a Kaggle weekly GPU quota that also gates every other offline experiment; the handoff prices it in wall-clock but not in quota-opportunity terms. State the quota budget and what this run displaces.

**[MINOR] Internal tension in the selection-rule logic.** The brief argues 2.05 "does not matter" because final selection is by config mean (floor mean 1.5413), yet the TV28 fork's n=1 mean is 1.62 > 1.5413. Under the stated rule as written, the unvalidated fork currently *outranks* the certified floor. Either the rule has a minimum-n requirement (state it) or the "do not redraw" guidance conflicts with the rule the campaign says it selects by.

## Questions for the authors

1. For each of the 13 flags, what is the code-level evidence that its effect surface is confined to prompt text? Point to the function each flag patches.
2. What is the exact numeric kill gate for the 2v2, written down before the run, and what do you do with a result at permutation p = 1/6?
3. Does `make_replicas` reset carryover/undo/untried state per replica, and is arm order randomized or counterbalanced across the four replicas?
4. Why is the 1.62 kernel-log banner read scheduled *after* the ledger correction it would validate?
5. Which dataset version will the bench session mount, and will that same version be pinned for any future TV28 redraw so the config identity matches what bench measured?
6. What fraction of your weekly GPU quota does the 9 h bench session consume, and what offline work does it displace?
7. Does `project_arc_final_selection_rule` impose a minimum n per config? If not, how is the n=1 fork mean of 1.62 handled at selection time?

## What I cannot judge

The correctness of the leaderboard archival pipeline (row counts, heartbeat/sha discipline, diff arithmetic); the statistical modeling of other teams' movement and the ladder hypothesis (statistician's domain); ARC-AGI-3 game internals beyond what the harness artifacts state; whether the 0.93 gap to #10 is closable by *any* strategy in the remaining time — I can only say this brief's actions do not move toward it, which the authors partially concede.

## Verdict: MAJOR-REVISION

## Score: 5/10