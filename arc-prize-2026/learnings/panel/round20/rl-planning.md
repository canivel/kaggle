## Summary (2 sentences)

The revision finally closes the fatal LB-provenance hole (account named, API-verified draw-by-draw ledger, stale-template root-caused), delivers genuinely good forensics on the canary v3 zero-action failure, and shelves the sentinel per panel recommendation — real progress. However, the governance machinery that is supposed to ratify everything has now failed three times in the same wedge class (R20 never ran, amendment still DRAFT, queue found EMPTY), the promotion gate remains mathematically unreachable before any arm can enter window 2+, and the boristown +0.14 filler-replacement — the highest-EV move on the board — is deferred for a third consecutive round on a false coupling to A17.

## Objections

**PRIOR-OBJECTION RESOLUTION STATUS (all verified before new comments):**

**[RESOLVED] Prior FATAL: LB ground truth unverified.** Fully closed and closed correctly: account identified (canivel), verification command published, draw-by-draw API-verified ledger attached, and — critically — the discrepancy was root-caused (stale hardcoded May-era template in `panel_round.py`) rather than papered over, with a reconciliation explaining both numbers (0.43 = forge-era May best; 1.33 = duck-fork era). This is exactly the artifact I demanded. The n=12 frozen stratum in today's brief chains cleanly onto the ledger's n=11. Pricing audits can resume.

**[RESOLVED] Prior MINOR: W2 confirmatory null zero-VOI.** Sentinel SHELVED by disposition memo, logged as closed on eval evidence with the scored draw consistent — verbatim my recommended disposition.

**[PARTIALLY-RESOLVED] Prior MAJOR: tail-model inconsistency.** A declared t-predictive now exists (today's 0.84 draw scored at t ≈ −0.9 under it) — that replaces the naked Gaussian for in-band classification, which is half of what I asked. But it is (a) trapped in the unratified DRAFT amendment, and (b) still not the demanded artifact: no exceedance/tail fit from which both "filler-only is losing" and "exploration is nearly free" prices derive. The empirical anchor now exists for free — 0 of 12 frozen draws exceed 1.33 while byte-identical public forks drew 1.39/1.47, so the artifact tail is real and our sampling of it is not; write the two-line exceedance model down.

**[UNRESOLVED] Prior MAJOR: promotion gate unreachable.** Nothing in this brief touches the promotion side; it is presumably inside DRAFT §(a)–(i) but I cannot verify text I have not seen. Mitigating fact: window 2 is riding filler, so no window is currently being spent against the unreachable gate — but war-v4 "waits on A17," and the moment it enters, this becomes load-bearing again. Must be fixed (exceedance-denominated promotion, not +0.06 mean-lift) before any arm clears entry bar §(c).

**[PARTIALLY-RESOLVED] Prior MAJOR: entry bar admits negative-VOI arms.** Operationally resolved for this window: Q5 states no arm clears entry bar §(c) and filler rides, which is the correct behavior. But §(c)'s text is unratified and unquoted, so I cannot confirm it requires *positive right-tail evidence* rather than mere non-harm. Show the bar's wording before window 3.

**[PARTIALLY-RESOLVED] Prior MAJOR: asymmetric stopping rule.** The sealed §9.1 gate boolean, the 3.5x envelope NO-GO penalty, and the C3 "no capability claim either way" discipline on v3 are all the right hygiene, and I credit that the v3 zero-score was NOT spun in either direction. But the exact GO/NO-GO/CONTINUE numeric boundaries and false-GO probability for the *scored* capability read remain unpublished (or sealed-and-unratified — same thing from where I sit). One paragraph, pre-v4-push.

**[PARTIALLY-RESOLVED] Prior MAJOR: boristown fork prioritization.** Promoted from Q5(iii) to a named amendment section §(i) "monitored-continuation filler replacement" with a scheduling question (Q4) — progress on paper, zero progress on the board. See escalation below.

**[UNRESOLVED] Prior MAJOR: §6 falsification disjunction text.** Still one unwritten sentence, now hostage to the R20 wedge. Carried.

**[UNRESOLVED] Prior MINOR: seal-termination downgrade logging.** Third round unmentioned. Carried.

**[UNRESOLVED] Prior MINOR: build-rail mutation accounting (general table).** The v4 description ("harness-side only, boot asserts unchanged") is per-push mutation awareness again, which is good, but the general rail-mutation column is still missing. Carried.

**[UNRESOLVED] Prior MINOR: A21 allocation policy.** Q5 again asks the panel to confirm pacing ad hoc. Mitigated by the fact that no arm clears the bar, so no allocation decision is actually live. Carried.

**[UNRESOLVED] Prior MINOR: Nov-2 target denomination.** All pricing still against today's wall (gold ≈1.49, "+0.16 above us") while the cutoff demonstrably compresses upward (1.44→1.47→1.49 in days). Carried; still a one-cell spreadsheet.

**NEW OBJECTIONS:**

**[MAJOR] The decision system itself is now the binding constraint — three panel-session failures in one wedge class, and a scored-window queue found EMPTY.** R20 died after prompt-write (same class as 07-21/07-22), leaving the amendment DRAFT for a third day, and the submission queue was empty at session start — an empty queue at the wrong hour forfeits a scored window outright, which at ~99 remaining windows and a fitted per-window E[max] contribution the team itself prices as nonzero is direct EV destruction with no offsetting information gain. A system that cannot reliably run its own ratification loop cannot claim its rules are pre-registered in any meaningful sense. Fix demanded: (a) a watchdog that detects a dead panel session and auto-relaunches within N hours; (b) a standing default-queue-head rule ("frozen-fork filler auto-arms at T−6h if queue empty") so an empty queue is structurally impossible; (c) a rule for what happens to DRAFT amendments after k failed ratification attempts (e.g., provisional adoption of the non-controversial subset). File the wedge class in the fingerprint store — it recurs and is absent from the incident table.

**[MAJOR] Third-round deferral of the boristown filler replacement is now the costliest standing decision, and the "hold for A17 outcome" coupling in Q4 is false.** The fork is a filler-*replacement*, not an arm: it consumes no exploration window, mutates no A17 rail state (submission lane vs eval lane are disjoint by the team's own architecture), and its adoption decision shares no information with the v4 canary outcome — so there is nothing to "hold for." Meanwhile the cost of delay is compounding: rank eroded 1.33→~#50+ while a diffed, monitored +0.14 floor-raise sat in a question queue, and P(any old-filler draw > 1.47) ≈ 0 under their own predictive (z ≈ 3.5 — every filler window between now and adoption is a window drawn from a dominated distribution). My panel answer to Q4: **schedule now**, with the previously-stated condition — control re-baseline (n ≥ 5 fresh draws) before harm-pause thresholds re-arm on the new floor. The only legitimate reason to hold is if §(i)'s "monitored-continuation" text is itself unratified — which loops back to the governance objection, not to A17.

**[MINOR] The 99.5% recovery-rate and 1.1x cadence figures are open-loop measurements on degenerate off-policy traffic; do not let them harden into on-policy expectations.** Every one of the 436 replayed turns was generated in games where `step_executed: False` throughout — the model never saw an advancing frame, a feedback loop, or the longer contexts of live play. Format-adherence rates measured on stuck-state transcripts need not transfer to closed-loop operation (this is the classic open-loop-replay ≠ closed-loop-behavior trap). The fix is cheap and procedural: pre-register the on-node v4 pass criterion *now* (e.g., "fenced-recovery + native hits ≥ 95% of turns AND step_executed on ≥ 90% of turns, else v4 FAIL → v5 with layer (ii)"), and make the banner disambiguate hermes-native vs fenced-recovered counts so that bundling (iv) with (i) doesn't destroy causal attribution if v4 passes.

**[MINOR] v4 composition ruling (answering Q3 concretely): authorize (i)+(iv)+(v), with (v) promoted to a loud-fail boot assert, and pre-commit the v5 escalation path to (ii) with the xgrammar/ACTION6 schema validated locally *in parallel today* rather than serially after a v4 failure.** The author's layering logic is sound — (i) is the only layer validated against own traffic and doesn't perturb what the model emits — but serial canary iteration has already cost three pushes (v1→v3), and pre-staging (ii)'s local schema validation costs $0 and collapses the v5 turnaround by a day if v4's on-node recovery disappoints. On the ρ_action sub-question: turn≈action is acceptable at canary stage *only* because the envelope margin is huge (1.1x vs 3.5x NO-GO); the exact-parity requirement at promotion must be written into the promotion gate text, not left as a caveat "on record."

## Questions for the authors (numbered)

1. Quote the exact text of entry bar §(c) and the promotion rule from DRAFT amendment §(a)–(i). Does the promotion side remain denominated in mean-lift, and if so, at what n does it become reachable before Nov 2?
2. What is the pre-registered on-node pass/fail criterion for canary v4 (recovery rate, step_executed rate, score band), written *before* the push? If none exists, why is the push authorized-pending rather than blocked-pending?
3. What concretely blocks scheduling the boristown §(i) filler replacement tonight, other than R20 ratification? Name the dependency on A17 explicitly or drop it from Q4.
4. What is the recurrence mechanism for the 07-21/07-22/07-25 session-death wedge, and why is this class absent from the fingerprint table when it has n=3?
5. Of the 2 unrecovered turns in the 434/436 replay, what did they contain — and are they a distinct failure mode that survives the adapter (i.e., a floor on v4's on-node recovery rate)?
6. Post-fork adoption at a ~1.47 floor: what are the re-based harm-pause threshold and re-baseline n, pre-registered now rather than improvised after the first low draw?

## What I cannot judge

vLLM/xgrammar internals (whether the cited FSM-failure and streaming-bug issues #16321/#31871 are correctly characterized), the AWQ chat-template-strip history, GPU serve-health telemetry (34.3 tps, shard-identity arguments for the /1→/2 pin), and Kaggle sandbox quantization specifics — these belong to the systems reviewer. I also cannot independently verify file-referenced artifacts (`runs/a17_recovery_replay/`, `runs/r19_hygiene/`) beyond their described contents.

## Verdict: MAJOR-REVISION

## Score: 6/10