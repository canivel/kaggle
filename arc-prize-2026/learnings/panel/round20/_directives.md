# R20 synthesis — 2026-07-26 (3 reviewers, prior=R19; relaunch after 07-25 wedge)

**Verdicts: 3× MAJOR-REVISION (scores 6/6/6), 0 FATAL.** Amendment
2026-07-24 does NOT ratify this round (machinery credited but unratified;
clause map demanded first). **Canary v4 is CONDITIONALLY AUTHORIZED by both
methodology and systems** — conditions below must be on file BEFORE the push.

## Binding pre-push conditions (v4)

1. **Pre-registered v4 gate criteria** (methodology NEW-MAJOR): minimum
   on-node fenced-recovery rate ≥0.95 of eligible turns; minimum
   executed-action count for a valid canary; cadence recheck computed ONLY
   from `step_executed=True` turns. **99.5% is recorded as a same-sample
   CEILING, not an estimate.**
2. **Attribution rule** (methodology NEW-MAJOR): written before push.
   Resolution on file: v3 already ran non-streaming (`openai_compat.py`
   `"stream": False`) ⇒ the vLLM streaming bug was never a candidate cause;
   v4 is a SINGLE-VARIABLE push (adapter only, no serve-config change).
   hits≈0 with native tool calls flowing ⇒ pathology absent under v4
   context (adapter dead code — investigate); hits at turn-scale ⇒ format
   pathology confirmed, adapter load-bearing.
3. **turn≈action REJECTED** (systems): the 1.1x cadence came from a stalled
   loop (0 executed actions, degenerate static-frame context) and biases
   toward false GO. v4 measures cadence from executing turns; **no GO
   interpretation from v4 alone** — sealed-gate ρ_action additionally needs
   the matched-concurrency 27B control leg (4-game) or a pre-registered
   correction.
4. **Zero-action in-game assert** (systems): abort loudly if actions_total
   is still 0 mid-window instead of burning the full 2.5 GPU-h.
5. **Quota classification + GPU-h table** (systems, third ask): evidence
   from the submission ledger that eval kernels consume GPU quota only,
   never submission windows; itemized weekly GPU-h budget.

## Other directives (today/tomorrow, all $0)

- Flag ALL post-07-24 statistics "provisional pending ratification"
  (methodology): the 0.84 scoring, n=12 stratum entry, Q5 adjudication.
- Panel-launch wedge (3 failures same class: 07-21/07-22/07-25) = named
  engineering incident; add a panel/session-wedge family + eval-kernel
  family to the fingerprint store (or retitle the weekly claim
  "submission-path incidents only"). Mitigation candidate: panel runs as
  detached background process with the daily loop collecting by file poll
  (today's successful pattern), plus watchdog retry.
- Replace min-max "in-band 0.82–1.33" with a central 90% t-predictive
  interval per draw (methodology MINOR — free, adopt).
- Publish: clause map for amendment §(a)–(i) vs carried objections; seal
  prospectivity sentence (A25, 4th ask); recomputed sentinel 0.71 p≈0.07 +
  re-priced P(touch) into the ledger; two-line exceedance model
  (rl-planning: 0/12 frozen draws >1.33 while byte-identical forks drew
  1.39/1.47).
- boristown Q4 = **HOLD** until three numbers are stated (methodology): k
  new-fork control draws, new pause-threshold derivation, old-stratum
  disposition rule. rl-planning dissents (highest-EV move, false coupling
  to A17) — carry as R21 agenda with the three numbers drafted.
- Exploration draw 2/12: filler rides (all three concur — no arm clears
  §(c)).

## Queue ruling

Tonight's head: frozen-fork filler (armed 08:30) — UNCHANGED. No reviewer
objected; boristown swap is HELD per above.
