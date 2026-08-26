## Summary (2 sentences)

This brief shows genuine methodological progress — the declared t-predictive is actually used on the 0.84 draw, the MK/CUSUM no-trend check I demanded now exists, the sentinel closed via a disposition memo, and the A17 v3 forensics correctly refuse a capability claim from a severed action channel. But the entire statistical protocol the brief operates under (amendment 2026-07-24, entry bar §(c), pooling rules) is still an unratified DRAFT because the ratifying panel round has now failed to launch three times, and the v4 canary design introduces new validation and confounding problems that must be pre-registered before the push.

## Objections

**[PRIOR — RESOLVED] Provisional and conflicting numbers.** The stale 0.43/1.56 template is root-caused (May-era hardcode, fixed 2026-07-24, `panel_round.py` now reads the ground-truth file) and the one-sentence reconciliation (0.43 = forge-era May best; duck fork lifted the floor from 07-05) is now written into the canonical briefing. Closed. This took three rounds longer than it should have, but it is done.

**[PRIOR — PARTIALLY-RESOLVED] Single-draw regime evidence / per-arm pre-registration.** No violation this round: Q5 correctly finds no arm clears entry bar §(c) and filler rides, and war-v4 explicitly waits on A17. But the codification of "named non-score observables + null criterion before A21 entry" lives inside the DRAFT amendment, which remains unratified (see new objection below) — the discipline is currently voluntary, not binding.

**[PRIOR — UNRESOLVED] Retrofitted seal rule (A25).** Fourth round of silence. The prospectivity sentence (seal-qualifying rounds begin R18+; R16/R17 do not count) still does not appear anywhere. One sentence; carried again, and I note R20's "3 reviewers, reduced bench" relaunch is exactly the drift vector I flagged.

**[PRIOR — PARTIALLY-RESOLVED] Falsification disjunction / self-satisfiable endpoint.** The amendment §(a)–(i) may contain the endpoint fix — I cannot tell from the brief, and it is DRAFT regardless. Until R20 ratifies and the clause mapping is published (see Question 1), the reset's endpoint structure remains formally what it was.

**[PRIOR — PARTIALLY-RESOLVED] Multiplicity control on the build-rail funnel.** An entry bar §(c) now exists and was applied ("no arm currently clears it") — that is the mechanism I asked for, operating once. Its content (whether it aggregates all prior evidence on a composition, including eval seeds and mechanism verdicts, into a stated prior) is unverified and unratified.

**[PRIOR — PARTIALLY-RESOLVED] E[max] / wall trajectory.** The t-predictive is declared and used (t ≈ −0.9 on the 0.84). Still missing: formal retirement of P(touch 1.44)≈0.18, the recomputed P(touch) under the declared model with σ̂'s CI propagated, and the multi-day rank/score fit. These are ledger edits, not experiments.

**[PRIOR — UNRESOLVED, MINOR] Rule-of-three bounds on 29/29 and 49/49.** Not mentioned. Carried.

**[PRIOR — PARTIALLY-RESOLVED] Pooled-posterior stationarity.** The 07-24 MK/CUSUM no-trend verdict is exactly the time-ordered check I demanded — good, and it is correctly cited as standing rather than re-litigated per draw. The general pooling rule apparently lives in the DRAFT amendment; it binds nothing until ratified.

**[PRIOR — PARTIALLY-RESOLVED] Harm-pause error rates.** The sentinel is now "SHELVED by disposition memo" — the instrument I asked for exists, though I cannot verify it states the ≈13% false-pause rate and evidence-dependence structure. The resume path for a *future* false pause is still undefined; that gap is live the moment war-v4 or any A17-derived arm enters A21.

**[PRIOR — PARTIALLY-RESOLVED] Tail-model shopping.** The single declared predictive model is adopted in practice (the 0.84 is reported under it, unremarkable, correctly). But the declaration is in a DRAFT amendment, and the retroactive recomputation of the 0.71 (p ≈ 0.07, not 0.044) and of P(touch) has not been published. Finish the recompute; it changes no decision and costs ten minutes.

**[PRIOR — PARTIALLY-RESOLVED] Independence-unverified aggregation for shelving.** The disposition memo route was taken (my proposed alternative to running W2), which is acceptable — conditional on the memo actually stating evidence weights and the seed-dependence structure, which I cannot verify from this brief. Publish the memo path and its one-paragraph content in the next brief.

**[PRIOR — PARTIALLY-RESOLVED] boristown re-baselining.** A "§(i) monitored-continuation" clause now exists in the DRAFT amendment, suggesting the baseline-change category was created. But the operational parameters — k control draws before any arm rides the new fork, the new pause threshold derivation, and the retire-vs-stratify rule for old-band draws — are stated nowhere in the brief. Q4 ("schedule now or hold?") must not be answered "now" until those three numbers are written.

**[PRIOR — RESOLVED, MINOR] "~4 ranks/day bleed."** The rate claim has been dropped; this brief says only "eroded to ~#50+" with no extrapolation feeding allocation. Acceptable.

**[NEW — MAJOR] The brief runs the statistics of an unratified protocol, and the ratification mechanism has now failed three times.** The 0.84 was scored against the DRAFT t-predictive, entered the frozen stratum (n=11→12) under the DRAFT pooling rule, and Q5 was adjudicated against DRAFT entry bar §(c) — while R20, the round that would ratify all of this, died before launch for the third time in the same wedge class (07-21/07-22/07-25). If R20 amends any clause, every number computed since 07-24 was produced under a rule that never existed. Fix: (a) flag all post-07-24 analyses "provisional, recompute under ratified rules" in the ledger until R20 closes; (b) treat the panel-launch wedge as an engineering incident with a named mitigation (watchdog/retry on reviewer launch), because a governance process with a 3-failure week is itself the weakest link in the experimental design.

**[NEW — MAJOR] The 99.5% recovery figure is a same-sample validation and must not be treated as the on-node recovery rate.** The fenced-python adapter was designed on the exact 436 turns it was then validated on — training set equals test set, so 434/436 is an upper bound on parser coverage, not an estimate of live performance. Worse, those 436 turns come from a *stalled* loop (0 actions executed, all games `gave_up`); once recovered actions execute, the context distribution shifts and the model's output format may drift, so the turn population v4 faces is not exchangeable with the replay population — the same caveat applies to the 1.1x cadence ratio. Fix: pre-register v4 gate criteria *before* the push — a minimum on-node `fenced-recovery hits/turn` rate (state it, e.g., ≥0.95), a minimum executed-action count, and the cadence recheck — and record now that 99.5% is a ceiling.

**[NEW — MAJOR] v4's (i)+(iv) bundle confounds two candidate root causes with no pre-registered attribution rule.** The brief itself flags vLLM #31871 (hermes streaming raw-text bug) as "plausibly OUR exact defect," then proposes shipping the recovery adapter *and* non-streaming in the same push. If v4 works, you cannot tell whether the format pathology or the streaming bug was causal — which determines whether v5 needs xgrammar and whether the adapter is load-bearing or dead code masking a fixed bug. The single-push design is defensible on window economics *only if* the attribution rule is written first: pre-register that `hits≈0` with actions flowing ⇒ streaming bug was causal (adapter removable); `hits≈1200/1200-scale` ⇒ format pathology confirmed (adapter load-bearing); intermediate ⇒ both live. This costs one paragraph and the banner already provides the observable.

**[NEW — MINOR] "In-band 0.82–1.33" is a min–max acceptance region that can only widen.** Using the observed range as the pre-registered expectation makes "in-band" asymptotically unfalsifiable — every new extreme extends the band it is then judged against. Replace with a central 90% (or stated coverage) t-predictive interval from the prior draws, recomputed per draw; the 0.84 passes either way, so this is free.

**[NEW — MINOR] "No NEW incidents this week" is true only by definitional carve-out.** The R20 launch failure (third in its class), the empty queue at session start, and the A17 canary zero-action runs all occurred this week and are all excluded from the fingerprint store by scope rules. Either add a panel/session-wedge family and an eval-kernel family to the table, or retitle the claim "no new *submission-path* incidents" — as written it overstates weekly health to the panel.

## Questions for the authors (numbered)

1. Publish the clause map for amendment §(a)–(i): which clause resolves, respectively, the §6 primary-endpoint fix, the seal prospectivity sentence, the general pooling rule, the harm-pause error rates + resume path, and the A21 entry-evidence aggregation? Any of my carried objections *not* covered by a clause should be named now.
2. Where are the recomputed t-predictive values for the sentinel 0.71 (p ≈ 0.07) and the retired-and-repriced P(touch), and will they be in the ledger before draw 2/12 is priced?
3. For v4: state the pre-registered on-node recovery-rate threshold, the minimum executed-action count that constitutes a valid canary, and the hits-based attribution rule between the adapter and non-streaming (my new MAJOR above). Also: does "turn≈action" for canary-stage ρ_action carry a stated tolerance, given the 1.1x came from a stalled loop?
4. For boristown adoption (Q4): state k (new-fork control draws before any arm rides), the new pause-threshold derivation, and the old-stratum disposition rule. Absent those three numbers my answer to Q4 is HOLD regardless of A17.
5. What is the engineering mitigation for the panel-launch wedge class (3 failures: 07-21, 07-22, 07-25), and why is it not a fingerprint family?
6. Confirm all analyses since 07-24 (including tonight's n=12 update) carry a "provisional pending R20 ratification" flag in the ledger.

## What I cannot judge

The systems-level claims: vLLM serving health metrics (34.3 gen_tps, stall/restart counts), xgrammar FSM failure modes and the ACTION6 schema's compatibility, AWQ chat-template stripping, GPU memory envelope arithmetic, and the plausibility of the Qwen fenced-code pathology diagnosis (I accept the cited community evidence at face value; systems reviewer should verify). I also cannot judge the RL/agent-design merit of the duck harness composition or whether the A17 depth lane is the right strategic bet — only whether its evidence chain is sound, which is what I reviewed. I cannot verify the *contents* of the disposition memo, the DRAFT amendment, or `runs/a17_recovery_replay/` from this brief.

## Verdict: MAJOR-REVISION

The trajectory is right — the t-predictive, MK/CUSUM, disposition memo, and entry bar are my prior objections materializing as machinery, and the A17 forensic chain is exemplary in refusing capability claims. But the machinery is unratified after three governance failures, the v4 design has a same-sample validation being quoted as a performance estimate and an unaddressed causal confound, and two one-sentence fixes (seal prospectivity, endpoint amendment) are now four rounds overdue. Conditional authorization of v4 is defensible **only** with the pre-registered gate criteria and attribution rule filed before the push.

## Score: 6/10