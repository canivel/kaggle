# Round 22 Panel Directives (synthesized 2026-07-28)

Panel role: ADVISORY (Sunday-cadence strategic review under the 07-27 restructure; this
cycle was user-ordered mid-week as round 22 with round 21 as prior). MAJOR-REVISION is the
known absorbing state and does not block the build rail.

## VERDICT SUMMARY
Unanimous 5/5 MAJOR-REVISION (scores 5,4,5,5,5; 17 MAJOR, 0 FATAL): the ledger-provenance
defect is fully fixed and the A17 gates are now at least summarized inline, but the three
load-bearing R21 allocation decisions — the readiness-gate A/B, the sentinel un-shelve
rule, and the numeric ρ_action kill threshold Y — are all still punted to the panel as open
questions rather than proposed with numbers, and v6 was fired mid-week before NC-4/NC-5 were
discharged. Every reviewer independently derived that the only 72B throughput data in hand
(v5 1500 s slice: ΣN=5 across 4 games) linearly projects to ~26–33 actions over the 7920 s
window — a ~3–4× miss on the prereg's own G2 ≥ 100 floor — and every reviewer flags that the
brief presents this without stating the implication.

## TOP DIRECTIVES (ranked)

1. **[5/5 reviewers: rl-planning, llm-agents, prog-synthesis, methodology, systems] Publish the v5-slice throughput projection now and pre-register the G2-FAIL contingency branch before v6 is read.**
   All five reviewers independently ran the same arithmetic on the v5 slice (ΣN=5 over
   1500 s ⇒ ~26–33 actions over the full 7920 s window, ~3–4× below G2 ≥ 100) and all five
   say the brief hides this behind "MEASUREMENT ONLY / no interpretation at k=1." The panel's
   consensus: refusing to interpret at k=1 is defensible doctrine for *capability* claims, but
   a throughput shortfall exposed by simple division is not a capability claim and G4 does not
   shield it. The sealed walk must read v6 expecting a G2 FAIL, with the reversion branch
   (slots → gated A/B, in stated priority order) pre-committed in writing, not deliberated post
   hoc. systems (N1) adds the Poisson band on N=5 (~2–10 events → 13–66 projected, upper bound
   still < 100); methodology adds N(lp85)=0 is a distinct dead-path signature (stuck loop /
   per-game init failure) that deserves its own grep, not aggregation.

2. **[5/5 reviewers: rl-planning, llm-agents, prog-synthesis, methodology, systems] Schedule the boristown readiness-gate A/B this week — replace filler draws, do not arbitrate false scarcity against sentinel draw #2.**
   Carried from R21 directive #1 (5/5); every reviewer marks it PARTIALLY-RESOLVED and every
   reviewer calls the residual the worst item in the document. It is a fork-not-build ~one-line
   diff behind an external 1.47 anchor (above our 14-draw max 1.33, near the 1.49 gold cutoff),
   yet a filler draw was burned every day 07-25 → 07-28. The panel rejects question 3's framing
   that the A/B and sentinel draw #2 "compete for the same filler slots": the ledger shows
   there is no scarcity — end filler (which by the brief's own P ≈ 2×10⁻⁴ cannot climb) and it
   funds both. Correct allocation (rl-planning, systems N4, methodology, prog-synthesis
   concur): readiness-gate A/B starts immediately at ~n=4 gated draws replacing the next 4
   fillers (zero incremental slots), sentinel draw #2 queued *behind* it, filler cut to ~2/week.
   It needs a calendar date (~Aug 2 per systems), not a question mark.

3. **[5/5 reviewers, three explicitly naming the mechanism: prog-synthesis, systems, llm-agents (+rl-planning, methodology on variance)] Name Y yourself with its formula, sign convention, and frozen-baseline derivation — asking the panel to invent Y is burden inversion.**
   The authors own the harness and the frozen fork's per-window action counts; the panel does
   not. prog-synthesis raises a blocking defect first: ρ_action is **directionally ambiguous as
   written** — if ρ_action = 480/ΣN₇₂B then ρ *decreases* as the 72B executes more actions, so
   the stated rule "ρ_action < Y ⇒ dead" kills the route for *good* throughput; the formula and
   the kill inequality contradict each other and no threshold on it is falsifiable until this is
   fixed. See the open-question disposition below for the extracted numeric value. This is a
   burden the authors must discharge in one sentence (formula, direction "higher = better", Y,
   inequality, derivation of the constant 480) before the sealed walk reads v6.

4. **[4/5 reviewers: prog-synthesis, methodology, rl-planning, llm-agents; systems provides the alternative single-seed compensation] Y read off a single k=1 seed has unknown false-kill/false-continue rates — pre-register k ≥ 2 seeds or a stated variance bound before any kill verdict.**
   methodology names this the recurrence of the exact single-sample failure mode that shelved
   the sentinel at n=1, now moved onto the build rail; rl-planning concurs and both note the
   per-game N counts (0–2 across four games) are themselves the first variance estimate
   (treat as approximately Poisson). Fix: pre-register k ≥ 2–3 seeds, OR state a one-sided error
   rate against a variance bound derived from that per-game dispersion. If k=1 is forced by GPU
   budget, state the *widened* Y that compensates (rl-planning Q3).

5. **[4/5 reviewers: prog-synthesis, methodology, llm-agents, systems] Specify the pre-registration commitment mechanism — the sealed walk is not credible while one principal operates the seal, the kernel pulls, and the brief.**
   prog-synthesis: nothing prevents (even unintentionally) glancing at the kernel log before Y
   is fixed. Fix: hash-commit the threshold document (Y, mapping, G2-fail branch, formula) and
   record the commit timestamp *before* the v6 kernel status flips to COMPLETE, or have another
   panel member hold the pull. systems and methodology fold this together with the G2-fail
   branch into a single hash-committed one-page document. This is the condition under which the
   "pre-observation" framing of question 1 actually holds.

6. **[3/5 reviewers: rl-planning, llm-agents, methodology] Discharge NC-4 from v6 bench data by having v6 emit raw parse-attempt / parse-failure / fenced-recovery counts — the adapter remains n=1 anecdote otherwise.**
   NC-4 (≥200-replay offline parse study) is admitted undischarged; the canary offers one
   successful roundtrip and one recovered fenced call (hits=1), i.e., n=1 in both directions
   (llm-agents: "prompt-it-better hand-waving with a sample size of one"). The v6 G1/G2 gates
   *would* deliver equivalent data, but only if v6's output reports raw counts (not just G1
   pass/fail) and only if v6 executes ≥100 actions at all — which directive #1 makes doubtful.
   llm-agents also flags a schema smell: `args={"action":"submit_action",...}` duplicates the
   function name inside the argument object — one grep against the game-API call site settles
   whether the hermes parser is tolerating malformed output the API may reject.

7. **[2/5 reviewers: systems N2, with methodology's k=1 caveat] Verify hardware parity between the canary kernel (RTX PRO 6000 Blackwell) and the scored-submission environment.**
   systems: if scored submissions rerun on a different GPU class/count, every v5/v6 number
   (337 s boot, tok/s, ρ_action) is measured on the wrong hardware and does not transfer — this
   is load-bearing for the entire A17 rail. Actionable: grep the frozen fork's *scored* run logs
   (the 14 filler draws) for the GPU name string and confirm it matches the canary's; if the
   scoring tier differs, re-label v6 output as an upper bound and re-bench on the scoring tier
   before any promotion arithmetic.

8. **[2/5 reviewers: methodology, prog-synthesis] Retire the min–max "band" and the in-sample z-score / shrinking-s artifacts.**
   The band endpoints remain the sample's own min–max, widen monotonically, and can never be
   falsified; the "s tightened 3 refreshes in a row" is near-deterministic arithmetic as interior
   draws accumulate, not a stability signal. Use leave-one-out or prior-refresh statistics for each
   draw's z-score; report a proper tolerance interval with a stated out-of-interval trigger; report
   MK/CUSUM as "insufficient power," not "no trend." (Carried from R21 directive #6, still UNRESOLVED.)

## OPEN-QUESTION DISPOSITION

**Question 1 — numeric ρ_action kill threshold Y (pre-observation):** DO NOT punt to the panel;
the authors must name it. The panel supplies a defensible value to be ratified or amended, but
first flags a blocking pre-condition (prog-synthesis): **the metric is directionally
self-contradictory as written and must be fixed before any Y is meaningful.**

- Extracted verbatim, most concrete proposal (systems N3): **"ρ_action ≥ 4.8 ⇒ 72B route dead
  (equivalently, projected full-window Σ N₇₂B < 100, i.e., G2 itself), with a secondary gate that
  projected actions/hr at t=1500 s of v6 must be ≥ 45 to continue the session."** Rationale
  (systems): 480/100 = 4.8, so the kill threshold *is* G2 re-expressed; the frozen fork scores
  0.97 mean at the 480-action denominator, and a ~5× action deficit cannot plausibly be bought back
  by per-action quality at the 72B tier when the external capability anchor (Opus 5, unconstrained)
  is 30.2%.
- Independent concurring formulation (llm-agents): **"kill the 72B route if executed actions < 100
  in the 7920 s window (G2, already registered) OR ρ_action implies < 50% of the frozen fork's
  measured per-window action count,"** because score is monotone in valid actions per episode.
- The two agree on the operative cut: **the kill threshold collapses onto G2 (Σ N₇₂B < 100 ⇒ dead).**
  The single most defensible statement of Y, reconciling the directional bug: define
  **ρ_action = Σ N₇₂B / 480 (higher = better), kill if ρ_action < 0.5** (i.e., < 50% of the frozen
  per-window rate), which is *stricter* than and dominated by G2 itself (Σ N₇₂B < 240 vs < 100);
  since both fire well above the ~26–33 projected count, **the practically-binding, unambiguous
  rule to hash-commit is: 72B route dead if projected/observed Σ N₇₂B < 100 over the 7920 s
  window.** Two unresolved authorship obligations attach (rl-planning Q3, methodology, systems Q3):
  (i) the constant **480 must be confirmed as the frozen fork's *measured* full-window
  executed-action count** (mean? min? on which draw?) — if unmeasured, that is itself a gap after
  six canary versions; and (ii) Y must be **applied to k ≥ 2 seeds or a variance bound**, not one
  k=1 draw (directive #4).

**Question 2 — sentinel arm disposition:** The panel adopts a pre-registered rule (against
shelved-at-n=1), but ranks it BEHIND the readiness-gate A/B. Three reviewers (rl-planning,
methodology, systems N4) explicitly rank the A/B strictly above sentinel draw #2 on information
value (5/5 directive, single-variable causal test, d ≈ 3.6 testable at n=3–4 against the banked
control; the sentinel's disposition memo itself says "no lift channel," n=1 at 0.71, p ≈ 0.07).
Disposition: **take option (b) — pre-register the sentinel un-shelve rule but queue it behind the
A/B**, i.e., un-shelve rule = **draw #2 after v6 lands, n ≥ 4–5, sequential stop: re-shelve only on
2 consecutive draws < 0.80 OR mean of first 3 < 0.80.** methodology imposes a hard condition
(carried into NC-11): the pre-registration MUST publish the rule's error rates, which methodology
computed — **P(draw < 0.80 | frozen) ≈ 0.11, mean-of-3 < 0.80 has p ≈ 0.017, the 2-consecutive
rule over 5 draws fires falsely ≈ 0.05, combined false-kill ≈ 6–7%** — plus the minimal detectable
effect at n=3–5. Shipping the rule without those numbers "delegates the objection instead of
answering it."

## NAMED CONDITIONS (continuing from R21's NC-8)

- **NC-9 (prog-synthesis, methodology, systems):** The ρ_action metric definition, its sign
  convention ("higher = better"), the numeric Y with its inequality, the derivation of the constant
  480 from the frozen fork's *measured* per-window action count, and the G2-FAIL contingency branch
  MUST be committed in a single hash-committed document with a timestamp recorded *before* the v6
  kernel status flips to COMPLETE. No Y on a directionally-ambiguous metric may be ratified.
- **NC-10 (methodology, rl-planning, prog-synthesis):** No kill/continue verdict may be read off a
  single (k=1) v6 seed. Pre-register k ≥ 2 seeds before any A17 kill decision, OR set Y with a
  stated one-sided error rate against a variance bound derived from the per-game N-count dispersion;
  if k=1 is forced by budget, publish the widened compensating Y.
- **NC-11 (methodology):** The sentinel un-shelve pre-registration may not fire until it publishes
  the rule's per-draw and family false-kill rates under the frozen null (methodology's ≈ 6–7%
  combined figure) and the minimal detectable effect at n=3–5.
- **NC-12 (systems):** Before any A17 promotion arithmetic, the canary/scored-environment GPU
  parity must be confirmed by grepping a scored frozen-fork draw's log for the GPU name string; a
  mismatch re-labels all v5/v6 numbers as an upper bound pending a re-bench on the scoring tier.

(R21's NC-1 through NC-8 carry forward unchanged. NC-1/first-push escalation was exercised in
anger — v6, a new artifact version, fired mid-week under user order with the panel demoted to
advisory — which prog-synthesis and methodology mark as the flagged scenario occurring; NC-2/NC-6
gate operating characteristics remain UNDELIVERED and are re-asserted as blocking the next
unreviewed weekday promotion.)

## WHAT CHANGED VS R21 (reviewers' own prior-objection tracking)

- **RESOLVED:** The ledger-provenance defect (R21 directive #3, NC-8). All five reviewers confirm
  `runs/lb_ground_truth.md` was refreshed 07-28 from the live API before any statistic was cited,
  the 07-26/07-27/07-28 draws are canonical and cross-checked against `submission_log.jsonl`, and
  n=14 / 0.9686 / 0.1384 recompute cleanly (three reviewers re-derived it independently). systems
  also marks R21's two MINORs RESOLVED: default-cannot-reach-gold is now stated and quantified
  (P ≈ 2×10⁻⁴), and NC-3's weekday resource cap is addressed (session cap + stall-kill +
  zero-action-abort ≈ 2.5 GPU-h, conditional on stating the numeric stall-kill timeout).
- **PARTIALLY-RESOLVED:** The A17 gates (R21 directive #2 / NC-5) — G1–G4 now appear inline with
  numbers, but the numeric kill threshold Y and the ρ_action→expected-LB mapping remain
  undischarged. The readiness-gate A/B (directive #1) — now surfaced as an option with correct
  prereg shape but still unscheduled. The sentinel un-shelve rule (directive #4 / NC-7) — the
  correct rule structure now appears but the disposition is still returned as OPEN; rl-planning
  and prog-synthesis warn one more round of deferral makes it UNRESOLVED. Push-gated-on-boot-canary
  — overridden by user order (cannot be un-rung) but blast radius bounded to a free kernel build.
- **UNRESOLVED / newly worsened:** The sealed-gate operating characteristics (NC-2) never appeared,
  yet the restructure was exercised in anger with exactly the uncontrolled error rates methodology
  warned about. The fenced-recovery adapter (NC-4) remains n=1 anecdote. The band / MK-CUSUM /
  in-sample-z artifacts (R21 directive #6) persist. War-v4 / three-way-convergence material (R21
  directive #7) was dropped rather than corrected and carries forward to whenever war-v4 resurfaces.
  **New this round:** v6 fired before its own named conditions (NC-4, NC-5) were discharged — an
  order-of-operations violation the panel flags as inviting post-hoc threshold-fitting — and the
  k=1 throughput slice already forecasts a G2 failure that the brief declines to state.
