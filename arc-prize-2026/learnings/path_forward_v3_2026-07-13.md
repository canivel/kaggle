# PATH FORWARD v3 — Revised After Panel Round 6 (5× MAJOR-REVISION, scores 6–7, zero fatals)

**Author:** claude-fable-5 · **Date:** 2026-07-13 · **Revises:** `path_forward_v2_2026-07-13.md` per `panel/round6/*` · **Evidence:** all v2 evidence, **plus new §E2 restart-validation analyses (run 2026-07-13 against `runs/null10/merged_null_benchmark.json`; script preserved)**
**Hard constraints (unchanged):** ~$68 RunPod remaining (reserve-only, zero default GPU spend); free Kaggle quota 30 h/wk; ~79 daily submission windows to Sep 30; Milestone-2 Sep 30; Final Nov 2 (~55 private games); no per-game parameters on private-set-facing logic.

**Filing note (LA-N4, RL/PS/ME/SY "cannot judge" items):** the round-6 review copy truncated mid-§R1 — a distribution artifact, not missing content. v3 is filed complete; `panel/round6/_author_response.md` lists section line-counts so completeness is checkable.

---

## Change-log keyed to round-6 objections

| ID | Objection (reviewer) | Disposition | Where |
|---|---|---|---|
| RL-B / PS-N1 / ME-NEW-1 / SY-N1 | The plan's primary free lever (+0.055 official) cannot pass its own +0.12 promotion gate; P(promote \| works-as-designed) ≈ 14% | **Accepted — two-track promotion pre-registered.** Track A (aggregate, +0.12) unchanged for large-effect candidates. Track B (small-effect): primary evidence = pre-registered event-log mechanism statistic with printed per-event false-attribution rate; window test demoted to regression guard (kill only at Δ̂ < 0). Track B promotions are always provisional and must re-confirm in the stack gate. Sep-30 projection recomputed P(promote)-weighted. Every candidate's expected effect is now checked against its assigned track before submission (the "gate-consistency check" is itself pre-registered). | §Instruments, §R1, §Targets |
| RL-A / LA-N2 / ME-NEW-2 / PS-N4 | Restart EV rests on an unsourced 0.4 depth discount, an untested seed↔restart exchangeability assumption, unspecified RESET context semantics, and a possible version-drift confound in the flip evidence | **Accepted — all four analyses run (§E2); the headline EV did not survive.** (1) Version confound *refuted*: all 25 games carry a single version suffix across all 10 seeds; 16/16 flips are intra-version. (2) Discount *measured* from budget-truncated good-run value curves: disc(90)=0.365 (assumed 0.4 was close), but disc(180)=0.079 — the second restart is nearly worthless. (3) Full EV re-derivation with depth-correct discounts and explicit FP loss: **under last-attempt scoring the net EV is ≈ 0 (−0.03 to +0.02 local) at every trigger — the +0.10 claim is retracted**; under best-across-attempts scoring EV = **+0.24 local** at (90, cap 2) because FP restarts cost nothing. R1 is therefore restructured around a free, pre-registered binary fact-check of the harness/API scoring semantics that decides the lever's fate before any window is spent. (4) RESET specified: fresh episode + fresh per-game analyzer context (no carried scratchpad/summaries), enforced in harness code and verified in transcript logs — required by our own context-pollution results. Exchangeability check pre-registered on first live runs. | §E2, §R1 |
| LA-N1 | The −0.29 gap to our own parent (Tufa duck 1.21 vs our 0.922) is unexplained and contaminates the base of every EV | **Accepted.** R0 now gates the **unmodified vanilla duck** as the first candidate (2 windows) before any fork-band port. If vanilla ≥ substrate, vanilla becomes the porting base and all lever EVs are re-derived against it. The comparison also separates fork-regression from environment drift: vanilla-now vs vanilla-at-Milestone-1 (1.21) measures drift; vanilla-now vs substrate-now measures our fork's net effect on the same games/versions/windows. | §R0 |
| PS-N3 / ME-NEW-3 | Provisional promotes entering the control class, and stale pre-promotion draws in the rolling mean, contaminate every downstream gate | **Accepted.** Pre-registered: (i) only *confirmed* promotions update the control class; a provisional build's redraws are logged in a separate class excluded from the rolling 6-draw mean; (ii) on a confirmed promotion the control window **resets to post-promotion draws only** — the temporarily widened SE is accepted, printed, and refilled by the default-redraw rule (~2–3 weeks to df 6; refill latency added to the window ledger). | §Instruments |
| PS-N2 / LA-M2r | Component fidelity gates passable by trivial predictors; exec-WM metric undefined; decision table may be authored after reading transcripts | **Accepted.** exec-WM gate redefined: object-level transition accuracy **on state-changing transitions only**, with a required **margin ≥ +15 points over a copy-last-frame identity baseline**, split **held out by seed**, n ≥ 200 transitions, plus a transfer report on r11l transcripts. Segmentation gate redefined: object-identity/track consistency across **≥50 consecutive-frame pairs** (not 20 static frames), pass = ≥90% with binomial 95% lower bound ≥ 75%. The forensics→intervention decision table is committed to `ITERATION_LOG.md` **before the first forensic transcript is read** (commit hash recorded there); Aug 3 remains the deadline for the one-page designs only. | §R2 |
| RL-M2r / ME-M4r | The R2 gate's claimed p<0.01 re-pools the clustered null; honest game-level bound is far weaker | **Accepted — honest number stated, p<0.01 retracted.** With 10 null seeds/game, the per-seed rule-of-three 95% UB is q = 0.3; P(one game passes ≥2/3 seeds) = 0.216; **P(≥2 of 3 games) ≤ 0.12 worst-case** (all three games simultaneously at their UBs). The local gate is a *screen*, not the proof: end-to-end selectivity comes from the conjunction with the pre-registered mechanism prediction, the r11l holdout (blocking authority), and the unchanged Track A +0.12 window confirmation. | §R2 |
| SY-N2 / SY-M2r | 12-h commit at the Kaggle session ceiling is a single point of failure; 27/30 h has no failure contingency | **Accepted.** Serving model stated: vLLM in-kernel on Kaggle GPU (no external API; no provider stalls, but vLLM hangs possible). Commit watchdog pre-registered: per-game wall caps enforced by harness-side async timeouts; a top-level watchdog **hard-kills and checkpoints at 10.5 h** so a pathological game costs a game, not the commit. Itemized ledger: 2 commits × 12 h + 2 smokes × 1.5 h = 27 h. **Any week containing a failed commit → cap drops to 1 new build; contingency candidates cut first.** R0-exit evidence now includes the usage-page reading for one *treated (token-inflated)* build, not just null. | §Windows |
| ME-NEW-4 / SY-n2 | df accounting pools non-exchangeable draws across promotions/re-centers | **Accepted.** σ̂ and df are computed **per control era** (era boundaries: confirmed promotion, confirmed drift re-center, game-version bump on a control game); the sign-flip rule consumes the era-local CI; df partially resets at each boundary and is refilled by default redraws (latency in ledger). | §Instruments |
| RL-C | 90-action trigger unswept; simulation must cover flip games | **Accepted — sweep run (§E2).** Trigger {90,120,150} × cap {1,2} × discount {measured, 0.2, 0.4, 0.6} published. The EV model replays all 250 transcripts including the 16 flip games (where the FP cost lives); the per-transcript simulation ships with the build. Under best-across scoring, (90, cap 2) dominates; under last-attempt scoring nothing clears +0.03. | §E2 |
| LA-N3 | Ported fork deltas escape the tokens/action kill rule; 1.56 anchor unverified | **Accepted.** The >10% tokens/action kill rule applies to all R0 ports. R0 audit step 1 verifies the 1.56 leader's kernel is public and Milestone-eligible; if not, the porting target is restated against the highest *audited* fork. | §R0 |
| SY-n1 | actions/level and tokens/action can trade against each other; combination rule unspecified | **Accepted.** Joint non-inferiority pre-registered: a candidate must not regress on *either* metric; plus a composite clears-per-wall predictor (validated against null10's measured throughput) reported for every candidate. | §Instruments |
| SY-M1r | Calibration seed ambiguity; de-scoping table absent | **Accepted.** The calibration seed is **seed 1 of the 3** (not additive; total stays $14–28). De-scoping table printed verbatim in §R2. | §R2 |
| PS-M6r/Q5 | Version bumps between weekly sentinels are a blind interval | **Accepted.** Every scored run (daily daemon) logs game_id+version suffix → detection latency ≤ 1 day, not 1 week. A gate decision taken inside a blind interval is voided retroactively if a gate-relevant game is later found to have bumped inside it. | §Instruments |
| RL-M3r / RL-Q6 | BFS one-pager must include *measured* dry-run tokens/action | **Accepted.** The Aug 3 BFS one-pager will include a measured dry-run tokens/action estimate on logged stall segments from null10 transcripts (not just a prediction) against the >10% kill criterion. | §R2 |
| RL-M5r / LA-N4 / PS-Q8 / ME-Q6 / SY | Truncated document; §Risks/§Windows/§R3 unverifiable | **Accepted.** v3 filed complete (filing note above); all referenced tables are in the body of this document. | header |

---

# Thesis (what actually wins now)

Redraws cannot be the strategy (P(top-100 by luck) ≈ 3% at the σ̂ point estimate) and always-on context injection is net-harmful as built. The v2 restart thesis has now been *stress-tested against its own corpus* (§E2) and survives only conditionally: within-game attempt variance is real and version-clean (16/25 games flip across seeds, all intra-version), but the net value of restarts depends on a harness fact — whether the scorecard scores **best-across-attempts** (EV +0.24 local) or **last-attempt** (EV ≈ 0). So the revised strategy, in cost order:

1. **Vanilla-base check + fork-delta audit (free, first):** gate the unmodified vanilla duck; audit the 1.28–1.56 fork band; port the game-agnostic deltas, each gated, each subject to the tokens/action kill rule. This leg is now first because it is the only leg whose EV does not depend on an unverified assumption.
2. **Attempt scheduler (free, conditional):** a one-day scoring-semantics fact-check decides it. Best-across → build and gate (Track B mechanism statistic + regression guard). Last-attempt → lever killed pre-emptively, zero windows spent, ledger reallocates to fork ports.
3. **One pre-registered shot at the level-2 wall ($≤68):** forensics → decision table (committed before transcripts are read) → redefined component fidelity gates → two-distinct-game gate (honest 0.12 worst-case screen) + r11l holdout + Track A window confirmation.

**Targets, P(promote)-weighted (SY-N1):** base = max(substrate 0.922, vanilla). Components as (official Δ × P(promote)): fork ports +0.10–0.175 × 0.4–0.8 → +0.08–0.20; scheduler +0.13 × P(best-across semantics) × ~0.8 → 0 to +0.10; R2 crack +0.19 × 0.15–0.3 → +0.03–0.06. **P-weighted draw mean ≈ 1.08–1.25; selection-best ≈ 1.2–1.4.** Honest statement: this **contests Sep-30 top-100 (projected cutoff 1.35–1.5) only in the upper quartile of outcomes** — favorable scheduler semantics, an R2 crack, and at least one fork port must land. The formal Sep-30 objective remains an instrumented checkpoint; the campaign is built for Nov-2 (~55 private games) where the compression hypothesis — a *tested* hypothesis via the R0 audit — does the work. Nothing shipped may key on game identity.

# Evidence base

**Carried forward (v1/v2, unchanged):** context pollution replicated (ar25 p=0.009/0.008; su15 p≈0; tokens/action null 435 → v2 543 → v1 633); death mode is STUCK; 5 games dead 10/10; 4 grinders 10/10-L1 0/10-L2 (+0.52 local on the three w=2 grinders); rank 187 at 1.02, draws mean 0.922, σ̂ 0.074, χ² CI [0.044, 0.213]; null10 replay §E (v2): 7.9% of clears after action 120, L1 EV retracted; v1 rehabilitation remains a hypothesis (ME-M5, v2 disposition stands).

## §E2 New: restart-lever validation (2026-07-13; resolves RL-A, LA-N2, ME-NEW-2, RL-C, PS-N4)

All computed from `runs/null10/merged_null_benchmark.json` (250 runs, per-action histories):

**1. Version confound refuted (ME-NEW-2.1).** Every one of the 25 games carries a **single version suffix across all 10 seeds** (0 unstable). All 16 flip games flip **within one version** (e.g. ar25-e3c638: [0,0,2,1,0,2,1,2,1,2]). The flips are genuine within-game stochasticity; cross-seed exchangeability is not confounded by drift *in this corpus*. (The earlier "15/24 version-unstable in July" observation was a longer window; null10's 10 seeds were collected inside one stable window.)

**2. Depth discount measured (RL-A.i).** Budget-truncated value curve over the 126 good-mode runs (value achieved in the first N−t actions as a fraction of full-run value, N = run's own action count; median N = 121, wall-capped):

| t | 90 | 120 | 150 | 180 | 240 |
|---|---|---|---|---|---|
| disc(t) | **0.365** | 0.254 | 0.159 | **0.079** | 0.032 |

The assumed 0.4 was nearly right for the *first* restart; the v2 derivation's error was elsewhere: (a) it under-discounted the second restart (true disc(180)=0.079, not 0.16), and (b) it did not charge the false-positive loss at full value. FP rates (good runs with first clear after t): 11.1% / 7.1% / 3.2% at t = 90/120/150.

**3. Full EV re-derivation, decomposed (trigger sweep, RL-C), in local game-points per 25-game draw, depth-correct measured discounts:**

| t, cap | gain (stuck runs) | FP loss | FP recovery | **net, last-attempt** | **net, best-across** |
|---|---|---|---|---|---|
| 90, 1 | +0.168 | −0.242 | +0.055 | −0.019 | +0.223 |
| **90, 2** | +0.185 | −0.242 | +0.059 | **+0.001** | **+0.243** |
| 120, 2 | +0.124 | −0.172 | +0.024 | −0.024 | +0.148 |
| 150, 2 | +0.076 | −0.061 | +0.007 | +0.022 | +0.084 |

**Conclusion: under last-attempt scoring the restart EV is ≈ 0 at every setting — v2's +0.10 ± 0.05 is retracted.** Under best-across-attempts scoring (FP restarts cost nothing) the lever is worth **+0.24 local ≈ +0.13 official** at (90, cap 2), which dominates the sweep. Sensitivity at assumed flat discounts {0.2, 0.4, 0.6} also published in the analysis script; the measured-curve numbers above are authoritative.

**4. Mechanism statistic (for Track B).** 138/250 runs are stuck at action 90. Model prediction: **≈1.5 restart-recovered ≥1-level clears per 25-game draw** (P(fresh good attempt clears ≥1 level in the post-restart budget) = 0.381). Per-event false-attribution rate (a stuck-at-90 run that would have cleared anyway, i.e. an FP): **14/138 = 10.1%**.

**5. What replay cannot settle (pre-registered live checks):** (a) seed↔restart exchangeability — first live scheduler runs must report the restart-attempt good-mode rate vs the cross-seed p on the same games; (b) RESET context semantics — transcripts must verify a fresh per-game analyzer context (no carried scratchpad/summaries; our ar25/su15 pollution results predict a correlated, degraded draw if anything carries).

# Standing instruments (free) — v3

- **null10**: version-pinned paired-control corpus, as v2, now with the §E2 verification that all 25 games are version-homogeneous within it. Refresh triggers unchanged.
- **Version-bump detection (PS-Q5):** every scored run logs game_id+version suffix → detection latency ≤ 1 day (daily daemon), not one week. Any gate decision taken inside a later-discovered blind interval on a gate-relevant game is retroactively voided.
- **Window gate v3 — two tracks (RL-B/PS-N1/ME-NEW-1/SY-N1):**
  - **Track A (aggregate):** unchanged — control = rolling mean of the 6 most recent control-class draws; candidate gets 2 scored windows; promote at Δ ≥ +0.12, kill at Δ̂ < 0; sign-flip rule and printed error rates as v2 (false-promote at Δ=0: 2.4%/24.5% across the σ̂ CI; false-kill of true +0.10: 4.9%/28.3%). Assigned to candidates whose pre-registered expected effect ≥ +0.12 official.
  - **Track B (small-effect / mechanism):** for candidates with expected effect < +0.12 official. Primary evidence = a pre-registered **event-log mechanism statistic** with a printed per-event false-attribution rate (for the scheduler: ≥3 restart-recovered clears across 2 windows, prediction ≈1.5/draw, attribution error 10.1%/event → P(≥3 events all-spurious) < 0.01 at the predicted event rate). The aggregate window test is only a **regression guard** (kill at Δ̂ < 0). Track B promotions are **always provisional**: excluded from the control class and required to re-confirm inside the stack gate.
  - **Gate-consistency check (pre-registered process rule):** before any candidate is submitted, its expected official effect is compared to its track's threshold and P(promote | works-as-designed) is printed. A candidate whose own EV cannot plausibly pass its assigned track is not submitted — the track assignment is fixed at pre-registration, not after data.
- **Control-class hygiene (PS-N3/ME-NEW-3):** only confirmed promotions update the control class; provisional-build draws are logged in a separate class excluded from the rolling mean. On a confirmed promotion the control window resets to post-promotion draws only; the widened SE is printed and refilled by default redraws (~2–3 weeks to df 6; latency carried in the window ledger).
- **Era-local σ̂/df (ME-NEW-4/SY-n2):** σ̂ and df computed per control era (boundaries: confirmed promotion, drift re-center, version bump on a control game); the sign-flip rule consumes the era-local CI. df ≈ 8 by the first candidate gate holds only within an unbroken era; after a boundary, df resets to the post-boundary count.
- **Joint non-inferiority (SY-n1):** every candidate must not regress on *either* actions-per-completed-level or tokens/action; a composite clears-per-wall predictor (validated against null10's measured tokens/sec throughput) is reported alongside.
- **Drift rule:** unchanged from v2 (per-game sentinel statistic, freeze-confirm-recenter).

# The plan

### R0 — Instruments + vanilla base + fork-delta audit (Jul 14–27; free)

1. Draw #6 completes the σ panel; offline scorer + version-pinned null10 committed; all thresholds in this doc pre-registered in `ITERATION_LOG.md` before any candidate submission.
2. **Vanilla-base gate (new, LA-N1):** submit the **unmodified vanilla duck** through the standard 2-window gate *before any port*. Readout: (i) vanilla-now vs 1.21 (Milestone-1) measures environment drift; (ii) vanilla-now vs substrate-now measures our fork's net effect on identical games/windows. **If vanilla ≥ substrate, vanilla becomes the porting base and every lever EV is re-derived against it** — reverting is then the cheapest intervention in the plan and is taken.
3. **Fork-delta audit:** as v2 (enumerate the 1.28–1.56 band, delta table, game-agnostic vs game-keyed classification, compression-hypothesis test), plus (LA-N3): **step 1 verifies the 1.56 leader's kernel is public and Milestone-eligible** — if not, the porting target is restated against the highest audited fork; **every ported delta is subject to the >10% tokens/action kill rule** and joint non-inferiority.
4. **Scoring-semantics fact-check (new, decides R1):** determine from the ARC-AGI-3 API/harness documentation and existing scorecard logs whether a game's official score is **best-across-attempts** or **last-attempt** within a scorecard. One day, free, binary, pre-registered. Outcome recorded in `ITERATION_LOG.md` before any R1 work.
5. **Quota + window ledgers published**; measured commit-hours verified against the Kaggle usage page **for one treated (token-inflated) build and the null build** (SY-N2).
**Exit:** thresholds logged; draw #6 scored; vanilla gate underway; audit deliverable filed; semantics fact recorded.

### R1 — Attempt scheduler (Jul 21–Aug 10; free; conditional on R0.4)

**Gate 0 (semantics):** if last-attempt scoring → **lever killed pre-emptively at EV ≈ 0 (§E2.3); zero windows spent; R1 windows reallocate to fork ports.** If best-across → proceed:
- **Design (from the §E2 sweep):** trigger = 90 actions since episode start with `lc == 0` (per-attempt counter); **cap 2 restarts** (cumulative counter, never resets); park after cap; park dominates restart; dead games bounded ≤270 actions then parked. Per-transcript simulation over all 250 null10 runs (including the 16 flip games, where FP fires) ships with the build (RL-C).
- **RESET semantics (LA-N2/RL-A.ii):** fresh episode **and fresh per-game analyzer context** — no carried scratchpad, summaries, or history. Enforced in harness code; verified in transcript logs before the gate is scored.
- **EV: +0.24 local ≈ +0.13 official** (measured-discount model, §E2.3), sensitivity band +0.08 to +0.30 across the discount table.
- **Promotion: Track B.** Primary = mechanism statistic (≥3 restart-recovered clears across 2 windows; prediction ≈1.5/draw; per-event attribution error 10.1%); regression guard = aggregate Δ̂ ≥ 0. Exchangeability check (§E2.5a) reported with the first window. P(promote | works-as-designed) ≈ 0.8 (event-based, vs 14% under the old aggregate-only gate — the RL-B incoherence is resolved by instrument choice, not threshold inflation).
- **Retry budget:** one (cap 1 variant) if the build fails. Preflight + runtime smoke mandatory.

### R2 — Level-2 wall (Jul 21–Aug 10; free CPU; ≤$68 reserve gated)

**Forensics (free, CPU):** as v2 (mechanic verbalization, hypothesis churn, action entropy, repeated-plan signatures, state coverage). **Ordering rule (LA-M2r): the forensics→intervention decision table below is committed to `ITERATION_LOG.md` — commit hash recorded — before the first forensic transcript is read.** Aug 3 is the deadline for the one-page designs only.

**Pre-registered decision table (unchanged from v2):**

| Observed signature (30 grinds) | Diagnosis | Intervention class |
|---|---|---|
| Mechanic never stated | Model-capability wall | Stall-scoped systematic exploration (duck+BFS) |
| Mechanic stated, then lost across turns | Context-management bug | Stall-scoped world-model pinning / verify loop |
| Mechanic stated, plan correct, execution wrong | Grounding failure | exec-WM action-verification loop |
| Low state coverage vs null | Exploration starvation | duck+BFS (coverage-driven) |

**Design specs due Aug 3, one page each:** as v2 (state space, simulated-vs-executed, cost per node), and for duck+BFS a **measured dry-run tokens/action estimate on logged stall segments from null10** (not a prediction) against the >10% kill criterion (RL-Q6).

**Component fidelity gates before any GPU spend — redefined (PS-N2):**
- **Segmentation:** object-identity/track consistency across **≥50 consecutive-frame pairs** from grinder transcripts (the 4-connected fragmentation failure is what's being quantified); pass = ≥90% point estimate **and** binomial 95% lower bound ≥75%.
- **exec-WM (if selected):** object-level transition accuracy **on state-changing transitions only** (identity transitions excluded from the numerator and denominator), **≥70% and ≥ +15 points above the copy-last-frame identity baseline computed on the same set**; split **held out by seed** (never random within-game); n ≥ 200 transitions; a transfer report on r11l transcripts accompanies the gate.
- Fail → intervention struck from the shortlist.

**Reserve unlock:** unchanged (falsifiable mechanism prediction + r11l holdout prediction pre-registered). **Cost provenance:** unchanged ($14–28 for 3 seeds, wall-capped; token inflation degrades actions-per-wall, not dollars). **Calibration (SY-M1r): the calibration seed is seed 1 of the 3 — not additive.** De-scoping table (verbatim): measured 3-seed projection ≤$35 → 3 seeds × 3 games (gate ≥2 games at ≥2/3 seeds); $35–50 → 2 seeds × 3 games (gate ≥2 games at 2/2 seeds); >$50 → 1-seed screen only, promotion evidence moves entirely to windows.

**Primary gate — honest error rate (RL-M2r/ME-M4r):** level-2 clears on **≥2 distinct games** of {sb26, su15, lp85} with **≥2/3 seeds each**. Null: 0/30 in null10; per-seed rule-of-three UB q = 0.3 → P(a game passes) = 0.216 → **worst-case false-pass ≤ 0.12** (simultaneous 95% UBs on all three games; the previously claimed p<0.01 is retracted). This gate is a *screen*; end-to-end selectivity is the conjunction: local screen (≤0.12) ∧ pre-registered mechanism prediction ∧ r11l holdout (blocking) ∧ Track A window confirmation (+0.12, its own printed error rates). A one-game crack does not pass. **Secondary:** per-game paired sign statistic over ~20 version-matched non-wall games, actions/tokens metrics only, joint non-inferiority applies.

**Promotion authority:** the free window gate (Track A — expected effect +0.19 ≥ +0.12), solely. GPU run is a local screen. Reserve retry only on the near-miss rule.

### R3 — Stack and freeze (Aug 11–Sep 30)

As v2, with control-class hygiene: candidates gate sequentially against the updated rolling control **containing confirmed promotions only**; provisional (Track B) promotes re-confirm here or are dropped; stack gate vs the vanilla-duck fork (≥ +0.12 on its own 2-window test at both era-local CI endpoints). Explore-min enters at zero prior credence, one gate + one retry. Multiplicity: expected false promotions across ~6 families recomputed with Track B included ≈ 0.2 (point) to 1.6 (CI upper); bounded by the stack gate, the vanilla floor, and the provisional-exclusion rule. **Freeze Sep 12.** Sep 13–30: 4–5 selection draws; ship criterion unchanged (both era-local CI endpoints; never below the vanilla-duck fork).

# Submission windows (~79) and quota — ledgers v3

**Window ledger (Jul 14–Sep 30, 11.3 wk):** R0 draw #6: 1 · **Vanilla-base gate: 2 (new)** · Fork-delta ports (2 × [2+1]): 6 · R1 scheduler (2+1 gate, 2 retry — **spent only if semantics favorable, else reallocated to ports/contingency**): 0–5 · R2 confirmation (2+1): 3 · Explore-min (2+1): 3 · Stack gate + vanilla floor: 3 · Contingency candidates: 8 · Weekly sentinels: 11 · Selection: 5 · **Enumerated: 42–47** · Default filler: best-build redraws (~32–37; double as control-class draws → era-local df refill after promotions/re-centers — the refill latency lives here). Priority rule unchanged: gates preempt redraws; only current-best redraws allowed.

**Kaggle quota ledger (30 h/wk), itemized (SY-N2):** 2 new-build commits × 12 h = 24 h + 2 smoke tests × 1.5 h = 3 h → 27 h ≤ 30 h. Re-submission of a committed version = 0 GPU-h (daemon-validated since May). **Watchdog: per-game async wall caps in-harness; top-level hard-kill + checkpoint at 10.5 h** — a pathological game costs a game, never the commit. **Failed-commit week → cap 1 new build; contingency candidates cut first; gate schedule re-checked at the degraded rate.** Serving is vLLM in-kernel on Kaggle GPU (no external API). Measured commit-hours for null *and one treated build* verified at R0 exit; >15 h measured → permanent cap 1/wk.

# Kill criteria (pre-registered)

- **R0:** vanilla ≥ substrate → revert; porting base and all EVs re-derived. Fork audit finds top band predominantly game-agnostic → compression hypothesis fails; consequence as v2. 1.56 kernel not public → target restated vs highest audited fork.
- **R1:** last-attempt semantics → lever dead pre-emptively, zero windows. Exchangeability check fails (live restart good-mode rate ≪ cross-seed p) → lever dead after first window. Mechanism statistic <3 events across 2 windows, or Δ̂ < 0 → dead (one cap-1 retry).
- **R2:** as v2 (no prediction → no spend; fidelity gate fail → struck; <2 games → dead unless near-miss; r11l contradiction → blocked), plus: decision-table commit hash must predate first transcript read or the forensics are void.
- **Context injection / global / re-baseline:** unchanged from v2, plus joint non-inferiority on every candidate and the gate-consistency check (no candidate submitted whose EV cannot plausibly pass its pre-assigned track).

# Risks

1. **Level-2 wall is a model-capability gap (high).** Unchanged; ceiling then = fork ports + (conditional) scheduler ≈ +0.08–0.30 official.
2. **Compression hypothesis false.** Tested at R0; pre-registered consequence unchanged.
3. **Scheduler semantics unfavorable (new, ~50%).** Last-attempt scoring kills leg 2 at zero cost; Sep-30 P-weighted mean falls to ≈ 1.08–1.18 and top-100 becomes a tail outcome. Accepted — the fact-check costs nothing and prevents 5 wasted windows.
4. **Seed/draw lottery.** Unchanged (ft09 26% of score; mitigations: version-matched pairing, stratification, era-local CIs, both-endpoint reporting).
5. **Cutoffs may outrun the S-curve / difficulty ratio 0.55 unvalidated.** Unchanged; ratio refit at R1 (now from fork-port per-game deltas if the scheduler is killed).
6. **Env fragility (proven 5×).** Fork-never-build, byte-matched metadata, preflight, runtime smoke — mandatory.

---
*Supersedes `path_forward_v2_2026-07-13.md`. §E (v2 replay) carries forward; §E2 analyses run 2026-07-13 against `runs/null10/merged_null_benchmark.json` (script: `learnings/panel/round6/_e2_analysis.py`). Disposition of every round-6 objection and answers to all reviewer questions: `panel/round6/_author_response.md`.*
