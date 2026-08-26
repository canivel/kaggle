# Pre-registration amendment — 2026-07-18b (post-R14)

Filed 2026-07-18 ~13:30 EDT, interactive daily-loop session, in response to panel
round 14 (`learnings/panel/round14/`: 5× MAJOR-REVISION, 0 accepts; new fatal-class
objection raised independently by methodology, llm-agents, and rl-planning).
Supersedes §3 of `learnings/war_room/grinder_cracking_design.md` (the "seals on
filing" clause). LB draw #5 (war-v1 final accumulation) has NOT been observed at
filing time (submits ~00:07Z Jul 19); nothing in this amendment conditions on it.

## A14 — §3 gate seal VOID; recalibrated gate design (discharges the R14 FATAL)

**Void declaration.** The §3 gate never validly sealed: (i) the panel reviewed a
truncated circulation ("Part 1 of 2", cut mid-sentence — argv budget defect in
`scripts/panel_round.py`), and a gate cannot seal on a document the panel has not
seen in full; (ii) the sealed primary prong was arithmetically unpassable — an
exact one-sided sign-flip at α = 0.0125 requires ≥7 uncontradicted nonzero wins
(2⁻⁷ ≈ 0.0078), while the doc's own §4 expectation is 1–4 nonzero improvements per
window. As written, the FAIL rule ("flag OFF") would deterministically park every
component the doc itself expects to help (methodology FATAL; llm-agents N1;
rl-planning independently). Both defects concede the panel's point in full.

**Recalibrated design (seals on R15 full-document circulation, before the first
flagged-window look):**

1. **Pooling unit (defined, per methodology Q2):** the paired unit is the GAME —
   per-game mean Δlevels_completed across the 3 certified seeds (n = 24 game-level
   pairs; su15 excluded per A12; exact zeros dropped before the flip; W/L reported).
   Not 75 game×seed pairs — cross-seed consistency is not demanded of variance-
   harvest components.
2. **Binding score decision = ONE cumulative sealed look:** final v3 stack (all
   mechanism-retained flags ON) vs the W0 baseline ((f)-only), 3 certified
   seed-only-diff replicates, compound prongs unchanged in form (pooled Δlc
   sign-flip α = 0.0125 one-sided; mean Δlog1p(RHAE) across seeds ≥ 0). The summed
   stack expectation (+0.07–0.19 rail) is the alternative this look is powered
   against; single components are not score-gated (they sit below the 3-seed MDE,
   as §2 itself states).
3. **Per-window looks DEMOTED to:** (i) the mechanism prong (trigger counter
   ≥1/run on ≥5 games + the component observable, incl. the A19 counter below) and
   (ii) the non-inferiority guard: pooled Δlc ≤ −0.10 → flag OFF. The guard is
   evaluated ONLY at the window's sealed look after seed 3 — never per-seed; there
   are no interim looks (resolves the "at any look" contradiction, methodology
   minor). Per-window score statistics are reported as descriptive monitoring,
   never as ON/OFF criteria.
4. **Retention rule:** a component stays in the cumulative stack iff mechanism
   prong PASS and non-inferiority PASS. "Mechanism fires, doesn't pay" is no longer
   a per-window kill — it is the question the cumulative look answers.
5. **P(pass | §4 expectations), published (methodology fix ii):** under the §4
   table (expected nonzero positive game-level means ≈ 4–8 at the full stack, ~0–1
   negatives), a binomial sketch gives P(pass) ≈ 0.05 (4 positives, clean) to
   ≈ 0.6 (7–8 positives, ≤0 negatives); point estimate ≈ 0.2–0.4. Honest reading:
   the cumulative look is a real test with real failure probability, not a rubber
   stamp — and not a false-negative machine.
6. **Cumulative-FAIL consequence (sealed now, per methodology Q5):** on FAIL, the
   stack is NOT dismantled (components already passed mechanism + non-inferiority);
   it ships with the honest label "mechanisms verified, score effect unconfirmed at
   rail MDE" and LB-accumulation status per A8. On PASS it is labeled a confirmed
   floor/mid raiser. Either way war-v4 remains the only registered wall-closer.

## A15 — Compressed-bench transfer rule (3× MAJOR)

A compressed-budget (40%-cap) window pass grants **provisional inclusion only**.
Before the cumulative look can claim score credit — and before any component
enters a scored-stack LB kernel — one FULL-budget certified confirmation replicate
of the accepted stack vs the W0 baseline must run on the rail. Per-component
trigger frequencies at full budget (from existing war_eval seeds 1–3 transcripts)
are published alongside every compressed-bench count, so the compression factor is
an explicit checkable assumption. All §2 ceilings measured at compressed budgets
are relabeled compressed-regime quantities.

## A16 — Banking retry de-bias (methodology MAJOR; rl-planning/llm-agents/prog-synthesis minors)

The frozen retry-target list (ft09/sc25/re86) is RETIRED — it was selected on the
same 3 seeds that estimated E[max-of-2] (winner's curse). Replacement, pre-registered
as an online game-agnostic policy: **retry a game iff its current-attempt outcome is
below its banked record AND the remaining soft-time budget covers the banked trace's
replay cost; order retries by (banked − current) descending.** The feasible ceiling
(≤ +0.15) must be recomputed under this policy with a permutation-calibrated
shrinkage haircut before the banking window opens; the banking mechanism prong
counts KAGGLE-side replay successes, not local ones (15/25 local engine versions
differ). Full-panel sign-off requirement (A9 adjacency) unchanged.

## A17 — war-v4 capability screen (3× MAJOR; pre-Aug-1, blocking)

Before the Aug 1 v4 registration: run Qwen3.6-72B-tier 4-bit under the IDENTICAL
harness on ft09/sb26/lp85/vc33 on the free Kaggle GPU build rail (30 GPU-h/wk),
with a measured tokens/s bench. Pre-registered go/no-go: **GO iff ≥2 levels beyond
the 27B baseline summed across the 4 games at full per-game budgets AND measured
throughput sustains ≥90% of the 27B action count under the binding budget** (the
binding budget is wall-clock on the scored rail; the expected 72B action count must
be computed from the measured tokens/s before registration). Throughput-adjusted
null (formula, closing the undefined else-branch): for each game, null_adj = the
levels the 27B baseline had completed by action N₇₂B, where N₇₂B = measured 72B
actions achievable in the wall-clock budget; 72B must beat Σ null_adj. NO-GO
finding ("72B replicates the ~1-level grinder profile") goes to the panel
immediately — the campaign would then have no registered wall-closer, and the
panel decides in July, not September.

## A18 — (d) offline prediction-accuracy metric (prog-synthesis MAJOR; before W1 seals)

The no-effect-FACT recurrence accuracy is being computed today on the N5
deterministic replay traces (`scripts/predict_metric.py` → `runs/predict_metric/`).
Pre-registered threshold, sealed before results are observed: (d)'s window
proceeds iff **recurrence accuracy P(no-effect again | prior no-effect observed) >
majority-class baseline** AND **trigger opportunities ≥1/run on ≥5 games**.
Otherwise (d) is killed cheaply now and W1 becomes (a)'s window.

**RESULT (observed ~14:30 EDT, after seal): (d) KILLED.** Pooled over 175
game-runs / 29,487 actions across 7 pulls (board_changed label integrity-verified
by independent frame hashing, 0/29,487 disagreements): recurrence accuracy 0.465
(Wilson 95% [0.436, 0.494]) at state_action granularity vs majority baseline
0.903 — decisive fail on the accuracy prong (trigger coverage passed, 68/175
runs). A recurring "no-effect" (state, action) pair actually changes the board
~54% of the time on these near-deterministic engines: the FACT rule would be
actively wrong most times it fired. Per the sealed rule, W1 becomes (a)'s window;
(c)'s disposition (formerly one flag with (d); forfeits standalone window under
the MDE/2 rule) goes to R15. Artifacts: `runs/predict_metric/{report.md,raw.json}`,
`scripts/predict_metric.py`.

## A19 — (c) mechanism prong upgrade (llm-agents minor N4)

"Verbatim-resubmit count = 0" is demoted to a code-shipped check (it is guaranteed
by the block's presence). The value-bearing observable added to (c)'s mechanism
prong: **post-block novel-family rate** — fraction of hard-block events followed
within 10 actions by an action outside the blocked family. Prong threshold: rate
> 0 on ≥3 games (the CONCEPT-lock finding predicts 0; this is the falsifiable bet).

## A20 — Declarations and process fixes

- **(g) per-game budget re-allocation is DEAD** (explicit resolution-by-removal;
  rl-planning/prog-synthesis demanded the declaration).
- **0.56× rail→LB conversion is an ASSUMPTION**, not a derived constant: all
  LB-unit claims carry a 0.4×–0.8× sensitivity band (llm-agents N5, methodology
  minor). §2's LB figures are so relabeled.
- **Circulation rule:** panel-facing documents must fit the reviewer argv budget;
  the full design doc + this amendment go to R15 in ≤2 parts with per-part sha256
  and untruncated END lines, before the recalibrated gate seals.
- **Timestamps (methodology Q3):** amendment 2026-07-18 (A8–A13) filed ~08:00 EDT
  Jul 18; this amendment ~13:30 EDT Jul 18; LB draw #5 submits ~20:07 EDT Jul 18
  (00:07Z Jul 19) and scores ~04:00 EDT Jul 19. Both amendments precede the draw's
  observation.
