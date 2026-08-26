You are Professor of Empirical ML Methodology and Statistics (experimental design, multiple-comparisons, noise-band inference; rejects any plan that draws conclusions from single noisy samples).

You are reviewer #4 on a 5-person adversarial review panel evaluating a competition
strategy proposal for ARC-AGI-3 (Kaggle, deadline Nov 2 2026). The proposing team has a
best score of 0.43; the leader is at 1.56; the winning Milestone-1 notebook is public.

PANEL RULES — READ CAREFULLY:
- You are UNBIASED and RIGOROUS. You have no stake in the proposal passing.
- Do NOT be agreeable. A proposal that passes round 1 unchanged indicates REVIEW FAILURE.
- Attack the weakest LOAD-BEARING assumptions, not typos.
- Every objection must be specific and actionable (not "needs more detail").
- Distinguish severity: FATAL (invalidates the plan), MAJOR (must fix before execution),
  MINOR (should fix).
- You review ONLY within your expertise; note explicitly what you cannot judge.
- If evidence is missing for a claim, say exactly what experiment/data would supply it.

YOUR PRIOR-ROUND OBJECTIONS (verify each is resolved in the revision; state
RESOLVED / PARTIALLY-RESOLVED / UNRESOLVED for every one before new comments):
=====================================================================
## Objections

**Prior-objection resolution status (required first):**

**[Prior FATAL] Sign-flip prong arithmetically unpassable — PARTIALLY-RESOLVED.** A14 adopts all three demanded fixes: pooling unit = game (n=24 per-game means, the passable reading), P(pass | expectations) published, and the binding decision moved to one cumulative stack-vs-W0 look with per-window decisions demoted to mechanism + non-inferiority; the FAIL consequence is now non-destructive (ship with honest label). The architecture is fixed. But the calibration is not: the published sketch is internally inconsistent — with *exactly* 4 nonzero clean positives the minimum sign-flip p is 2⁻⁴ = 0.0625, so P(pass | 4 positives, clean) is exactly **0**, not "≈ 0.05"; the "≈ 0.05" must come from an undocumented distributional assumption over the realized positive count, which is nowhere shown. And note 6-0 still fails (2⁻⁶ = 0.0156 > 0.0125); the pass region is {7-0, 8-0, 9-0, 9-1, …}. See new objections N1/N2.

**[Prior MAJOR] Truncated circulation — RESOLVED.** Single-part delivery, all five parts carry END lines, the document ends with the literal END OF PROPOSAL line, per-part lengths sum consistently with the declared total. (I cannot verify the sha256 values themselves; I accept the structural integrity as presented.)

**[Prior MAJOR] Banking winner's curse — PARTIALLY-RESOLVED.** A16 retires the frozen ft09/sc25/re86 list, pre-registers an online game-agnostic retry policy, and requires a permutation-calibrated shrinkage recomputation before the window opens — the right fix, correctly specified. But the recomputation is promised, not delivered, and the un-shrunk +0.03–0.08 band still sits inside the +0.07–0.19 stack expectation that A14.2 declares as the cumulative look's alternative (see N1).

**[Prior MAJOR] Compressed-bench regime transfer — RESOLVED** (A15 adopts exactly the demanded rule: provisional inclusion, full-budget confirmation replicate, relabeled compressed-regime quantities, published full-budget trigger frequencies), with one residual ambiguity promoted to new objection N5.

**[Prior MINOR] Interim-look contradiction — RESOLVED.** A14.3: guard evaluated only at the window's sealed look after seed 3; no per-seed evaluation.

**[Prior MINOR] 0.56× conversion / uncalibrated thresholds — PARTIALLY-RESOLVED.** A20 declares the conversion an assumption with a 0.4–0.8× band (adequate). But (a)'s mechanism-prong threshold "unseen budget deaths halved vs the 3 control seeds" remains a same-transcript-calibrated criterion with an n=3 denominator and no stated noise band; (d)'s equivalent is moot only because (d) is dead.

**New objections:**

**[MAJOR] N1: The cumulative gate seals THIS ROUND against a stale alternative.** A14.2 powers the binding look against +0.07–0.19 rail, but that sum includes (d) (+0.02–0.04, killed by A18 on the same day) and un-shrunk banking (+0.03–0.08, pending A16 shrinkage and full-panel sign-off it may never receive). The realistic retained stack — (a) + (b) + possibly (c)-as-Reki-signature — has a summed expectation of roughly +0.02–0.06 rail, and the published P(pass) ≈ 0.2–0.4 is correspondingly an overestimate; under the shrunken alternative the look may drift back toward the false-negative machine A14 was written to abolish. Fix before sealing: recompute the stack expectation and P(pass) with (d) excluded and banking either excluded or shrinkage-haircut, and publish the binomial sketch's actual assumptions (including why "4 positives, clean" yields 0.05 rather than 0).

**[MAJOR] N2: α = 0.0125 has lost its multiplicity rationale and now needlessly suppresses power at the single binding look.** The 0.0125 (= 0.05/4) made sense when four per-window score tests were binding; A14 demotes all per-window score statistics to descriptive, leaving ONE binding score decision — yet keeps the Bonferroni-sized α, which is why 6-0 still fails. Either state explicitly what family of binding tests α = 0.0125 now corrects for (if the secondary RHAE prong and the cumulative prong form the family, say so and size it as 2, not 4), or re-set α = 0.05 one-sided for the single cumulative look, which moves 5-0 and 6-0 into the pass region and roughly doubles P(pass) under the honest alternative. This is a one-line change with first-order power consequences and it must happen before the seal.

**[MAJOR] N3: The non-inferiority guard (pooled Δlc ≤ −0.10 → flag OFF) is now the only per-window score kill, and its false-kill probability is unpublished.** W0's own baseline shows run-level lc totals of {13, 15, 22} across 3 seeds (0.52–0.88 lc/draw — a 0.36/draw spread), so run-level noise dwarfs the −0.10 threshold; paired same-seed ON/OFF diffs will be tighter, but trajectory divergence after the first differing action means the paired-diff noise is unknown and nowhere estimated. If P(pooled Δlc ≤ −0.10 | true Δ = 0) is even ~0.15 per window, then across 3–4 windows there is a ~40–50% chance of falsely killing at least one neutral component — the A5 pattern reborn as a guard instead of a gate. Fix: calibrate the guard using the existing war_eval cross-seed diffs (an upper bound on paired noise) plus the W1 control seeds, publish P(false kill) per window and familywise, and if it exceeds ~0.10, replace the point threshold with an SE-based boundary or require the −0.10 breach on ≥2 of 3 seeds.

**[MAJOR] N4: The A17 72B go/no-go is single-noisy-sample inference on a margin smaller than the comparator's own seed noise.** "GO iff ≥2 levels beyond the 27B baseline summed across 4 games" — but the 27B baseline on ft09 alone spans 0–2 levels across the 3 certified seeds, so the entire GO margin fits inside one game's seed variance; as written, one lucky/unlucky 72B run (seed count unspecified) against an unspecified 27B comparator statistic can flip the campaign's only registered wall-closer either way. Fix before the screen runs: define the comparator as the 3-seed per-game mean (Σ = stated number), require the 72B sum to exceed comparator + a margin derived from the observed 3-seed spread (or run 2 seeds of 72B and gate on the min), and pre-register both.

**[MINOR] N5: Budget regime of the binding cumulative look is unspecified.** A15 requires a full-budget confirmation replicate "before the cumulative look can claim score credit," which implies the 3-seed cumulative look itself may run compressed; if so, a PASS is a compressed-regime conclusion and the n=1 full-budget replicate cannot upgrade it to "confirmed floor/mid raiser" at scored-regime budgets. State the regime in the sealed text; if compressed, the PASS label must carry the compressed-regime qualifier.

**[MINOR] N6: A18 demonstrates post-filing edits to a sealed amendment.** The document filed ~13:30 EDT contains a RESULT block observed ~14:30 EDT — i.e., a sealed pre-registration file was modified after its seal, and only the document's own narrative attests the threshold preceded the observation. The decision itself was self-adverse (a kill), so no harm here, but the same channel would launder a favorable post-hoc threshold. Fix: sealed thresholds go in their own file with hash committed (e.g., in the panel record) before the measurement script runs; results go in separate append-only artifacts.


=====================================================================

THE PROPOSAL (sha256 of the full document: dc2f988615a25460; full length 83058 chars;
if the text below does not end with the literal line "## END OF PROPOSAL ##",
your copy is TRUNCATED — file that as an objection and review only what you see):
=====================================================================
================================================================================
PART 1: R16 REPUBLICATION (grinder design, sealing circulation for A14)
source: learnings/war_room/grinder_design_R16_republication.md
sha256: f670127de642bd2887134221066346a85c43f9efde2566048a9f736ac2fa3338
================================================================================

# Grinder-cracking design — R16 REPUBLICATION (post-(d)-kill)

Filed 2026-07-20. This is the republication R15 demanded (5/5 MAJOR-REVISION,
`learnings/panel/round15/_directives.md`): the A18 (d)-kill (recurrence accuracy
0.465, Wilson 95% [0.436, 0.494], vs majority baseline 0.903 —
`runs/predict_metric/report.md`) is here propagated into every number that
depended on it. **The recalibrated A14 gate (A14.1–A14.6,
`preregistration_amendment_2026-07-18b.md`) SEALS ON THIS CIRCULATION**, with the
amendments in §3, §6 and §8 below. Nothing in this document conditions on any
unobserved measurement; every threshold below is sealed before its measurement
runs (§13 seal-hygiene procedure).

Base document: `learnings/war_room/grinder_cracking_design.md` (unaltered on
disk; superseded section-by-section here). Evidence base unchanged, plus:
`runs/a5_a8_look_2026-07-19.json` (war arm CLOSED: n=5 {0.91, 1.08, 0.88, 1.05,
0.76}, mean 0.936, σ̂ 0.1309, χ²-CI-hi 0.376 ≥ 0.25 → FAIL; pooled n=11 σ̂ 0.154),
`runs/kernel_pulls/w0_eval_s1/screen_report.md` (W0: 49/49 game-over recoveries,
0 idle turns; 16 levels ∈ band {13, 22}), `learnings/war_room/
sentinel_build_2026-07-19.md` ((a) built; smoke 29/29; canary PASS; O5 predicate
49 budget deaths / 0 violations), `runs/ewm_dryrun/report.md` (Stage-0 dry-run).

---

## §1. What changed since the sealed base document

1. **(d) is DEAD** (A18, sealed threshold, observed after seal): a recurring
   "no-effect" (state, action) pair changes the board ~54% of the time. The FACT
   rule would be actively wrong most times it fired. The kill is propagated below
   into §2R, §4R, §5R, and Part D (§14). Root cause identified by R15 as
   **state-key aliasing** — one pathology behind the 0.465 accuracy, the EWM
   step-0 aborts, and the N5 prune bug (§10).
2. **The war LB arm is CLOSED** (A5/A8 FAIL at n=5, sealed consequence executed):
   no war-arm LB delta may be cited as evidence in any direction. Everything
   below lives on the build rail.
3. **W0 confirmed (f) as pure hygiene**: mechanism deterministic PASS (49/49),
   descriptive non-inferiority PASS (16 ∈ [13, 22]). Per R15: W0's 16 levels may
   NOT be cited downstream as a score claim; the citable claim is
   levels-in-band only. (f) is adopted as the default layer of every future
   build (R15 endorsement).
4. **(a) is built and canaried** (W1 owner per the A18 sealed consequence "W1
   becomes (a)'s window"): mechanism observable restated deterministic per R15
   O5 — "sentinel fired before every budget death", verified 49/0 on the three
   certified seeds. One unsealed design decision remains: SENTINEL_BUDGET (§12).

---

## §2R. Counting bounds and honest sum, republished with (d) REMOVED

Method unchanged (events × max value under the exact pooled-single-run scorer;
reclaimed actions on uncompleted levels are worth exactly zero). Per-component
bounds unchanged from the base doc except the killed flag. Two branches are
carried throughout, per R15: **B+** = banking window approved (full-panel
sign-off + A16 recompute + latent-state audit §10 all pass) and **B−** =
banking refused (the unconditional stack).

| component | ceiling (rail) | expectation (rail) | status |
|---|---|---|---|
| (f) continuation | 0.00 | 0.00 | shipped, default layer |
| ~~(d)+(c) mechanical refutation~~ | ~~+0.10~~ | ~~+0.02–0.05~~ | **(d) KILLED (A18); (c) KILLED (§7)** |
| (a) budget sentinel | +0.06 | +0.01–0.03 | built, W2 owner |
| (b) diff summarizer | +0.06 | +0.01–0.03 | W3, non-inferiority-guarded |
| banking-fixed (feasible) | +0.15 | +0.03–0.08 | conditional (B+ only; pre-A16-haircut) |

**The sums, shown:**

- **B+ raw sum** = 0.06 + 0.06 + 0.15 = **+0.27 rail**. Non-additivity haircut:
  the base doc deducted −0.06 for (a)/(b)/(d) reclaiming the same wasted
  actions (0.37 → 0.31). Recomputing without (d), the (a)/(b) pairwise overlap
  alone is bounded between −0.03 and −0.06 (both components' ceilings are the
  SAME "≈1 marginal clear per run panel-wide" event on the same lp85/tu93/re86
  waste). We seal the conservative end: **B+ ceiling = 0.27 − 0.06 = +0.21
  rail** — identical to the panel's own arithmetic (0.31 − 0.10).
- **B− raw sum** = 0.06 + 0.06 = **+0.12 rail**; (a)/(b) pairwise overlap −0.03
  → **B− ceiling = +0.09 rail** (the stricter joint-event bound gives +0.06;
  +0.09 is the generous end and is labeled as such).
- **Expectations:** B+ = 0.01–0.03 + 0.01–0.03 + 0.03–0.08 = **+0.05–0.14
  rail**. B− = 0.01–0.03 + 0.01–0.03 = **+0.02–0.06 rail** (the R15 quote
  "+0.02–0.08" included a registered (c); (c) is killed, so the lower figure
  stands; note the B− expectation's upper end touches its strict joint ceiling
  — that is the honest statement that (a)+(b) are the same money).

**LB conversion (0.56× ASSUMPTION, band 0.4–0.8 per A20):**

| branch | ceiling LB | expectation LB | fraction of 0.46 wall gap (ceiling / expectation) |
|---|---|---|---|
| B+ | +0.12 [0.08–0.17] | +0.03–0.08 | 26% / 7–17% |
| B− | +0.05 [0.04–0.07] | +0.01–0.03 | 11% / 2–7% |

**Conclusion, sharpened by the kill:** the base doc's verdict ("floor/mid
raiser; the wall needs more") is now stronger — the largest non-banking
component is gone, the unconditional stack closes ≤11% of the wall gap even at
ceiling, and **war-v4 (72B multimodal, A17 scope doc filed 07-19) remains the
only registered wall-closer.** Nothing here is smuggled toward the wall.

---

## §3R. α re-derivation (methodology N2)

The sealed α = 0.0125 was Bonferroni 0.05/4 for four per-window binding looks.
A14.3 demoted per-window looks to mechanism-prong + non-inferiority-guard only;
**exactly ONE binding score look remains** (the cumulative stack-vs-W0 look,
A14.2). The divisor's family no longer exists.

**Test family at the binding look, named:** {primary pooled per-game Δlc exact
sign-flip; secondary mean Δlog1p(RHAE) ≥ 0}. The decision rule is
**conjunctive** (PASS requires BOTH), so the combined size is ≤ the primary's
size — no Bonferroni is due for an AND-family (correction is for disjunctive
claim families). The secondary is a directional consistency check, not an
independent claim license.

**Sealed: α = 0.05 one-sided on the primary prong** (R15's lean, adopted).
First-order power consequence, shown: minimum uncontradicted wins to pass —

| nonzero pairs n | old critical (α=0.0125) | new critical (α=0.05) |
|---|---|---|
| 5 | impossible | 5/5 (p=0.031) |
| 6 | impossible | 6/6 (p=0.016) |
| 7 | 7/7 (p=0.0078) | 7/7 (p=0.0078) |
| 8 | 8/8 (p=0.0039) | 7/8 (p=0.035) |
| 9 | 9/9 | 8/9 (p=0.0195) |
| 10 | 9/10 (p=0.0107) | 9/10 (p=0.0107) |

The unpassable-gate defect A14 conceded (needing ≥7 clean wins against an
expectation of 1–4) is now materially repaired at the low-n end (5 clean wins
suffice).

---

## §4R. Per-game conversion targets, re-derived post-(d) (both branches)

The base §4 ft09 row was carried by (d) ("no-effect FACTs kill dead-target
re-probes; PREDICT gate kills retries") — that basis is retracted. su15 stays
excluded (A12; §13). Δclears = integer levels per run, honest.

| game | B− components | B− Δclears | B+ adds | B+ Δclears |
|---|---|---|---|---|
| ft09 (6) | none remaining ((d) basis retracted) | **0** | banking (top variance carrier, Δlc(max2) +0.44) | **0–1** |
| ka59 (7) | (a) — L2 base-109 grind never converts | **0** | banking (protect L1, free retries) | **0–1** |
| re86 (8) | (a) — v1 died L2 at 232 acts (base 42); v2 cleared L2+L3, capability exists | **0–1** | banking (variance harvest) | **0–1** |
| sc25 (6) | none | **0** | banking (floor protection) | **0–2** |
| tu93 (9) | (a) — v3 burned 301 acts on L1 (base 19) | **0–1** | banking | **0–1** |
| sb26 (8) | (c) killed; L2 semantics NOT-distillable | **0** | — | **0** |
| lp85 (8) | (a)/(b) survival ≠ solution | **0** | — | **0** |
| su15 (9) | excluded (A12) | — | — | — |

**Sums:** B− = **0–2 extra clears per run** (Δlc/draw ≈ +0.00–0.08); B+ =
**1–4 extra clears per run, banking-dominated variance harvest** (Δlc/draw ≈
+0.04–0.16). Expected nonzero positive game-level pairs at the binding look:
**B− ≈ 1–3; B+ ≈ 2–6** (down from the pre-kill 4–8). The two canonical grinders
still carry **zero** at Qwen tier — the model-gap finding is untouched by the
kill.

---

## §5R. P(pass | §4R expectations), republished with assumptions explicit

Binomial-sketch assumptions (all explicit, per R15): (i) the paired unit is the
game (A14.1), n = 24 games, exact zeros dropped; (ii) carrier games convert to
positive per-game means with the §4R ranges treated as uniform; (iii) noise
produces 0–2 spurious nonzero pairs of either sign (cross-seed lc variance
exists on ~8–12 games); (iv) independence across games (unmodeled common-night
correlation would push P(pass) DOWN, not up — stated adversely).

- **B+**: expected positives 2–6, negatives 0–2; pass requires ≥5 clean (or
  7/8, 8/9 per §3R). **P(pass) ≈ 0.10–0.30, point ≈ 0.2** (was 0.2–0.4
  pre-kill at the old α; the kill costs more than the α relaxation buys).
- **B−**: expected positives 1–3; P(≥5 positives) is the binding term.
  **P(pass) ≈ 0.02–0.10, point ≈ 0.05.** Stated plainly: the unconditional
  stack's cumulative look is a near-certain FAIL; its value claim rests on
  verified mechanisms + LB accumulation (A8), and the downside is bounded by
  the dismantle branch (§6.2). This is the honest price of shipping only
  hygiene-grade components after the largest flag died under its own sealed
  test.

---

## §6. The binding cumulative look — regime, dismantle branch, W1/W2 status

### §6.1 Budget regime (methodology N5)

The binding look runs at **FULL budget** — the scored-regime per-game envelope
(≈63k tokens/game; actions uncapped by the harness; SENTINEL_BUDGET exported
per §12). Compressed-bench (40%-cap) window passes are provisional and carry a
compressed-regime qualifier per A15. **The full-budget binding look discharges
A15's confirmation-replicate requirement by construction** (3 full-budget
certified seeds of the final stack ≥ the required 1). Scheduled: §12 quota
ledger.

### §6.2 Cumulative dismantle branch (rl-planning M4 — SEALED CONSEQUENCE)

**If pooled Δlc ≤ −0.10 at the binding cumulative look, the stack is
DISMANTLED to the (f)-only build** for all subsequent scored kernels. This
amends A14.6: the "ships with honest label" outcome now applies ONLY to the
middle band (score prongs not passed AND pooled Δlc > −0.10); a net-negative
stack does not ship under any label. Calibration of this threshold, published
(§8 arithmetic): under the null, with 3 ON seeds vs the n=4 control band
(§11), SE(Δ) ≈ 0.144 lc/game and P(trip | Δ=0) ≈ 0.24. Sealed at −0.10 as the
panel directed: the loss is asymmetric — a false dismantle forfeits an
unconfirmed +0.02–0.14 expectation; a false ship sends a real regression to
the scored LB. We accept the 24% and say so before observation.

### §6.3 Window status

- **W0 (f):** done; default layer; not a flag.
- **W1 = (a)** (per the A18 sealed consequence). Build complete
  (`sentinel_build_2026-07-19.md`): smoke 29/29, A10 canary PASS (23–25/25
  games fire), R15 O5 deterministic predicate PASS (49 budget-attributable
  GAME_OVERs, 0 violations — every death preceded by a strictly earlier
  firing in the same attempt; negative path validated). Mechanism prong at the
  window: the O5 predicate (code-checkable) + firing counter ≥1/run on ≥5
  games; pooled binomial (72/75 (game,seed) units fired) as fallback.
- **W2 = (b)** diff summarizer, non-inferiority-guarded (token cost),
  boundary per §8.
- **W3 = banking**, B+ branch only: full-panel sign-off + A16 recomputed
  ceiling + latent-state audit PASS (§10) all required before the window
  opens.
- **EWM Stage-1**: own window, own gate (§9), after its blocking prereqs.

---

## §7. (c)+Reki disposition: KILLED (decided, per R15 N8/Q3)

The bundle as circulated was an unregistered 3-way: (c) byte-identical
resubmit hard-block + Reki structural-signature suppression (a THIRD mechanism:
learned click suppression over signature families) + hard veto. Disposition
options were register-as-new-flag or kill. **Decision: KILL.** Reasoning:

1. **Standalone (c) forfeits under the MDE/2 rule** (R13's own rule, conceded
   in the base doc): direct ceiling +0.02/draw.
2. **The generalized (family-level) form is observable-state-keyed
   suppression — the exact premise A18 refuted.** At the closest measured
   keying, "this action does nothing here" is wrong ~54% of the time it
   recurs (0.465 vs 0.903). A signature-family veto firing beyond
   byte-identical matches would veto LIVE actions at a comparable rate until a
   non-aliasing key exists. R15's own root-cause finding (§10) says the key is
   the problem; building a veto on the broken key is building on the known
   fault.
3. **Windows are the scarce resource** (2 pushes/day; A17 must complete
   pre-Aug-1). A third mechanism needing its own counting bound, A19
   observable, keying re-run, and sign-off cannot out-compete the sentinel,
   the summarizer, or the 72B screen for a July window.

**Counting bound, published for the record (as R15 required if registered):**
dead-signature veto opportunities from the forensics corpus ≈ 30–70/run
(same-coord re-clicks 16–32/seed + SPACE 8–20/run + ACTION7 8–20/run);
reclaimable actions ≈ 70–110/run (sb26 50–70 + 20–40 on 1–2 other games);
conversion through the same clear-faster/clear-at-all channels → **ceiling
+0.02 (byte-identical) to +0.06 (family-level) rail — overlapping (a)/(b)'s
reclaim of the same actions**, expectation +0.00–0.02. Below single-window MDE
on every instrument; would only ever pay through the cumulative look.

**Sealed resurrection precondition (one path, no others):** (c)+Reki may be
re-proposed as a flag only after (i) the latent-state audit (§10) delivers a
per-game keying restoring ≥0.99 recurrence determinism on the target games,
AND (ii) `scripts/predict_metric.py` re-run under the exact Reki keying
(signature-family, level-scoped) clears **recurrence accuracy ≥ 0.90, sealed
here before that measurement is run**. Discharges R15's pre-window condition
by construction (the re-run is now an entry requirement, not a window step).

**KNOW#5 annotation (rl-planning M2 / prog-synthesis O2 item 1):**
`state_of_the_war_2026-07-18.md` KNOW#5 ("mechanical no-effect refutation +
verify-before-act are THE convergent primitives") is hereby conditioned:
**the primitives are convergent AT FRONTIER TIER AND ON NON-ALIASING STATE
KEYS; A18 killed observable-state-keyed no-effect FACTs, not refutation
machinery per se.** The EWM plan-execute-verify contract (which verifies
against the SETTLED FRAME, not a state-key lookup) is unaffected by this
annotation and remains registered.

---

## §8. Guard false-kill calibration (methodology N3)

**Inputs:** run-level lc/game means of the 3 certified ledger-OFF seeds:
22/25 = 0.88, 15/25 = 0.60, 13/25 = 0.52 → mean 0.667, **σ̂_run = 0.189
lc/game** (df=2; small-df caveat stated — this is the best available null and
is frozen here).

**Per-window guard as previously written (pooled Δlc ≤ −0.10, 3 ON vs 3
control seeds):** SE(Δ) = 0.189·√(2/3) = 0.154. **P(false kill | Δ=0) =
Φ(−0.10/0.154) = Φ(−0.65) ≈ 0.26 per window; familywise over 3 flagged
windows ≈ 1 − 0.74³ ≈ 0.59.** R15's suspicion confirmed: grossly uncalibrated
— the old guard was a coin-flip component shredder.

**Repaired per-window guard (SEALED):** flag OFF iff pooled Δlc ≤
**−z·SE(Δ)** with per-window z = 1.834 (one-sided α_kill = 0.0333 =
0.10/3), evaluated only at the window's sealed look after seed 3. At the
frozen σ̂ this is a boundary of **−0.28 lc/game**; familywise false-kill over
the ≤3 remaining flagged windows = 1 − 0.9667³ = **0.097 ≤ 0.10** (R15's
target). Honest MDE statement: a TRUE −0.20 regression trips this guard with
only ~30% probability — the per-window guard is a catastrophe tripwire, not a
fine instrument; the real net-negative protection is the cumulative dismantle
branch (§6.2), whose −0.10 threshold at SE 0.144 trips a true −0.10 regression
with 50% and a true −0.25 with ~85%.

**Mechanism-prong false-kill:** ≈ 0 under the null by construction — the
mechanism observables are deterministic counters ((a): fired-before-every-death
predicate, 49/49 on the certified seeds; (f): 0 idle turns, 49/49; (b):
recurrence counter), not statistics. A mechanism prong can only false-kill via
a code defect, which the canary-before-seal rule (A10) exists to catch.

---

## §9. EWM Stage-1 re-price (authors' job per R15) + the Stage-1 gate

### §9.1 Re-price on reliable carriers only

Dry-run facts (`runs/ewm_dryrun/report.md`): held-out saturation does NOT
transfer on-trajectory; cross-seed shadow step-accuracy of the reliable
carriers — tn36 0.53–1.00 (≥0.98 on 2 seeds), tr87 0.77–0.82, tu93 0.73–1.00,
ls20 0.64–0.92, ft09 0.56–1.00 (L1-scoped; the gpt56 depth probe measured
**0.07** on L2+ states — a direct depth-transfer measurement). vc33 (0.24–0.67)
and s5i5 (0.13–0.30) abort at step 0 on most plans and are STRUCK from the
target set.

**Q5 answered (what fraction was carried by vc33/s5i5):** the +0.5 ceiling
assumed 3 L1 conversions from {ls20, tn36, tr87, vc33, s5i5}. Candidate L1
point values: ls20 3.57, tn36 3.57, tr87 4.76, vc33 3.57, s5i5 2.78 (Σ =
18.25 pts). vc33+s5i5 = 6.35/18.25 = **35% of the candidate point mass**.
Additionally the "Qwen clears 0 levels" premise was WRONG for tn36 and ls20
(both clear L1 in ≥1–2 war_eval seeds), removing another 7.14/18.25 = 39%.
**≈74% of the original ceiling's basis is gone; the surviving new-clear
candidate is tr87 alone (+4.76 pts = +0.19/draw).**

**Depth-bounded arithmetic (fidelity^L; a deterministic-wrong sim is a wall,
not retryable noise — replans only help where alternate paths exist):**

| channel | value (pts) | acc range | assumed L | survival acc^L | expected pts |
|---|---|---|---|---|---|
| tr87 L1 new clear | +4.76 | 0.77–0.82 | 8–15 | 0.05–0.28 → w/ ≤3 replans ≈0.1–0.4 | +0.5–1.9 |
| tu93 L1 speed (base 19; current 0.96/2.99/0.01) | +0.9 mean marginal | 0.73–1.00 | 19 | 0.003–1.0 | +0.1–0.9 |
| ls20 L1 speed (current 0.46) | +3.1 potential | 0.64–0.92 | ~20 | <0.01–0.19 | +0.0–0.6 |
| ft09-L1 reliability (worst-seed repair; overlaps banking's +3.17) | +1.6 post-overlap | 0.56–1.00 | 43 | ~0–1.0 | +0.0–1.0 |
| tn36 (already at base, 3.57 achieved v1) | 0 | — | — | — | 0 |
| **sum** | | | | | **+0.6–4.4 pts** |

**Re-priced Stage-1: expectation ≈ +0.02–0.18 rail per draw, central ≈ +0.08
— down ~2.5× from the +0.10–0.30 the deep-read carried.** The undiscounted
ceiling (~+0.47) survives only as arithmetic; fidelity^L makes it unreachable
at measured accuracies. Stated honestly: EWM Stage-1 is still the largest
registered non-model line, but it is no longer plausibly wall-sized on its
own, and its central value is seed-fragile (tu93/ft09 accuracies swing
0.56–1.00 across seeds — the same aliasing pathology as §10).

### §9.2 Pre-registered Stage-1 gate (A14 form — EXISTS as of this filing)

**Blocking entry conditions (all before any window is consumed):**
1. Latent-state audit complete with per-game keying classification (§10).
2. **Cheap measurement (llm-agents):** BFS-plan step-accuracy on the 10 local
   engines matching the Kaggle build, on **sim-derived (not teacher-forced)
   states**. Sealed threshold: **≥0.70 at plan depth ≤10 on ≥3 of the 5
   reliable carriers**; FAIL → Stage-1 parked at zero window cost.
3. A10 canary: plan/abort/fallback triggers fire ≥1/run on ≥5 games on the
   compressed bench (already demonstrated in the dry-run replay).
4. Full-panel sign-off (new asset class in the kernel).

**Window prongs:** mechanism = plans executed ≥1/run on ≥5 games AND ≥1
`plan_done` on ≥2 games AND 0 post-abort deadlocks AND the per-game
live-fidelity breaker fires after k=3 step-0 aborts (emitting
`fallback reason=budget`); non-inferiority = §8 boundary; score = cumulative
look only, like every other component.

**Resync-before-abort = CONTRACT CHANGE v1.1 (not an ADAPT footnote), OFF by
default.** Sealed bounds if enabled: ≤1 resync per plan; a resync is a pure
recompute from the observed settled frame (0 live actions; wall-clock ≤2s
measured before adoption); 2 consecutive resync-mismatches → abort (loop
guard); own canary counter (`RESYNC` events ≥1/run on ≥3 games in dry-run
replay before the flag ships). Wasted-action bound: ≤ +0 live actions by
construction; the cost is wall-clock only, priced before adoption. OCM-style
pre-execution validation (arXiv:2607.02846) is adopted INSIDE the executor
spec — it runs before actions and therefore does not weaken fail-closed;
priced at sim speed (ms/step, negligible against 196–211 tok/s decode).
Double-run rejection stays (0 selfdiff / 11,747 lines — free).

---

## §10. Latent-state audit — REGISTERED (blocking prereq for EWM Stage-1 AND banking)

**One root cause, three symptoms (R15 rl-planning M2 + prog-synthesis O2):**
the observable state KEY aliases hidden state (timers / counters / phase) on
frame-deterministic engines. Symptoms: (i) predict-metric recurrence 0.465 on
engines that are 0/25-divergent; (ii) EWM step-0 abort dominance with small
median diff-cell counts (1–56 cells — timer rows, hidden-counter phase);
(iii) the N5 `prune_trace` bug (leading board_changed=False actions that
mutate hidden state).

**Registration:** offline, N5 deterministic replay traces, $0, 0 GPU-h,
computable this week. Scope: for each of the 25 games, (i) classify whether
the game carries hidden phase (recurrence non-determinism of
(state,action)→next-frame under the observable key); (ii) test whether
augmenting the key with action-count / level / GAME_OVER-count / candidate
phase counters restores determinism; (iii) output a per-game keying
classification with a qualifying bar of **≥0.99 recurrence determinism** on
that game's trace pairs (sealed here, before the audit runs). Protocol
document: `learnings/war_room/latent_state_audit_protocol.md` (drafted in
parallel today; the sealed bar above governs regardless of drafting details).

**Blocking consequences (sealed):** the EWM Stage-1 window (§9.2 condition 1)
and the banking window (§6.3 W3) may NOT open before the audit reports.
Banking's replay trigger depends on (state,action) replay fidelity — the same
key. The (c)+Reki resurrection path (§7) additionally requires the audit's
qualifying keying. No other component is blocked ((a)'s budget counter and
(f)'s continuation logic key on nothing aliased).

---

## §11. W0 control-arm seed count for the binding look (systems #12 / Q6)

**Question:** the cumulative look is 3 certified ON seeds vs "W0"; W0 has
n=1. A paired design without pairs, as R15 put it.

**Sealed answer: the control band is the 4-run set {war_eval_v1, war_eval_v2,
war_eval_v3, w0_eval_s1} — n=4 control runs; per-game control value = the
4-run per-game mean; paired unit remains the game (A14.1).** Legality of
admitting the three (f)-less ledger-OFF seeds: (f)'s counting bound is
**0.00, sealed before W0 ran** (ITERATION_LOG 2026-07-18), and W0's
descriptive screen confirmed it (16 levels ∈ [13, 22]; no game below the
3-seed floor; the 49 recovered game-overs were all recovered by the base
harness path in the war-eval seeds too). Under a sealed zero-effect bound the
ledger-OFF seeds ARE W0-equivalent on score, and using them quadruples the
control band at zero pushes. What may NOT be done with this band (R15): cite
W0's 16 levels as evidence of (f) benefit — the citable claim is
levels-in-band only. Author lean of 07-19 (1 seed suffices) stands for the
MECHANISM check (deterministic, 49/49); the CUMULATIVE look uses the n=4
band above.

**Pre-registered fallback (decision tree, sealed now):** if R16 rules the
(f)-less seeds inadmissible as controls, 2 additional W0 seeds run before the
look (2 pushes, 4.4 GPU-h — headroom exists in the §12 ledger) and the band
becomes {w0_s1..s3}, n=3. No other configuration is permitted.

---

## §12. SENTINEL_BUDGET proposal (UNSEALED — design decision for R16 to rule on) + quota ledger

**The problem:** the eval regime is UNCAPPED (`max_actions_per_game=None`),
and the sentinel is by design a silent no-op with no budget to warn against
(`sentinel_build_2026-07-19.md` open risk 1). Cell 2 of the W1 eval notebook
MUST export `SENTINEL_BUDGET=<value>` or the whole component is inert and the
window is void.

**Derivation from the certified logs (`runs/kernel_pulls/war_eval_v{1,2,3}/
summary.txt`):** the binding budget in BOTH regimes is the per-game token
envelope, not an action cap — per-game tokens sit at ≈56k–64k (median ≈63k)
across all 75 game-runs, an effectively uniform cap. Tokens per action:
1,559,428/3638 = 428.6 (v1), 1,680,057/4026 = 417.3 (v2), 1,604,469/3985 =
402.6 (v3). **Implied per-game action capacity = 63k / (403–429) ≈ 147–156
actions.** Realized medians: 117 / 144 / 165 actions/game; means 145.5 /
161.0 / 159.4. The eval runs complete 25 games in 2h12m at 196–211 tok/s; the
~8h scored rerun across 25+ games reflects more games and scheduling, not a
larger per-game envelope — the per-game token cap is the same harness config
(assumption; verifiable by the same tokens/game grep on any scored-run pull).

**Proposal: `SENTINEL_BUDGET=150`** (per level attempt, the patch's unit),
thresholds default 50/75/90% → warnings at actions 75 / 113 / 135 of an
attempt. Rationale: 150 = the token-implied scored-regime action capacity
(147–156, stable across all three certified seeds), which is exactly the
budget the sentinel is meant to model. Checks against the recorded deaths:
sb26 move-limit death at 140 — all three warnings precede; lp85 GAME_OVERs at
131–133 — the 50%/75% warnings precede (the in-game 60-click resource on lp85
is a game mechanic the sentinel cannot and should not model); tu93's
301-action L1 grind — the full ladder fires with 150 to spare. **Mandatory
pre-seal check (A10):** re-run `compressed_canary.py` at B=150 on the three
recorded seeds; the W2 gate does not seal unless ≥5 games fire per run
(predicted comfortably: 17/25 games exceed 88 recorded actions). Post-run
verification: grep the build log for `SENTINEL v=1 kind=budget_threshold`;
**zero events on a run containing any ≥75-action attempt ⇒ the budget was
unset ⇒ the window is VOID (not FAIL)** — the feedback_kaggle_dataset_code_sync
class of silent no-op is thereby excluded from ever counting as evidence.

**Quota ledger (systems #12 — the A14 look and A15 replicate, previously
unscheduled, now scheduled):** rail = free Kaggle GPU builds, 30 GPU-h/wk;
one 25-game eval ≈ 2.2 GPU-h.

| week | item | GPU-h |
|---|---|---|
| Jul 21–27 | W1 (a): canary@B=150 (CPU, 0) + 3 certified seeds | 6.6 |
| Jul 21–27 | W2 (b): 3 certified seeds | 6.6 |
| Jul 21–27 | A17 72B screen (4 games full budget + tokens/s bench on the named SKU) | ~10 |
| Jul 21–27 | latent-state audit (offline CPU) | 0 |
| | **week total** | **≈23.2 / 30** |
| Jul 28–Aug 3 | **A14 binding cumulative look**: 3 FULL-budget seeds of the final stack (discharges A15 by construction, §6.1) | 6.6 |
| Jul 28–Aug 3 | W3 banking IF B+ approved | 6.6 |
| Jul 28–Aug 3 | EWM Stage-1 IF §9.2 gate passes | 6.6 |
| Jul 28–Aug 3 | fallback W0 seeds 2–3 IF §11 fallback triggered | 4.4 |
| | **week total (max path)** | **≈24.2 / 30** |

Push budget 2/day is the binding constraint on the max path (≈11 pushes over
7 days); the conditional items are mutually orderable within it.

---

## §13. Seal hygiene, su15, A17 cross-reference

**Seal hygiene (methodology N6), adopted as standing procedure:** every
threshold sealed in this document (the §7 resurrection bar 0.90; the §9.2
cheap-measurement bar 0.70/L≤10/3-of-5; the §10 keying bar 0.99; the §8
z=1.834 boundary; the §6.2 dismantle −0.10; the §12 B=150 pending R16 ruling)
is extracted verbatim into its measurement's own hash-committed thresholds
file under `runs/sealed/` BEFORE the measurement script first runs; results go
to separate append-only artifacts. The circulation stamp on this document
hash-commits the master copies.

**su15 (rl-planning minor):** A12 exclusion HOLDS for the sealed cumulative
look. Registered now: after the A13 GPT-5.6 re-probe completes, an amendment
MAY re-admit su15 for war-v4 and EWM evaluations only, with full-panel
sign-off; it is never re-admitted retroactively into any look already sealed.

**A17 (4/5 reviewers):** the gate-boolean repair (GO iff [≥2 levels AND
actions ≥90% of 27B] **OR** [beats Σ null_adj with registered margin]),
comparator definition over the 3 certified 27B seeds, marginal-result rule,
and SKU naming/verification are discharged in
`learnings/war_room/a17_72b_screen_scope.md` (filed 2026-07-19, incorporating
the multimodal-harness finding: the swap target is Qwen2.5-VL-72B-AWQ). Not
re-litigated here; this document only schedules its quota (§12) and records
that the screen is pre-Aug-1 blocking.

---

## §14. Part D republished — strategy priorities with (d) and (c) struck

| priority | line | re-priced basis | status |
|---|---|---|---|
| 1 | EWM Stage-1 plan-execute-verify on reliable carriers {tn36, tr87, tu93, ls20, ft09-L1} | expectation **+0.02–0.18 rail, central +0.08** (§9.1); largest registered non-model line | BLOCKED by §10 audit + §9.2 cheap measurement + sign-off |
| 2 | (a) budget sentinel | ceiling +0.06, expectation +0.01–0.03 | **W1 owner**; built, canaried; SENTINEL_BUDGET ruling pending (§12) |
| 3 | war-v4 72B multimodal screen (A17) | **the only registered wall-closer** | scope doc filed; pre-Aug-1 |
| 4 | (b) diff summarizer | ceiling +0.06, expectation +0.01–0.03 | W2, guard per §8 |
| 5 | banking-fixed | ≤ +0.15 pre-A16-haircut, expectation +0.03–0.08 | conditional (B+): sign-off + A16 + §10 audit |
| 6 | su15 GPT-5.6 re-probe (A13) | epistemic repair, ~$10 | after (f) local rig |
| 7 | filler | E[max@~106] ≈ 1.39; ~29% touch 1.44 | every window nothing credibly beats (+0.06–0.12 rule) |
| — | ~~(c)+(d) / (c)+Reki refutation flag~~ | ~~+0.10~~ | **STRUCK: (d) A18-killed; (c) killed §7** |
| — | ~~(g) budget re-allocation~~ | — | dead (A20) |

---

## §15. Dream digest review (Dreams/2026-07-19-124559.md)

Reviewed as required. The 2026-07-19 KAOS digest is a recency-only dry-run
cycle (0.00s, window all-time): 3 episodes (2 completed, 1 in flight), $0
spend, and **skills_scored=0 — exactly what the sealed expectation predicts**
(the skill library is empty; `feedback_kaos_improvements` documents that GPU
benchmarks remain impractical locally, so no skills have been admitted). The
hot-memory table is retrieval-flat (all hits=0, scores 0.48–0.50 = pure
recency prior over the R15-cycle documents — it is correctly surfacing this
campaign's active corpus, which is a sanity signal, not information), the
Hebbian graph is empty, and no failure fingerprints or consolidation
proposals were emitted. The cold list surfaces pre-campaign v8–v14 documents
as natural archive candidates; no action required. **Nothing actionable;
nothing panel-worthy.**

---

## §16. Directive-discharge table (every R15 directive → where discharged)

| # | R15 directive | discharged at |
|---|---|---|
| 1 | §2 stack sum republished with (d) removed (post-kill ceiling ≈ +0.21) | §2R (sums shown: 0.27 − 0.06 = +0.21 B+; +0.09 B−) |
| 2 | §4 Δclears re-derived under BOTH banking branches | §4R |
| 3 | P(pass) republished, binomial-sketch assumptions explicit | §5R |
| 4 | (c) disposition DECIDED; Part D corrected before ratification | §7 (KILL + counting bound + sealed resurrection path); §14 |
| 5 | α re-derivation (N2): name family or reset 0.05 one-sided | §3R (family named; conjunctive; α = 0.05 sealed) |
| 6 | Cumulative dismantle branch (M4): Δlc ≤ −0.10 → (f)-only | §6.2 (sealed; null trip-rate 0.24 published) |
| 7 | Guard false-kill calibration (N3): publish P(false kill), repair if >0.10 | §8 (0.26/window, familywise 0.59 → SE boundary z=1.834, familywise 0.097) |
| 8 | Budget regime of binding look stated (N5) | §6.1 (FULL budget; discharges A15 by construction) |
| 9 | KNOW#5 conditioned on keying (O2-1) | §7 (annotation) |
| 10 | Latent-state audit registered, blocking EWM Stage-1 AND banking | §10 (offline, N5 traces, $0; 0.99 bar sealed) |
| 11 | Reki-keying predict_metric re-run w/ sealed threshold before any (c) window | §7 (converted to sealed resurrection precondition, bar 0.90) |
| 12 | EWM re-price: reliable carriers, Q5, fidelity^L, Stage-1 gate exists, resync = contract change w/ bounds, OCM priced, cheap measurement | §9.1–§9.2 |
| 13 | A17 gate boolean/comparator/SKU repairs | `a17_72b_screen_scope.md` (07-19), cross-referenced §13 |
| 14 | Quota ledger incl. A14 look + A15 replicate scheduled (systems #12) | §12 ledger |
| 15 | W0 control-arm seed count for the cumulative look stated | §11 (n=4 band {3 war-eval + w0_s1}; sealed fallback) |
| 16 | (a) mechanism observable deterministic (O5) | §6.3 / sentinel build report (49 deaths / 0 violations) |
| 17 | W0: default-layer adopt, no seed-2, 16 levels not citable, band claim only | §1.3, §11 |
| 18 | Seal hygiene: thresholds hash-committed pre-measurement (N6) | §13 |
| 19 | su15: keep A12 exclusion; register post-A13 re-admission path | §13 |

All 19 directives discharged. The recalibrated A14 gate — as amended by §3R
(α), §6.2 (dismantle branch), §8 (guard boundaries), and §11 (control band) —
**seals on this circulation.**

END OF R16 REPUBLICATION


--- END OF PART 1 ---

================================================================================
PART 2: A17 AMENDMENT DRAFT (72B screen; files on sign-off)
source: learnings/preregistration_amendment_2026-07-20_A17.md
sha256: 28929db8d033fa59a8c50397d58006ca485e64190cf12000027c9a67ccb20413
================================================================================

# Pre-registration amendment — 2026-07-20 (A17 revision, post-R15)

**STATUS: DRAFT — NOT FILED.** Prepared 2026-07-20 for panel sign-off; to be appended
to the R16 circulation (or filed as `preregistration_amendment_2026-07-20.md` on
sign-off). Supersedes §A17 of `learnings/preregistration_amendment_2026-07-18b.md` in
full. Incorporates the four R15 A17 directives (repaired gate boolean, sealed
comparator statistic, hardware-SKU verification, quota ledger) per
`learnings/panel/round15/_directives.md`; the arithmetic is imported verbatim from
`learnings/war_room/a17_72b_screen_scope.md` (filed 2026-07-19). Everything below
seals BEFORE any bench observation: no 72B kernel has been pushed, no 72B tokens/s
number exists, and no term in this amendment conditions on one.

## A17′ — war-v4 72B capability screen (revised; pre-Aug-1, blocking)

### 1. Model artifact — sealed, with negative seal

1.1. The screen model is **Qwen2.5-VL-72B-Instruct-AWQ**, Kaggle Model
`qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1` (official QwenLM artifact,
verified present; 11 safetensors shards, 43,023,138,387 B ≈ 43.0 GB, AWQ W4A16),
attached as a Kaggle Model source — no upload, no download, no cloud spend.

1.2. **Negative seal:** the harness is MULTIMODAL — it renders the current grid as a
4× upscaled image (`MULTIMODAL_CONTEXT=current_grid`, `MULTIMODAL_UPSCALE=4`) and the
27B baseline is itself a VL model (`Qwen3_5ForConditionalGeneration` +
`Qwen2VLImageProcessor` in the server log). **Any text-only 72B artifact renders the
screen VOID** — it deletes the visual channel and confounds capability with a modality
regression. A run on a text-only model is discarded unscored; it neither GOes nor
NO-GOes the gate.

1.3. If a Qwen3-tier VL-72B AWQ artifact appears on Kaggle before the first push, it
may be substituted (same swap procedure) with a one-line filed note; no other
substitution is permitted. The original A17 phrase "Qwen3.6-72B-tier" is retired: no
such attachable Kaggle artifact exists (searched 2026-07-19, models + datasets, nil).

### 2. Games, comparator, and capability prong

2.1. Games: **ft09, sb26, lp85, vc33**, identical harness, full per-game fixed
~7920 s wallclock window, on the free Kaggle build rail.

2.2. **Comparator statistic (sealed): per-game MAX over the 3 certified 27B seeds
(war_eval v1/v2/v3 + W0), on BOTH sides.** The screen tests capability existence,
which is a max-property; max-on-both-sides is symmetric and matches the banking
line's order statistic. Frozen 27B side: ft09 MAX 2, sb26 1, lp85 1, vc33 2 →
**Σ 27B MAX = 6**.

2.3. **Capability prong:** Σ(72B per-game MAX lc) ≥ Σ(27B per-game MAX lc) + 2 =
**≥ 8**. The 72B-side max is taken over however many 72B seeds the quota affords (≥1).

2.4. **Marginal-result rule (pre-stated):** if the capability prong lands at exactly
+1 (Σ 72B MAX = 7), or either prong sits within one level of its threshold, run ONE
additional 72B seed on the two decisive games (largest 72B-vs-27B per-game gap) and
re-evaluate MAX. If still +1 after that seed → NO-GO. No further re-rolls.

### 3. Throughput-adjusted null — sealed formula and frozen arithmetic

3.1. The screen is a fixed-wallclock race, not fixed-action. Define
**ρ = tok/s(27B) / tok/s(72B)**, both measured from the `generated tokens/sec`
`summary.txt` line **on the same SKU** (§5). ρ is **self-measured: no external
throughput anchor exists for 72B-AWQ on this card** (daily brief 2026-07-19 §1c);
our bench is the reference. 27B reference: 192 tok/s (`w0_eval_s1/summary.txt`).

3.2. **N₇₂B(game) = ⌊(1/ρ)·N₂₇B(game)⌋**, where N₂₇B = the W0 27B baseline's total
actions in that game (Σ actions_per_level, `runs/kernel_pulls/w0_eval_s1/benchmark.json`).

3.3. **null_adj(game)** = the number of levels the W0 27B baseline had fully completed
by action N₇₂B (cumulative walk of its frozen `actions_per_level`; a level counts iff
its block closes within N₇₂B actions).

3.4. Frozen worked example, from W0 actions_per_level (ft09 [27,10,2]; sb26 [16,209];
lp85 [8,139]; vc33 [7,19,43]):

| game | N₂₇B | N₇₂B (ρ=2.5) | null_adj (ρ=2.5) | N₇₂B (ρ=3.0) | null_adj (ρ=3.0) |
|---|---:|---:|---:|---:|---:|
| ft09 | 39 | 15 | 0 | 13 | 0 |
| sb26 | 225 | 90 | 1 | 75 | 1 |
| lp85 | 147 | 58 | 1 | 49 | 1 |
| vc33 | 69 | 27 | 2 | 23 | 1 |
| **Σ** | | | **4** | | **3** |

Worked walk, ft09 at ρ=2.5: N₇₂B = ⌊39/2.5⌋ = 15; level-1 closes at cumulative action
27 > 15 → zero levels credited. A 72B merely matching 27B skill clears 0 ft09 levels
in the throttled budget — the throughput penalty made concrete. At ρ=3.0, vc33:
N₇₂B = ⌊69/3⌋ = 23; L1 closes at 7, L2 at 26 > 23 → null_adj = 1.

3.5. **Σ null_adj = 4 if measured ρ ≤ 2.5; = 3 if 2.5 < ρ ≤ 3.0.** If measured ρ
falls outside [2.4, 3.1], null_adj is recomputed at the measured ρ by the frozen walk
of §3.3 on the frozen W0 data — the PROCEDURE seals, leaving no post-hoc freedom.

### 4. Gate boolean — sealed decision rule (verbatim from the scope doc)

```
GO  iff
    ( CAPABILITY:  Σ(72B per-game MAX lc)  ≥  Σ(27B per-game MAX lc) + 2      # ≥ 8
      AND
      ACTION-PARITY:  Σ N₇₂B  ≥  0.90 · Σ N₂₇B )                             # throughput not binding
  OR
    ( CAPABILITY  (same ≥8 bar)
      AND
      THROUGHPUT-ADJUSTED:  Σ(72B per-game lc)  ≥  Σ null_adj  +  MARGIN )   # throughput binding, but wins anyway
NO-GO otherwise.
```

4.1. **MARGIN = +1 level**, registered: Σ(72B lc) ≥ 5 at ρ≤2.5 / ≥ 4 at ρ≤3.0. The
margin protects against a ρ-measurement error stepping null_adj by one integer.

4.2. The test is exact — integer level counts, no p-value; the margin IS the test.
n = 1–2 72B seeds cannot power a sign-flip test and the panel is asked not to demand
α on a capability-existence screen; this is stated now so it cannot be litigated later.

4.3. This repairs the R14-era defect the R15 panel named: the original A17 conjunction
(capability AND ≥90% action parity) auto-failed under any real 2.5–3× slowdown,
making NO-GO deterministic and null_adj dead code. The disjunction closes that branch.

### 5. Hardware SKU and self-measurement seal

5.1. Both rails are the **same verified physical SKU**: NVIDIA RTX PRO 6000 Blackwell
Server Edition ×1, ~96 GB (build-rail log `w0_eval_s1/…eval.log` CUDA check; scored
rail `machine_shape: NvidiaRtxPro6000` + harness hard-assert). Build-time throughput
therefore transfers to the scored budget with no cross-SKU correction.

5.2. The 72B tokens/s probe MUST run on this exact SKU before N₇₂B is computed. If
any kernel log prints a different GPU name, the null (§3) and the gate (§4) are
recomputed from scratch and the offending run is not scored.

### 6. Budget and deadline

6.1. **~7.5 GPU-h total** (canary push + scored bench + optional marginal seed, at
~2.5 GPU-h/push) on the free Kaggle rail's 30 GPU-h/wk; **$0 cloud spend** (zero-budget
rule). Contention with A14 cumulative look + A15 full-budget replicate is a stated
dependency: the weekly scheduler must keep (screen + A14 + A15 + open v3 windows)
≤ 30 GPU-h in each of the Jul 20–27 and Jul 27–Aug-3 weeks; the canary and scored
bench are the protected pair, the marginal seed yields first.

6.2. **The screen must READ OUT (gate evaluated, GO/NO-GO recorded) before Aug 1.**
It is blocking for war-v4 scoping: no v4 registration may file without this readout.

### 7. Pre-push runtime tests — sealed as BLOCKING

No scored push occurs until all three pass; a scored run made without them is discarded.

7.1. **Serve-config tool-call round-trip:** Qwen2.5-VL takes `--tool-call-parser
hermes`, has NO qwen3 reasoning parser and NO thinking mode — the 27B's
`qwen3_coder`/`--reasoning-parser qwen3`/`preserve_thinking` flags are removed, and
`LOCAL_ANALYZER_ENABLE_THINKING=false` is set. The boot smoke test is extended to
assert a TOOL CALL round-trips (not merely a chat completion), since a silent parser
mismatch is the highest-probability zero (`feedback_test_before_submit` class).

7.2. **Reset-path A/B — byte-identical to the frozen fork:** per the reset-fragility
caution (daily brief 2026-07-19 §1b: a reset-cap change turned a 9-min agent into a
1-hour 0-score run), the 72B swap changes ONLY the model + its serve-config constants.
The v4-eval builder asserts the reset constants (`ONLY_RESET_LEVELS=true`,
`max_runtime_minutes: 45`) and the ~7920 s game-window deadline are byte-identical to
the 27B baseline; the W0 27B seeds are the implicit control arm. A run whose window
is not ~7920 s voids the null comparison and is discarded, not scored.

7.3. **Preflight structural checks** (`scripts/preflight.py`) pass — no scratch-built
kernel drift (fingerprint family `provenance:scratch-built`, n=5, stays at n=5).

### 8. Fail consequences — sealed now

8.1. **NO-GO → the war-v4 line CLOSES for the campaign.** The frozen 27B stack is the
terminal model; the finding ("72B replicates the ~1-level grinder profile under the
binding budget") goes to the panel immediately, which then decides in July — the
campaign proceeds with no registered wall-closer. **No partial credit**: a near-miss
is a NO-GO after the §2.4 marginal seed, full stop.

8.2. **No re-screen without a materially different artifact.** "Materially different"
means (exhaustively): a different model family or generation (e.g. a Qwen3-tier
VL-72B appearing on Kaggle), a different parameter class, or a quantization change
that alters measured ρ by ≥ 0.5. Re-running the SAME artifact with tuned serve flags,
prompts, or seeds is not material and is prohibited.

8.3. **GO → the war-v4 build window opens**, gated by its OWN subsequent
pre-registration (A14-form gate: sealed prongs, sealed consequences) which must
circulate to the panel before v4 consumes any scored window. GO grants the right to
register, not the right to ship.

### 9. Seal hygiene

Per methodology N6 (R15): this amendment is the hash-committed threshold file for the
A17 screen; it is committed BEFORE any 72B measurement script runs. Bench results land
in separate append-only artifacts (`runs/…/summary.txt`, `benchmark.json`); the gate
evaluation (~Jul 30) cites this file's sha and applies §3–§4 arithmetic with no free
parameters.

— END A17′ (DRAFT — NOT FILED; seals on panel sign-off at R16 circulation) —


--- END OF PART 2 ---

================================================================================
PART 3: LATENT-STATE AUDIT PROTOCOL (R15 blocking prereq, discharged)
source: learnings/war_room/latent_state_audit_protocol.md
sha256: f8720e8c001b574bb3fcb21f4350ac4c7b0f9b92ce00b08a6532a0e131676038
================================================================================

# Latent-state audit protocol — quantifying hidden state behind the observable frame

Status: v1, 2026-07-20. Panel R15 mandate (5/5): state-aliasing is one root cause
behind (1) predict-metric recurrence acc 0.465 (`runs/predict_metric/report.md`),
(2) EWM step-0 plan aborts (`runs/ewm_dryrun/report.md`), (3) the N5 prune_trace
bug (`runs/war_eval_v1/prune_replay_diag.json`). This audit is a BLOCKING prereq
for EWM Stage-1 and any banking/replay build.

Implementation: `scripts/latent_state_audit.py` (stdlib-only, CPU, $0, offline).
Output: `runs/latent_state_audit/report.md` + `report.json`.

## 1. Data

- **Primary**: per-action event traces `runs/kernel_pulls/*/artifacts/*_events.jsonl`
  and `runs/phase1_ab/seed1/artifacts/*_events.jsonl` (auto-discovered). Each
  `type=action` event carries the full settled 64x64 `board`, `action_display`
  (fully qualifies ACTION1-5 direction and ACTION6 MOUSE(r,c)), `board_changed`,
  `level`, `score`. `type=initial` gives the pre-play frame; `type=analysis`
  frames are skipped (no action taken; predict_metric confirmed 0 digest drift).
- **Cross-reference**: `runs/ewm_dryrun/raw.json` per-game sim fidelity
  (step_acc, abort-step distribution) — the EWM-consumer view of the same
  aliasing.
- **Anchors**: N5 determinism audit (all 25 games frame-deterministic under
  full-history replay) — so ALL aliasing found here is *hidden state*, not
  engine stochasticity-from-reset. True per-transition stochasticity would show
  up as ALIASED-UNRESOLVED; N5 says any such finding on within-stream data must
  be treated as an unmodeled deterministic variable, not noise.

Analysis unit = versioned game id (e.g. `ls20-9607627b`); report row = 4-char
game id (worst verdict across versions if a game appears in several engine
versions — engine-version drift must never masquerade as hidden state).

## 2. Aliasing measurement (a)

A transition is `(s, a) -> s'` with `s` = blake2b-8 digest of the board the
agent acted on, `a` = `action_display`, `s'` = digest of the settled board.
RESET is an ordinary action. `t` = actions since last RESET (counts no-ops —
the N5 bug proved no-ops tick hidden state).

For every key with >= 2 visits, the empirical next-frame outcome distribution:

- **determinism** = sum(max outcome count per key) / sum(visits), over repeat
  keys, visit-weighted. 1.0 = frame is Markov on the observed support.
- **entropy** = visit-weighted mean Shannon entropy (bits) of the outcome
  distribution per key. > 0 = aliased.
- **aliased-key rate** = repeat keys with > 1 distinct outcome / repeat keys.
- Two scopes: **pooled** (keys shared across all streams of the versioned game
  — what banking/persistent models see) and **within-stream** (keys scoped per
  trace — what an in-run EWM sim sees). Within-stream aliasing is the stronger
  finding: the same session, same engine, same frame, same action, different
  outcome ⇒ hidden state moved.

## 3. Candidate hidden variables (b)

Augment the key with a candidate `h`: key = (s, a, h). Recompute determinism.
Candidates, ordered cheapest-first (the *minimal* augmentation reaching >= 99%
determinism wins):

| rank | candidate | h | class |
|---|---|---|---|
| 1 | level | pre-action level | observable-meta |
| 2 | score | pre-action score | observable-meta |
| 3 | meta | (level, score) | observable-meta |
| 4 | parity | t mod 2 | hidden phase |
| 5 | mod3 / mod4 / mod5 | t mod k | hidden phase |
| 8 | prev_bc | did the previous action change the board | hidden history |
| 9 | hist1 / hist2 / hist3 | last k action keys | hidden history |
| 12 | meta_parity | (level, score, t mod 2) | compound |
| 13 | meta_hist1 | (level, score, last action) | compound |
| — | tcount | exact t (diagnostic only, degenerate) | diagnostic |

**Support guard**: an augmentation is only eligible as a resolver if its
remaining repeat-visit mass >= max(10, 20% of base repeat visits). Otherwise it
merely shattered the keys (any injective function "resolves" everything at n=1)
and is reported as SUPPORT-COLLAPSED.

## 4. Per-game verdict (c)

- **CLEAN** — base determinism >= 0.99 (frame is Markov on observed support).
- **CLEAN-META** — resolved by observable metadata (level/score); the *full
  observation* is Markov even though the raw grid is not.
- **ALIASED-RESOLVABLE(h)** — a hidden candidate h reaches >= 0.99 with support.
- **ALIASED-UNRESOLVED** — no candidate resolves. Per N5 this means a hidden
  variable outside the candidate family (deep counter, object-internal state),
  not stochasticity — but it is operationally equivalent to stochastic for any
  frame-conditioned model.
- **LOW-SUPPORT** — < 20 repeat visits; no verdict earned either way.

## 5. Consumers — how the table answers them

**EWM Stage-1 (carrier selection + resync question)**
- Safe carriers = CLEAN / CLEAN-META games: a sim keyed on the frame (plus
  visible meta) can be faithful; mismatch-aborts there are sim bugs or engine-
  version drift, not aliasing.
- ALIASED-RESOLVABLE with a *phase* resolver (parity / mod-k): the sim drifts
  out of phase but reality stays deterministic ⇒ **resync-before-abort works**
  (re-read the settled frame, re-plan; or better, add the phase variable to the
  sim state). ALIASED-UNRESOLVED ⇒ resync does NOT restore predictability —
  abort-and-fallback is correct; do not carry EWM there.
- Cross-check column: ewm_dryrun step_acc + step-0 abort share should
  anti-correlate with determinism; step-0 aborts on games whose aliasing is
  phase-resolvable are exactly the "timer/hidden-counter phase misalignment"
  failure R15 named.

**Banking / replay**
- N5 already proved: full unpruned replay from RESET survives on all 25 games.
  The audit refines that to *partial/pruned* replay: a banked trajectory may be
  spliced or pruned ONLY in CLEAN / CLEAN-META games (frame Markov ⇒ a matching
  frame is a sufficient resync point). In ALIASED games — resolvable or not —
  banking must be **full-replay-only from RESET, zero pruning** (the exact
  prune_trace failure mode: dropped leading no-ops = dropped hidden-state
  mutations).

## 6. Selftests (must pass on every run)

1. **Hidden mod-3 counter**: synthetic game whose action only fires when a
   hidden counter (invisible in the frame) % 3 == 0. Audit must find base
   aliasing and recover `mod3` as the minimal resolver at >= 99%.
2. **Clean Markov walk** → CLEAN, zero entropy.
3. **Coin-flip transitions** → ALIASED-UNRESOLVED (no candidate, incl. history,
   may claim it).

## 7. Limitations (declared, not hidden)

- Determinism is measured on *observed support*; rarely-visited keys can hide
  aliasing (Wilson-style caution applies at low repeat counts — hence
  LOW-SUPPORT).
- Candidate family is finite; UNRESOLVED means "not resolvable by cheap
  counters/history windows", not "stochastic" (N5 forbids that reading).
- Streams come from agent policies, so key coverage is policy-biased; a CLEAN
  verdict is "no aliasing seen where the agent actually walked".


--- END OF PART 3 ---

================================================================================
PART 4: LATENT-STATE AUDIT RESULTS
source: runs/latent_state_audit/report.md
sha256: 2b7e0c29f1616421cb34613640b426ea296d23b46fb50221a1c6a12197615960
================================================================================

# Latent-state audit — per-game hidden-state quantification

Protocol: `learnings/war_room/latent_state_audit_protocol.md`. Selftest: **PASS** (synthetic hidden mod-3 counter recovered; clean stream = CLEAN; coin-flip stream = UNRESOLVED).

Coverage: 40 versioned games, 200 streams, 33777 actions; analysis-frame drift events: 0.

Determinism = P(modal next frame | frame, action) over keys seen >= 2x, visit-weighted (pooled across streams of the same engine version). Entropy = mean outcome entropy (bits). 'within' = keys scoped to a single stream (strongest hidden-state evidence). Resolved = augmented determinism >= 0.99.

## Verdict table

| game | streams | actions | rep.visits | det | H bits | within det | resolver | det.res | verdict | EWM step_acc | step0-abort/plan | EWM carrier | resync | banking |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|---|
| ar25 | 8 | 1020 | 184 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | - | - | SAFE | NOT-NEEDED | PREFIX-SAFE |
| bp35 | 8 | 2914 | 1534 | 0.996 | 0.012 | 0.997 | - | - | **CLEAN** | - | - | SAFE | NOT-NEEDED | PREFIX-SAFE |
| cd82 | 8 | 1362 | 300 | 0.753 | 0.547 | 0.608 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| cn04 | 8 | 1685 | 275 | 0.938 | 0.131 | 0.968 | mod3 | 1.000 | **ALIASED-RESOLVABLE(mod3)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| dc22 | 8 | 787 | 73 | 0.671 | 0.658 | 0.548 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| ft09 | 8 | 801 | 232 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.922 | 0.116 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| g50t | 8 | 865 | 199 | 0.693 | 0.640 | 0.511 | - | - | **ALIASED-UNRESOLVED** | - | - | NO | NO | FULL-REPLAY-ONLY |
| ka59 | 8 | 957 | 108 | 0.741 | 0.530 | 0.519 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| lf52 | 8 | 1107 | 97 | 1.000 | 0.000 | - | - | - | **CLEAN** | 0.532 | 0.479 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| lp85 | 8 | 724 | 142 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.129 | 0.889 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| ls20 | 8 | 739 | 126 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.793 | 0.384 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| m0r0 | 8 | 1674 | 821 | 0.618 | 0.858 | 0.589 | - | - | **ALIASED-UNRESOLVED** | - | - | NO | NO | FULL-REPLAY-ONLY |
| r11l | 8 | 808 | 99 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | - | - | SAFE | NOT-NEEDED | PREFIX-SAFE |
| re86 | 8 | 1193 | 72 | 0.958 | 0.083 | 0.500 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| s5i5 | 8 | 417 | 36 | 0.972 | 0.056 | 0.833 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | 0.293 | 0.707 | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| sb26 | 8 | 1791 | 332 | 0.985 | 0.041 | 0.976 | parity | 0.996 | **ALIASED-RESOLVABLE(parity)** | 0.167 | 0.833 | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| sc25 | 8 | 1341 | 221 | 0.760 | 0.519 | 0.691 | mod5 | 1.000 | **ALIASED-RESOLVABLE(mod5)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| sk48 | 8 | 1310 | 365 | 0.767 | 0.586 | 0.724 | - | - | **ALIASED-UNRESOLVED** | - | - | NO | NO | FULL-REPLAY-ONLY |
| sp80 | 8 | 1658 | 489 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.564 | 0.524 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| su15 | 8 | 1087 | 338 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.312 | 0.724 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| tn36 | 8 | 932 | 31 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.965 | 0.094 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| tr87 | 8 | 1740 | 156 | 0.910 | 0.207 | 1.000 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | 0.800 | 0.236 | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| tu93 | 8 | 1057 | 667 | 1.000 | 0.000 | 1.000 | - | - | **CLEAN** | 0.922 | 0.152 | SAFE | NOT-NEEDED | PREFIX-SAFE |
| vc33 | 8 | 697 | 60 | 0.983 | 0.033 | 0.967 | parity | 1.000 | **ALIASED-RESOLVABLE(parity)** | 0.458 | 0.591 | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |
| wa30 | 8 | 2390 | 425 | 0.739 | 0.704 | 0.735 | mod4 | 1.000 | **ALIASED-RESOLVABLE(mod4)** | - | - | PHASE-AUGMENT | YES | FULL-REPLAY-ONLY |

Rows reflect the benchmark engine version (most streams). Minority-version drift and near-misses:

- **cn04**: older engine version(s) disagree — cn04-65d47d14:ALIASED-RESOLVABLE(mod5) (phase1_ab/seed1 era); engine-version drift, NOT merged into the verdict.
- **ka59**: older engine version(s) disagree — ka59-9f096b4a:ALIASED-UNRESOLVED (phase1_ab/seed1 era); engine-version drift, NOT merged into the verdict.
- **m0r0**: candidate(s) mod3, mod4, mod5 reach >= 0.99 determinism on the repeat support that survives augmentation, but fail the support guard (SUPPORT-COLLAPSED) — plausibly resolvable with more data; treated as UNRESOLVED until then.

## Candidate breakdown (aliased games only)

### cd82 (cd82-fb555c5d) — base det 0.753, 68/126 keys aliased, 35 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.753 | 300 | y | - |
| score | observable-meta | 0.753 | 300 | y | - |
| meta | observable-meta | 0.753 | 300 | y | - |
| parity | hidden-phase | 1.000 | 190 | y | YES |
| mod3 | hidden-phase | 1.000 | 190 | y | YES |
| mod4 | hidden-phase | 1.000 | 190 | y | YES |
| mod5 | hidden-phase | 1.000 | 190 | y | YES |
| prev_bc | hidden-history | 0.838 | 216 | y | - |
| hist1 | hidden-history | 0.720 | 168 | y | - |
| hist2 | hidden-history | 0.705 | 88 | y | - |
| hist3 | hidden-history | 0.708 | 48 | n | - |
| meta_parity | compound | 1.000 | 190 | y | YES |
| meta_hist1 | compound | 0.720 | 168 | y | - |
| tcount | diagnostic | 1.000 | 190 | n | - |

### cn04 (cn04-2fe56bfb) — base det 0.938, 17/106 keys aliased, 5 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.938 | 275 | y | - |
| score | observable-meta | 0.938 | 275 | y | - |
| meta | observable-meta | 0.938 | 275 | y | - |
| parity | hidden-phase | 0.976 | 245 | y | - |
| mod3 | hidden-phase | 1.000 | 235 | y | YES |
| mod4 | hidden-phase | 1.000 | 235 | y | YES |
| mod5 | hidden-phase | 1.000 | 235 | y | YES |
| prev_bc | hidden-history | 0.954 | 261 | y | - |
| hist1 | hidden-history | 0.955 | 242 | y | - |
| hist2 | hidden-history | 0.968 | 217 | y | - |
| hist3 | hidden-history | 0.975 | 198 | y | - |
| meta_parity | compound | 0.976 | 245 | y | - |
| meta_hist1 | compound | 0.955 | 242 | y | - |
| tcount | diagnostic | 1.000 | 235 | n | - |

### dc22 (dc22-fdcac232) — base det 0.671, 24/36 keys aliased, 19 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.671 | 73 | y | - |
| score | observable-meta | 0.671 | 73 | y | - |
| meta | observable-meta | 0.671 | 73 | y | - |
| parity | hidden-phase | 1.000 | 23 | y | YES |
| mod3 | hidden-phase | 0.957 | 23 | y | - |
| mod4 | hidden-phase | 1.000 | 21 | y | YES |
| mod5 | hidden-phase | 0.920 | 25 | y | - |
| prev_bc | hidden-history | 0.848 | 33 | y | - |
| hist1 | hidden-history | 0.653 | 49 | y | - |
| hist2 | hidden-history | 0.667 | 33 | y | - |
| hist3 | hidden-history | 0.737 | 19 | y | - |
| meta_parity | compound | 1.000 | 23 | y | YES |
| meta_hist1 | compound | 0.653 | 49 | y | - |
| tcount | diagnostic | 1.000 | 21 | n | - |

### g50t (g50t-5849a774) — base det 0.693, 59/83 keys aliased, 45 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.693 | 199 | y | - |
| score | observable-meta | 0.693 | 199 | y | - |
| meta | observable-meta | 0.693 | 199 | y | - |
| parity | hidden-phase | 0.978 | 93 | y | - |
| mod3 | hidden-phase | 0.989 | 91 | y | - |
| mod4 | hidden-phase | 0.989 | 91 | y | - |
| mod5 | hidden-phase | 0.989 | 91 | y | - |
| prev_bc | hidden-history | 0.858 | 106 | y | - |
| hist1 | hidden-history | 0.717 | 145 | y | - |
| hist2 | hidden-history | 0.702 | 114 | y | - |
| hist3 | hidden-history | 0.716 | 88 | y | - |
| meta_parity | compound | 0.978 | 93 | y | - |
| meta_hist1 | compound | 0.717 | 145 | y | - |
| tcount | diagnostic | 0.989 | 91 | n | - |

### ka59 (ka59-38d34dbb) — base det 0.741, 28/45 keys aliased, 26 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.741 | 108 | y | - |
| score | observable-meta | 0.741 | 108 | y | - |
| meta | observable-meta | 0.741 | 108 | y | - |
| parity | hidden-phase | 1.000 | 41 | y | YES |
| mod3 | hidden-phase | 1.000 | 33 | y | YES |
| mod4 | hidden-phase | 1.000 | 37 | y | YES |
| mod5 | hidden-phase | 1.000 | 29 | y | YES |
| prev_bc | hidden-history | 0.947 | 57 | y | - |
| hist1 | hidden-history | 0.724 | 98 | y | - |
| hist2 | hidden-history | 0.736 | 87 | y | - |
| hist3 | hidden-history | 0.728 | 81 | y | - |
| meta_parity | compound | 1.000 | 41 | y | YES |
| meta_hist1 | compound | 0.724 | 98 | y | - |
| tcount | diagnostic | 1.000 | 23 | n | - |

### m0r0 (m0r0-492f87ba) — base det 0.618, 298/340 keys aliased, 282 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.618 | 821 | y | - |
| score | observable-meta | 0.618 | 821 | y | - |
| meta | observable-meta | 0.618 | 821 | y | - |
| parity | hidden-phase | 0.728 | 309 | y | - |
| mod3 | hidden-phase | 1.000 | 147 | n | - |
| mod4 | hidden-phase | 1.000 | 147 | n | - |
| mod5 | hidden-phase | 1.000 | 147 | n | - |
| prev_bc | hidden-history | 0.674 | 313 | y | - |
| hist1 | hidden-history | 0.613 | 741 | y | - |
| hist2 | hidden-history | 0.606 | 688 | y | - |
| hist3 | hidden-history | 0.614 | 611 | y | - |
| meta_parity | compound | 0.728 | 309 | y | - |
| meta_hist1 | compound | 0.613 | 741 | y | - |
| tcount | diagnostic | 1.000 | 147 | n | - |

### re86 (re86-8af5384d) — base det 0.958, 3/27 keys aliased, 3 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.958 | 72 | y | - |
| score | observable-meta | 0.958 | 72 | y | - |
| meta | observable-meta | 0.958 | 72 | y | - |
| parity | hidden-phase | 1.000 | 47 | y | YES |
| mod3 | hidden-phase | 1.000 | 51 | y | YES |
| mod4 | hidden-phase | 1.000 | 33 | y | YES |
| mod5 | hidden-phase | 1.000 | 19 | y | YES |
| prev_bc | hidden-history | 1.000 | 66 | y | YES |
| hist1 | hidden-history | 0.984 | 61 | y | - |
| hist2 | hidden-history | 0.982 | 55 | y | - |
| hist3 | hidden-history | 0.979 | 47 | y | - |
| meta_parity | compound | 1.000 | 47 | y | YES |
| meta_hist1 | compound | 0.984 | 61 | y | - |
| tcount | diagnostic | 1.000 | 19 | n | - |

### s5i5 (s5i5-18d95033) — base det 0.972, 1/15 keys aliased, 1 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.972 | 36 | y | - |
| score | observable-meta | 0.972 | 36 | y | - |
| meta | observable-meta | 0.972 | 36 | y | - |
| parity | hidden-phase | 1.000 | 32 | y | YES |
| mod3 | hidden-phase | 1.000 | 32 | y | YES |
| mod4 | hidden-phase | 1.000 | 32 | y | YES |
| mod5 | hidden-phase | 1.000 | 32 | y | YES |
| prev_bc | hidden-history | 1.000 | 31 | y | YES |
| hist1 | hidden-history | 0.929 | 14 | y | - |
| hist2 | hidden-history | 0.917 | 12 | y | - |
| hist3 | hidden-history | 0.917 | 12 | y | - |
| meta_parity | compound | 1.000 | 32 | y | YES |
| meta_hist1 | compound | 0.929 | 14 | y | - |
| tcount | diagnostic | 1.000 | 32 | n | - |

### sb26 (sb26-7fbdac44) — base det 0.985, 5/119 keys aliased, 3 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.985 | 332 | y | - |
| score | observable-meta | 0.985 | 332 | y | - |
| meta | observable-meta | 0.985 | 332 | y | - |
| parity | hidden-phase | 0.996 | 274 | y | YES |
| mod3 | hidden-phase | 0.989 | 266 | y | - |
| mod4 | hidden-phase | 0.996 | 255 | y | YES |
| mod5 | hidden-phase | 0.996 | 234 | y | YES |
| prev_bc | hidden-history | 0.993 | 305 | y | YES |
| hist1 | hidden-history | 0.996 | 267 | y | YES |
| hist2 | hidden-history | 1.000 | 237 | y | YES |
| hist3 | hidden-history | 1.000 | 211 | y | YES |
| meta_parity | compound | 0.996 | 274 | y | YES |
| meta_hist1 | compound | 0.996 | 267 | y | YES |
| tcount | diagnostic | 0.995 | 220 | n | - |

### sc25 (sc25-635fd71a) — base det 0.760, 44/85 keys aliased, 43 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.760 | 221 | y | - |
| score | observable-meta | 0.760 | 221 | y | - |
| meta | observable-meta | 0.760 | 221 | y | - |
| parity | hidden-phase | 0.915 | 129 | y | - |
| mod3 | hidden-phase | 0.971 | 104 | y | - |
| mod4 | hidden-phase | 0.982 | 112 | y | - |
| mod5 | hidden-phase | 1.000 | 98 | y | YES |
| prev_bc | hidden-history | 0.993 | 136 | y | YES |
| hist1 | hidden-history | 0.789 | 180 | y | - |
| hist2 | hidden-history | 0.773 | 141 | y | - |
| hist3 | hidden-history | 0.760 | 100 | y | - |
| meta_parity | compound | 0.915 | 129 | y | - |
| meta_hist1 | compound | 0.789 | 180 | y | - |
| tcount | diagnostic | 1.000 | 86 | n | - |

### sk48 (sk48-d8078629) — base det 0.767, 79/129 keys aliased, 61 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.767 | 365 | y | - |
| score | observable-meta | 0.767 | 365 | y | - |
| meta | observable-meta | 0.767 | 365 | y | - |
| parity | hidden-phase | 0.750 | 200 | y | - |
| mod3 | hidden-phase | 0.957 | 117 | y | - |
| mod4 | hidden-phase | 0.974 | 114 | y | - |
| mod5 | hidden-phase | 0.982 | 112 | y | - |
| prev_bc | hidden-history | 0.754 | 264 | y | - |
| hist1 | hidden-history | 0.768 | 340 | y | - |
| hist2 | hidden-history | 0.762 | 311 | y | - |
| hist3 | hidden-history | 0.762 | 286 | y | - |
| meta_parity | compound | 0.750 | 200 | y | - |
| meta_hist1 | compound | 0.768 | 340 | y | - |
| tcount | diagnostic | 0.982 | 112 | n | - |

### tr87 (tr87-cd924810) — base det 0.910, 11/51 keys aliased, 0 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.910 | 156 | y | - |
| score | observable-meta | 0.910 | 156 | y | - |
| meta | observable-meta | 0.910 | 156 | y | - |
| parity | hidden-phase | 1.000 | 143 | y | YES |
| mod3 | hidden-phase | 1.000 | 143 | y | YES |
| mod4 | hidden-phase | 1.000 | 143 | y | YES |
| mod5 | hidden-phase | 1.000 | 143 | y | YES |
| prev_bc | hidden-history | 0.910 | 156 | y | - |
| hist1 | hidden-history | 0.928 | 139 | y | - |
| hist2 | hidden-history | 0.933 | 120 | y | - |
| hist3 | hidden-history | 0.933 | 105 | y | - |
| meta_parity | compound | 1.000 | 143 | y | YES |
| meta_hist1 | compound | 0.928 | 139 | y | - |
| tcount | diagnostic | 1.000 | 143 | n | - |

### vc33 (vc33-5430563c) — base det 0.983, 1/28 keys aliased, 1 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.983 | 60 | y | - |
| score | observable-meta | 0.983 | 60 | y | - |
| meta | observable-meta | 0.983 | 60 | y | - |
| parity | hidden-phase | 1.000 | 54 | y | YES |
| mod3 | hidden-phase | 1.000 | 32 | y | YES |
| mod4 | hidden-phase | 1.000 | 52 | y | YES |
| mod5 | hidden-phase | 1.000 | 29 | y | YES |
| prev_bc | hidden-history | 1.000 | 58 | y | YES |
| hist1 | hidden-history | 0.979 | 47 | y | - |
| hist2 | hidden-history | 1.000 | 31 | y | YES |
| hist3 | hidden-history | 1.000 | 25 | y | YES |
| meta_parity | compound | 1.000 | 54 | y | YES |
| meta_hist1 | compound | 0.979 | 47 | y | - |
| tcount | diagnostic | 1.000 | 9 | n | - |

### wa30 (wa30-ee6fef47) — base det 0.739, 105/142 keys aliased, 98 involve a no-effect outcome

| candidate | class | det | rep.visits | eligible | resolves |
|---|---|---:|---:|---|---|
| level | observable-meta | 0.739 | 425 | y | - |
| score | observable-meta | 0.739 | 425 | y | - |
| meta | observable-meta | 0.739 | 425 | y | - |
| parity | hidden-phase | 0.673 | 263 | y | - |
| mod3 | hidden-phase | 0.906 | 106 | y | - |
| mod4 | hidden-phase | 1.000 | 86 | y | YES |
| mod5 | hidden-phase | 1.000 | 86 | y | YES |
| prev_bc | hidden-history | 0.654 | 257 | y | - |
| hist1 | hidden-history | 0.741 | 394 | y | - |
| hist2 | hidden-history | 0.736 | 364 | y | - |
| hist3 | hidden-history | 0.732 | 336 | y | - |
| meta_parity | compound | 0.673 | 263 | y | - |
| meta_hist1 | compound | 0.741 | 394 | y | - |
| tcount | diagnostic | 1.000 | 86 | n | - |

## Findings (ties to the three R15 failures)

1. **Hidden phase counters are the dominant aliasing mechanism**: 11/14 aliased benchmark games are fully resolved (det -> ~1.000) by a small modular counter of actions-since-RESET (parity or mod 3/4/5) — an invisible blink/tick phase. Observable metadata (level/score) resolves NOTHING: the hidden variable is truly outside the observation.
2. **This is the predict-metric 0.465 mechanism**: in the aliased games, most aliased (frame,action) keys have a no-effect outcome on one phase and an effect on the other (see 'involve a no-effect outcome' counts). A no-effect FACT keyed on (frame,action) alone is wrong whenever the phase differs on recurrence — exactly the ~54% flip rate R14 measured.
3. **This is the N5 prune_trace mechanism**: no-op actions still advance the phase counter; dropping leading no-ops desyncs the phase and the first replayed action lands on a different frame (step-0 frame_divergence on sc25/m0r0 — sc25 is mod5-aliased here; m0r0 is the worst unresolved game, det 0.618).
4. **EWM step-0 aborts split into two causes**: on ALIASED games (s5i5, sb26, vc33, tr87) low sim step_acc co-occurs with phase aliasing — the sim is phase-blind, and resync/phase-augmentation fixes it. But lf52, lp85, sp80, su15 are frame-Markov CLEAN yet still have step_acc < 0.6 — those sims are just wrong (sim bugs / engine-version drift), and NO amount of state augmentation or resync will save them; they need sim fixes, not aliasing work.

## Consumer answers

- **EWM Stage-1 safe carriers** (frame(+meta) is Markov): ar25, bp35, ft09, lf52, lp85, ls20, r11l, sp80, su15, tn36, tu93
- **Resync-before-abort viable** (phase variable drifts, reality deterministic): cd82, cn04, dc22, ka59, re86, s5i5, sb26, sc25, tr87, vc33, wa30
- **EWM no-go** (unresolved aliasing — abort-and-fallback is correct): g50t, m0r0, sk48
- **Banking prefix-splice safe**: ar25, bp35, ft09, lf52, lp85, ls20, r11l, sp80, su15, tn36, tu93; all other audited games are FULL-REPLAY-ONLY from RESET with ZERO pruning (N5: full unpruned replay survives on all 25; the prune_trace bug dropped hidden-state-mutating no-ops).


--- END OF PART 4 ---

================================================================================
PART 5: DAILY BRIEF 2026-07-20 (incl. R16 open questions)
source: learnings/daily_brief_2026-07-20.md
sha256: 06dd56df334f88645f9ebfb6ee94485121227040c76a8a02fb9c621dfad0fa2a
================================================================================

# Daily Brief — 2026-07-20 (Monday)

## §1a Result deep-dive

### Filler draw = 0.92 — in-band, uninformative by design, ledger updated
Frozen-fork filler scored **0.92** (submitted 00:07Z, COMPLETE). Under the pooled process model this is z = −0.28 — dead-center noise. Pre-registered expectation (filler = vanilla-band draw, 0.76–1.33 observed) met; nothing went "wrong"; no mechanism claim rides on a filler draw by construction.

**Ledger update (n=12 pooled across both closed arms):**
- Control arm n=7 {0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92}: mean 0.980, σ̂ 0.166, χ²-CI [0.107, 0.365]
- Pooled n=12 (+ war {0.91, 1.08, 0.88, 1.05, 0.76}): mean **0.962**, σ̂ **0.147**, χ²-CI **[0.104, 0.250]**

σ̂ ticked down from 0.154 → 0.147; the 0.13–0.17 bracket holds. Note for pricing honesty: a naive iid-normal simulation at these parameters gives E[max@105 remaining] ≈ 1.33, P(touch 1.44) ≈ 6% — noticeably below the LB process model's mixture estimate (E[max@~105] ≈ 1.39, P(1.44) ≈ 0.29). The divergence is the ft09-L2 right tail the normal approximation cannot see (a ≥1.30 night is 44% ft09-L2 per `runs/lb_process_model/`). The mixture model remains the honest pricing basis; the normal figure is a lower bound. Window pricing unchanged: experiments must credibly claim ≥ +0.06–0.12.

War arm remains CLOSED (A9); no draw #6 ever. Queue incident caught this morning: `pending` was empty after the daemon popped the filler — re-armed frozen fork (preflight ALLOW, T1–T4 OK, 0 warns, recurrence clean). Protocol note: the daemon pops without auto-refilling; refill belongs to the morning loop checklist.

### Leaderboard
Leader Yuto Kojima 1.86 (resubmitted 00:44 today) — unchanged. Wall (16th) = 1.44 unchanged; Tecnod8.AI 1.61 now #2; ~14 teams in the 1.43–1.61 band; Tufa Labs 1.45. **Us: 40th @ 1.33** (85 entries). Notable: #3 DhanaLakshmiMalla at 1.60 in only 3 entries — worth a fork-audit glance if they publish. 1830 teams. No structural shift.

## §1b Discussions sweep (since 07-19)
1. **#727629 "schema-harness.github.io 99%" (CreateAMind, new):** external site claiming frontier models hit ~99% on the 25 public games; no method, no code; harness "reruns weak games with stronger models and keeps the better score" (max-over-draws cherry-picking); keithtyser skeptical, Yakunin links arXiv:2602.02710 as "possibly wishful thinking". **IGNORE** — closed-source frontier-API + keep-best-rerun is exactly the single-draw-luck/public-overfit pattern we refuse; nothing transfers to the offline Kaggle runtime. Re-triage only if code drops.
2. **#727505 "Constraint Before Control" — new author comment:** Yakunin reports his own architecture scores 0.17 and only after disabling nearly everything (memory, attempt limits, structured verification all died on runtime compatibility). **IGNORE**, but it independently reinforces two priors: simplicity-wins under this runtime, and the surviving minimal loop is a degenerate cousin of our plan-execute-verify EWM line.
3. **#727119 host thread:** Greg Kamradt confirms hosts cannot access notebooks until teams open-source them. **IGNORE** (informational; mildly reassuring for private-kernel confidentiality).

## §1c Research sweep (window: arXiv Fri Jul 17 + Mon Jul 20)
- **arXiv:2607.15439 — "Do Coding Agents Need Executable World Models, Simplification, and Verification to Solve ARC-AGI-3?" (Rodionov, announced TODAY): ADOPT (directional), must-read.** Ablates four nested Codex-based EWM agents: **verification ranks first in all four settings**; flexible executable-model WITHOUT verification loses to a plain textual baseline; **fixed-interface + verification + simplification with gpt-5.6-sol solves every public game (~99%)** (author flags public-set saturation). This is the exec-sims author we already pivoted toward, independently confirming our plan-execute-verify recipe. Direct read on our step-0 aborts: EWM-minus-verification underperforming textual baselines is exactly what phase misalignment produces — the fix per this ablation is MORE verification + a FIXED world-model interface, not less EWM. (Also likely the mechanism family behind discussion #727629's "99%" claim — same number, same week.)
- **arXiv:2607.15524 — Recursive Harness Self-Improvement (Jul 20): ADAPT.** Prompt-level harness spec refined via pairwise feedback over its own revision history, no weight updates; low-effort agents beat max-effort baselines at −60% cost via better context management. Formalized version of our evolve_claude.py loop; the pairwise-over-revision-history selection rule is cheap to adopt. Caveat: synthetic ML tasks, not games.
- **arXiv:2607.15193 — Plover plan-centric steering (Jul 17): ADAPT (narrow).** Failures are structurally repairable when plans stay visible and interventions are localized → supports treating EWM step-0 abort as localized plan-REPAIR (edit + re-verify) instead of full abort/replan. Human-in-loop part irrelevant.
- **Backfill for the state-aliasing blocker (older, never swept): arXiv:2605.05583 Belief Memory (probabilistic multi-hypothesis ledger, Noisy-OR updates under partial observability) + arXiv:2602.09138 PABU (history as noisy proxy for latent state): ADAPT both** — closest published mechanisms to the latent-state audit consumers; read before finalizing audit design.
- IGNOREs: 2607.15660 ToolVerse (training-based), 2607.15550 SeerGuard (trained safety WM), 2607.15901 DSWorld (training-heavy), 2607.15516 cache-aware compression (API-cache cost model, not local vLLM).
- Categories with nothing new: test-time adaptation for agents; memory/selective-injection; **72B-AWQ throughput anchors (still none — A17 bench remains the reference, self-measured)**.

## §1d Same-day artifacts (for R16)

### LATENT-STATE AUDIT COMPLETE — R15's blocking prereq DISCHARGED, root cause FOUND
`scripts/latent_state_audit.py` (selftests PASS incl. synthetic hidden-counter recovery), protocol `learnings/war_room/latent_state_audit_protocol.md`, results `runs/latent_state_audit/report.{md,json}`. 200 streams / 33,777 actions across 8 pulls.
- **Headline: a hidden modular counter of actions-since-RESET (parity or mod-3/4/5) IS the state-aliasing mechanism.** 11/14 aliased games resolve to det ≈ 1.000 with a 1–3-bit phase augmentation. It is deterministic drift, not stochasticity (consistent with N5's 0/25 divergent).
- **Verdicts: 11 CLEAN** (ar25 bp35 ft09 lf52 lp85 ls20 r11l sp80 su15 tn36 tu93) / **11 ALIASED-RESOLVABLE** (phase-augment; incl. sb26 vc33 s5i5 tr87) / **3 ALIASED-UNRESOLVED** (g50t sk48 m0r0).
- **Explains all three R15-linked failures:** (1) predict-metric 0.465 — aliased (frame,action) keys have no-effect on one phase, effect on the other (wa30 98/105, sc25 43/44); (2) N5 prune bug — no-ops advance the phase, dropping leading no-ops desyncs it (sc25, the original abort case, is mod5-aliased); (3) EWM step-0 aborts split into TWO causes: phase-blind sims on aliased games (s5i5/sb26/vc33 — fixable via phase augment or resync) vs plain bad sims on CLEAN games (lf52/lp85/sp80/su15, step_acc 0.13–0.56 — need sim repair, aliasing work won't save them).
- **Consumer rulings:** EWM ship-now carriers = **ft09/tn36/tu93** (step_acc 0.92–0.97 AND clean/resolvable); resync-before-abort WILL work on phase-aliased games (deterministic counter re-anchors on re-read) — and the cheaper permanent fix is the 1–3-bit phase variable in sim state; banking prefix-splice/prune is legal ONLY in the 11 CLEAN games, everywhere else full-replay-from-RESET with zero pruning.

### R16 republication FILED
`learnings/war_room/grinder_design_R16_republication.md` — all 18 in-scope R15 directives discharged (table in its §16; R15 #13 discharged in the A17 scope doc, cross-referenced). Headlines: (d)-less stack ceiling **+0.21 rail** (banking-approved branch B+; unconditional B− = +0.09 with the generous haircut, +0.06 strict — both stated); **α = 0.05 one-sided** (Bonferroni rationale abolished with the single binding look; min clean wins 7→5); P(pass) B+ ≈ 0.2, B− ≈ 0.05 with sealed dismantle branch (Δlc ≤ −0.10 → (f)-only; null trip-rate 0.24 published); per-window guard recalibrated to SE-based z=1.834 (familywise false-kill 0.097 vs old 0.59); **(c)+Reki KILLED** (forfeits under MDE/2; observable-state keying = the exact A18-refuted premise; resurrection only via audit-verified ≥0.99 keying + Reki-keyed predict_metric ≥0.90); **SENTINEL_BUDGET = 150**/level-attempt proposed (token-implied capacity 147–156 across 3 certified seeds; mandatory B=150 canary re-run pre-seal; zero-events-on-long-run ⇒ VOID not FAIL); W0 control band n=4 pooled {war_eval v1/v2/v3 + w0_s1} with sealed 2-extra-seed fallback; EWM Stage-1 re-priced +0.02–0.18 rail (central +0.08, down ~2.5×; tr87 = only surviving new-clear; Stage-1 gate now exists: sim-derived-state BFS ≥0.70 @ L≤10 on ≥3/5 carriers); quota ledger fits A14 cumulative look + A17 in ≤24.2 of 30 GPU-h/wk.

### A17′ amendment DRAFT filed
`learnings/preregistration_amendment_2026-07-20_A17.md` (DRAFT — NOT FILED; seals on R16 sign-off). Repaired gate disjunction (capability AND [action-parity OR beats Σ null_adj + 1-margin]); comparator = per-game MAX both sides (Σ 27B MAX = 6 → capability bar ≥ 8); frozen null_adj walk with worked examples (Σ null_adj = 4 @ ρ≤2.5 / 3 @ ρ≤3.0, procedure seals for out-of-band ρ); marginal-seed rule; blocking pre-push tests (hermes tool-call round-trip, reset-path byte-identity, preflight); NO-GO closes war-v4 for the campaign, "materially different artifact" defined exhaustively (§8.2 is the one clause needing panel judgment).

## Open questions → R16 (full 5-reviewer; A14 SEALS on this round)
1. **A14 gate seal** on the republished arithmetic: (d)-less §2/§4, α=0.05 one-sided, P(pass) B+/B−, dismantle branch, guard z=1.834. This is the sealing circulation — confirm or object now.
2. **SENTINEL_BUDGET = 150 ruling** (the unsealed design decision blocking today's sentinel pushes). Author lean: accept 150 + the mandatory pre-seal B=150 canary re-run; note the VOID-not-FAIL zero-events clause.
3. **(c)+Reki kill** — ratify the kill + sealed resurrection path. Note the latent-state audit (§1d) landed the same day and already gives the per-game keying data the resurrection clause requires: 11 games have ≥0.99-determinism augmented keys. Does the panel want the resurrection evaluated NOW against the audit, or does the kill stand until someone files the Reki-keyed predict_metric ≥0.90 prong?
4. **EWM Stage-1** — ratify the re-price (+0.02–0.18 rail, central +0.08), the Stage-1 gate, resync-before-abort as contract v1.1, and the audit-informed carrier set (ship-now: ft09/tn36/tu93; phase-augment path for aliased games; sim-repair needed on lf52/lp85/sp80/su15 despite CLEAN engines).
5. **Banking implications of the audit** — prefix-splice legal only in the 11 CLEAN games; full-replay-from-RESET elsewhere. Does this change the A16 online retry policy scope?
6. **A17′ draft sign-off** (§1d) — esp. §8.2 "materially different" definition. On sign-off it files as preregistration_amendment_2026-07-20.md and the screen proceeds this week.
7. **W0 control band n=4 pooling** — legal, or trigger the sealed 2-extra-seed fallback?
8. Dream digest 2026-07-19: recency-only, skills_scored=0 as sealed expectation predicts — reviewed in the republication §15; nothing actionable claimed. Object if disagreed.


--- END OF PART 5 ---

=== END OF R16 CIRCULATION (5 parts, all END lines above must be present) ===

## END OF PROPOSAL ##
=====================================================================

OUTPUT FORMAT (exactly this structure):
## Summary (2 sentences)
## Objections
For each: [SEVERITY] title — body (2-5 sentences, specific)
(minimum 3 objections in round 1; in later rounds, review your prior objections'
resolution first, then add new ones only if real)
## Questions for the authors (numbered)
## What I cannot judge
## Verdict: ACCEPT | MAJOR-REVISION | REJECT
## Score: N/10
