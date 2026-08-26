# R24 review — methodology (reviewer #4)

## Summary

The lane selection in §4 is well-argued and I would vote to authorise it on the evidence as
presented; what I cannot vote for today is §6, because R24 is asked to *seal* it, and I
measured the inherited K3 gate against this campaign's own `runs/null10` and found it fails a
**true null arm 51.1% of the time** — so the one push the proposal buys returns close to a coin
flip, and the lane-(c) falsifier in S1b is defined on the eleven games that cannot exhibit the
failure mode while excluding both games that actually failed. These are not gatekeeping
objections: they came with numbers off the repo's own disk, and every one of them is fixable
offline, at $0, before the seal — which is exactly the shape of revision that should terminate
in one pass rather than another round.

---

## Objections

### [FATAL] The inherited non-harm gate has a measured 51% false-FAIL rate under a true null; "inherited verbatim" is rhetorical, not legitimate

§6.2 and the §6.4 objection table rest the gate's credibility entirely on provenance — "inherited
verbatim from the A22 M1/sentinel A5/A8 thresholds … cannot be accused of being retuned to fit."
Un-retuned is not the same as calibrated, and I calibrated it. `runs/null10/` holds ten
identical vanilla runs over the same 25 games; the 90 ordered pairs give the exact null
distribution of the P1 statistic (one run paired against one run, per game):

| statistic | null distribution | inherited gate | type-I error |
|---|---|---|---|
| mean Δlc | mean 0.000, sd **0.1223**, 5th pct **−0.200** | ≥ −0.128 | **12.2%** |
| worst-game Δlc | modal value **−2** (−1:44, −2:41, −3:4, 0:1) | ≥ −1.0 | **50.0%** |
| either leg (K3 FAIL) | — | conjunction | **51.1%** |

Per-game Δlc sd is **0.699**. A −2 on some game is the *modal* null outcome, so the worst-game
leg carries essentially zero information and the K3 conjunction is a coin flip. The provenance
is worse than uncalibrated: `runs/sentinel_eval_v1/screen_report.md` L5 shows −0.128 originated
as the **observed harm point-estimate of the war-v1 arm** (mean Δlc −0.128, sd 0.392, sign-flip
p = 0.9495 — a *non-significant* result promoted into a decision threshold), and it was estimated
against the **10-run-averaged** `null10` baseline, whereas §6.2 applies it against the
**single-run** `runs/kernel_pulls/war_eval_v1/`. Per-game Δ sd goes 0.392 → 0.699 (1.78×) across
that substitution. Same number, different measurement, different operating characteristics —
verbatim inheritance transported the digit and discarded the calibration.
**Fix, free and offline, before the seal:** recompute both legs from the 90 null pairs. A
defensible one-sided 5% pair is **mean Δlc ≥ −0.200** and **(# games with Δlc ≤ −2) ≤ 3**
(null P(≥3) = 4%, P(≥4) = 0%), and the seal must state the measured type-I rate of whatever it
adopts. Until §6.2 quotes an operating characteristic, it is not a gate.

### [FATAL] S1b's kill condition selects on the outcome: it excludes both documented failures and tests a proposition already recorded as proven

§4/S1b gates lane (c) on "step-0 `frame_divergence` must clear on the 11-game prefix-splice-safe
set", and §3(c) makes the same set the kill condition. That set is not neutral. Per
`learnings/panel/r16_circulation.md` L946–971 and L1258 it is *exactly* the CLEAN games
(determinism 1.000, no aliasing) — the sub-population in which phase desync is impossible by
construction. Both games that actually threw step-0 `frame_divergence` — **sc25** (mod-5 aliased)
and **m0r0** (ALIASED-UNRESOLVED, det 0.618), confirmed in
`runs/war_eval_v1/bank_fire_validation.json` — are FULL-REPLAY-ONLY and therefore **excluded
from the gate**. Compounding it: the operation under test is full unpruned replay, and L1258
already records "N5: full unpruned replay survives on all 25". So S1b as written re-tests a
recorded positive on the easy subset, and lane (c) survives even if sc25 and m0r0 abort exactly
as before. **Fix:** gate on **sc25 and m0r0 specifically** under full unpruned replay from RESET,
report all 25, and pre-register in the seal that clearing the 11 CLEAN games is a null result,
not a pass. As written S1b is unfalsifiable in the only direction that matters.

### [MAJOR] The A22 death record survives, but the "monotonic in eviction pressure" claim — the framing sentence of the whole proposal — does not

R24 is asked to confirm §1.1 as a formal record and adopt §2.2 as a standing ordering constraint.
Against the null above: v2 (−0.320) and v2.1 (−0.360) sit outside the null mean-Δlc distribution
entirely (P(mean < −0.320) = 0/90), so **the lane's death on the mean leg is sound** — say so, and
drop the worst-game leg from the record, because v2.1's cited "ar25 −2, sc25 −2" is the modal null
outcome, not evidence. What does not survive is the monotonicity: the steps −0.200 → −0.320 →
−0.360 are 0.120 and 0.040 against a null screen sd of 0.1223, and because all three share one
baseline the SE of the *differences between arms* is larger still (≈ 0.198). v1's −0.200 sits
exactly at the null 5th percentile. v2.1's own sign-flip p = 0.0781 is not significant. Likewise
ρ(`evicted_chars`, Δlc) −0.13 → −0.403 described as "nearly 3× stronger" is a Fisher-z difference
of 0.30 with SE 0.30 (p ≈ 0.33), and is one correlate among several reported. §2.2's discriminator
("harm monotonic in eviction pressure, ρ = −0.403") is the load-bearing empirical claim behind
both the Tycho reconciliation and the ordering constraint that sequences away all context work.
Restate it at the strength the data supports: *three screens moved the mean consistently negative;
the dose-response is not resolved at this n.*

### [MAJOR] K4's floor of 0.15 is pre-registered before the free measurement that calibrates it, and the estimator is undefined

§6.4 calls `namespace_reuse_rate < 0.15` one of "two independent pre-registered falsifiers." But
§5.4 *schedules* the transcript-forensics pass that would measure the duck's existing
affordance-trigger and cross-turn name-reference behaviour — and §6.2 fixes the floor anyway. The
statistic has no baseline arm (it is a one-arm descriptive with an absolute cut), no stated
denominator (all turns, or `python`-tool turns?), no stated aggregation (mean of per-game rates,
or pooled over ~1,686 turns?), and no null spread. Addendum B3 correctly identifies K4 as the
right instrument for the open weak-model question — which makes leaving it uncalibrated worse,
not better. **Fix, free, zero pushes, already on the plan:** run §5.4's forensics *first*, publish
the baseline *attempted* cross-turn reference rate with its per-game spread (attempted reuse is
measurable in existing transcripts even though the ephemeral sandbox makes it fail), then set the
floor as baseline + δ with δ stated, and freeze the estimator definition in the seal. Do not seal
this number today. Fold in B7's code-well-formedness pre-mortem while the transcripts are open.

### [MAJOR] The promotion threshold has migrated off the sealed frozen control onto a growing record, in the favourable direction, with the seal date left open

`runs/lb_ground_truth.md` and `boristown_ab_prereg_2026-07-29_DRAFT.md` §3 state that A/B control
parameters are **FROZEN at n=15 (0.9727, 0.1343)** and that "later fillers accrue to the record,
not to the sealed control." §6.5 and Addendum A1 compute against the record instead. I reproduced
all three: frozen control → **1.1042**; record n=25 → **1.0823**; record n=26 → **1.0772**. The
substitution lowers the promotion bar by **0.027** and it keeps falling, because the campaign
itself banks a filler draw every night below the frozen mean. §6.5 then says "recompute at the
then-current n at seal time" — leaving the seal date, and therefore the bar, to the party being
gated. That is optional stopping on the gate. **Fix:** state which control is authoritative; if
the record supersedes the frozen one, that is a prereg amendment and must be argued as such, and
the n and date must be pinned *now*, not "at seal time."

### [MAJOR] The screen has no success criterion, yet S3 promotes on "PASS" — and ≥10 statistics come off one seed with no family-wise control

P1 PASS = non-harm (K3) + engagement (K4). Neither measures benefit. An arm with exactly zero
effect passes both provided the model defines any variable — and per the first objection, K3
*fails* a null arm half the time, so the one authorised push returns a near-coin-flip on a
criterion that cannot detect the thing the lane is for. §6.4's "reads mechanism, not lift" is
honest, but S3's "P1 PASS ⇒ S4" then spends the next cycle on that reading. Separately, the run
emits mean Δlc, worst-game Δlc, `actions_per_level_completed`, `namespace_reuse_rate`, fallback
count, live-child/orphan/`RLIMIT_CPU` canaries, plus the A22-precedent M2 ratios, M3 reprop and a
pearson correlate — ten-plus numbers off 25 games, no correction, and a documented history of
post-hoc narrative built on whichever moved (M3 at −6.49pp, p = 0.0001, from a mechanism that was
provably absent; ρ = −0.403). §5.1's decision to drop the M3 arm is the right call and should be
generalised rather than treated as a one-off. **Fix:** name exactly one decision statistic in the
seal, label every other output non-inferential, and fill the blank in §6.2 — "above a
pre-registered fraction of turns, voids the arm" is a placeholder in a document being sealed today.

### [MAJOR] The co-primary is a ratio whose denominator is the primary endpoint, quoted as a single-run point estimate

§2.3 and §6.2 promote `actions_per_level_completed` with "baseline **165.4**" — four significant
figures from one run, no interval. The same statistic across null10's ten identical runs spans
**195.5–322.9** (mean 234.2, sd 40.9, **CV 17.5%**). Worse, the denominator *is* the primary
endpoint, so the metric is mechanically anti-correlated with it: any arm that completes fewer
levels can look more efficient. That is precisely the Goodhart hazard §6.4 claims to be guarding
against, imported as the guard. **Fix:** quote an interval from null10, and redefine the
efficiency metric on levels *attempted* (or actions-to-first-clear per level) so the endpoint is
not in the denominator. Keeping it non-gating at first screen is necessary but not sufficient.

### [MAJOR] S1's falsifier is numerically undefined and infinitely re-runnable — the free test needs its analysis frozen harder, not softer

§4/S1's gate reads "Carrier set must **expand beyond ~4 games**." A tilde has no place in a
pre-registration. Over 24 sims there is no stated accept predicate for an "accepted transition
match", no coverage-accounting rule, no on-trajectory sampling rule, and — the real hazard — no
cap on how many protocol variants may be tried. A $0 offline test that can be re-run all week is
*more* exposed to analytic flexibility than the one that costs a push, not less; the cost
asymmetry runs opposite to the discipline the proposal applies. Note also that r16_circulation
L1255 already lists **11** EWM Stage-1 safe carriers, which needs reconciling with "~4" before any
bar is set. **Fix:** an integer bar (e.g. ≥8/24), a frozen accept predicate and abstention/coverage
accounting, and a pre-registered cap of one protocol variant, with any further variant reported as
exploratory and barred from the S5 gate.

### [MAJOR] "The field is quiet" is used as evidence in two places by an instrument with unmeasured recall — and Addendum B2's correction is absorbed without re-deriving the conclusion

§3(c) lists "Field 3 is near-empty across two consecutive sweeps" as evidence *against* a lane, and
§5.2 uses the same move to argue the compaction death record is sealable. Addendum B2 has now shown
the instrument missed a ~20-paper cluster; B8 records one search returning 152 hits of which 40 were
read (26%). Absence of publications was never evidence about mechanism merit even when accurate, and
it is now demonstrably unreliable — yet B6 still concludes a third quiet sweep makes §5.2 "sealable
as written," and B9 reports the lane ranking "UNCHANGED" after its stated premise was withdrawn. A
conclusion whose premise is falsified must be re-derived in writing, not re-asserted; B2 is right to
demand the down-rank be restated on the generalisation rail, and the panel should require those
words. The same addendum then commits the error it warns against by citing VaG (2608.05810,
cross-task skill pools, non-monotonic accumulation) against lane (c), which is within-game trace
replay — a different object, as B2 itself says one sentence earlier. **Fix:** strike
absence-of-literature from every evidence table; require each sweep to report
enumerated-vs-read counts so a null result carries its own coverage bound.

### [MINOR] §A1's ledger arithmetic is correct; two small defects in how it is assembled

I reproduced it from the draw list: n=26 → mean **0.9365**, s **0.1539** (A1 quotes 0.1540 —
rounding), and 0.9365 + 1.7011 × 0.1540 × √(1/4 + 1/26) = **1.0772** (A1: 1.0773). Both fine, and
the n=25 → n=26 correction of the stale §6.5 figure is properly made. Two defects: (i) the df is
pooled (n₁+n₂−2 = 28) while the s is control-only (df 25) — using df = 25 gives 1.0778, immaterial
in size but it shows the formula was copied rather than derived; (ii) √(1/4 + 1/n) assumes the
gated arm's per-draw σ equals the control's, which is asserted nowhere. The closed war arm (n=5)
gives s = 0.1309 against the control's 0.1540, which is reassuring — state it rather than assume it.
The arithmetic is used correctly; the *choice of control* is the problem, and that is objection 5.

### [MINOR] A5 understates the independence problem, and the base rate is now unusable

The disclosure is correct and the ruling that round-over-round score trends are invalid for R24 is
right — I adopt it and nothing in this review rests on a comparison to R10–R23. But byte-identical
prompt files preserve the *stimulus*, not the *independence*: the reviewers now run as subagents
**inside the campaign session being reviewed**, rather than on the separate KAOS rail, which is a
different and larger change than the model swap alone. Second-order consequence worth recording:
with the model changed, the campaign's own 11-round base rate (zero ACCEPTs) can no longer be used
to calibrate what an R24 verdict means in either direction. **Cheap fix:** re-run one archived prior
packet (R23 is the obvious choice) under the new model to obtain an anchor, and stratify R24
separately in any retrospective.

### [MINOR] Addendum A3's Feng report is correctly de-rated, and the packet should keep it that way

A3 does the de-rating properly (177th, no ablations, four confounded mechanisms — design evidence,
zero efficacy evidence) and pre-empts overreading. I concur and add one methodological point for the
minutes: Feng's report is the only 27B datum in three sweeps, which makes it the highest-temptation
citation in the packet. If it enters the P3 prereg at all it enters as a *feasibility* claim about
the substrate and may not be cited in support of any effect size or in the S4 gate rationale.
Likewise B3's second-hand "filesystem store degrades 32%" must stay out of the seal until directly
read, as B3 itself says.

---

## Questions for the authors

1. Was any null-vs-null screen ever run before −0.128 and −1.0 were adopted as gate lines? If so,
   where is it; if not, does the panel accept the null10-derived operating characteristics above
   (12.2% / 50.0% / 51.1% type-I) as the calibration of record?
2. §6.2 pairs against a single run (`war_eval_v1`), while the threshold was estimated against a
   10-run mean (`null10`). Is there a reason not to pair P1 against null10 directly, or against a
   ≥3-run baseline, given the baseline runs already exist on disk?
3. Under S1b, if the 11 CLEAN games clear but **sc25 and m0r0 still abort at step 0**, is lane (c)
   PASS or FAIL? The text implies PASS. Is that intended?
4. What is `namespace_reuse_rate`'s denominator, aggregation, and measured baseline in existing
   transcripts — and why is 0.15 sealed before the §5.4 forensics pass that would supply all three?
5. Which control is authoritative for any future scored-draw gate: the frozen n=15 (1.1042) or the
   rolling record (1.0772 today, falling)? If the latter, at what pinned n and date?
6. What outcome of the P1 screen would constitute evidence that state-externalisation **helps**, as
   opposed to "did not measurably hurt and the model defined some variables"?
7. §6.2's fallback-count VOID clause reads "above a pre-registered fraction of turns" — what is the
   fraction?
8. S1's bar is "beyond ~4 games" while r16 L1255 lists 11 EWM Stage-1 safe carriers. What is the
   integer bar, and how many L0 protocol variants may be run before the result is exploratory?
9. §1.1's monotonicity and the ρ −0.13 → −0.403 "3× stronger" claim: will these be restated at the
   supported strength before §1.1 is entered as a formal record?

## What I cannot judge

Whether lane (a) is the right *scientific* bet — the three-team convergence argument, the
portability of Tycho's schema, whether a 27B actor can carry a programmatic world model, and the
substance of AppDeltaWorld / MASS / 2608.06370 — is outside my remit; I defer to the
prog-synthesis, LLM-agents and RL-planning reviewers. I also cannot judge the engineering claims:
whether `RLIMIT_CPU` re-accounting is tractable, whether a long-lived sandbox child is safe at
concurrency 16, whether warpack monkeypatching genuinely avoids notebook drift, or the
`provenance:scratch-built → ERROR` execution risk that A4's fingerprint table raises for S2. I did
not verify that all 25 recorded traces and all 24 sims are present and replayable (A6 flags this as
unverified; I confirmed `exec_wm/sims/` and `exec_wm/observations/` exist and that
`bank_fire_validation.json` covers **4** games, not 25 — the S1b re-run therefore extends an n=4
harness to n=25, which is new work, not a re-run). I take no view on the §5.3 governance rulings
except to note that ruling 2 (provenance de-rating) is methodologically sound and B5's 44pp figure
is a reasonable quantitative floor for it.

## Verdict: MAJOR-REVISION

On the panel's absorbing-state question, my ruling, stated plainly. The critique is legitimate:
eleven rounds of MAJOR-REVISION with no terminating condition is itself a methodological defect,
and the correct response is to make the verdict terminate by construction — the seal should list
the exact items that must change, with a rule that a re-submission satisfying that list is adopted
without a further panel. I would apply that here. I would also have voted ACCEPT on §4's lane
selection, §5.1's drop of the M3 arm, §5.3's rulings and §5.4's no-regret list on the evidence
presented; the reasoning there is careful, the confound discipline is real, and §6.4's
anticipated-objections table is unusually honest for this campaign.

What blocks ACCEPT is narrow and specific: R24 is asked to **seal §6 today**, and §6 contains an
instrument I measured at 51% type-I error on the campaign's own on-disk null, a lane-(c) falsifier
defined on the eleven games that cannot exhibit the failure mode while excluding both games that
did, an adoption floor fixed before the free measurement that calibrates it, and a promotion
threshold that has drifted off its sealed control in the favourable direction with the seal date
left open. I did not go looking for these; I ran the null because the "inherited verbatim" defence
invited the check, and it came back at a coin flip. Sealing §6 as written would bank an
uncalibrated instrument for the next three lanes the way −0.128 was banked for the last three.

Every fix is offline, $0, zero pushes, and mostly already scheduled: recalibrate both K3 legs from
`runs/null10` (90 pairs, one afternoon), re-point S1b at sc25 and m0r0, run §5.4's forensics before
fixing K4's floor, pin the control n and date, and pick one decision statistic. That is one
revision pass, not another round. **Authorise S1 and the lane selection now; hold the S2 seal for
the recalibration.**

## Score: 6/10
