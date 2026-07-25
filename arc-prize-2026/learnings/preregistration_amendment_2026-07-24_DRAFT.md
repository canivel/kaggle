# Preregistration amendment 2026-07-24 — **DRAFT** (for R20 ratification; NOT SEALED)

**Status: DRAFT.** Build-rail intent only (A22). Nothing below is sealed, and no
sealed/amendment/prereg file is modified by this document. It becomes binding only
if a panel round (R20+) ratifies it. Machinery and all numbers:
`runs/r19_hygiene/r19_hygiene_stats.py` → `runs/r19_hygiene/r19_hygiene_stats.json`
(seed 20260724, 200k MC reps per cell). Responds to the R19 directives in
`learnings/panel/round19/{rl-planning,methodology,systems}.md`.

Ledger inputs: frozen control n=10 (chronological)
{0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82} → mean 0.975, s 0.1557.
Pooled n=15 adds war {0.91, 1.08, 0.88, 1.05, 0.76} → mean 0.962, s 0.1444.
LB state per `runs/lb_ground_truth.md` (canonical, API-verified).

---

## (a) Declared tail model — one model for all ledger z/p, everywhere

**DECLARED MODEL (methodology R19 Q2, one answer, applied everywhere):** every
ledger z/p, exceedance probability, and E[max] figure is computed under the
**t-predictive**: for a control sample of size n with mean x̄ and sd s, a new draw
X satisfies (X − x̄) / (s·√(1+1/n)) ~ t(ν = n−1). Multi-draw quantities (E[max],
touch probabilities, rule error rates) use the **joint** form: (μ, σ²) drawn from
the normal–inverse-χ² posterior (noninformative prior), then iid normal draws —
this preserves cross-window dependence through the shared unknown (μ, σ) and has
exactly the t-predictive as its single-draw marginal. No Gaussian-z, no ad-hoc GPD.

**Recomputed under the declared model (frozen n=10 unless noted):**

| quantity | value |
|---|---|
| 0.71 draw: t, one-sided p (frozen) | t = −1.62, **p ≈ 0.070** (pooled: −1.69, p ≈ 0.057) |
| — old Gaussian p (superseded) | 0.044 |
| P(single filler draw ≥ 1.33) | **0.029** (pooled 0.014) |
| P(single filler draw ≥ 1.44) | **0.0096** (pooled 0.0032) |
| P(single filler draw ≥ 1.47) | **0.0071** (pooled 0.0021) |
| P(single filler draw ≥ 1.49) | **0.0058** (pooled 0.0016) |
| E[max over 101 remaining windows] | **1.403** (median 1.378; q10–q90 [1.24, 1.59]; pooled E[max] 1.346) |
| P(101-window max ≥ 1.44 / 1.47 / 1.49) | **0.329 / 0.262 / 0.225** (pooled ≥1.44: 0.176) |
| P(101-window max ≥ 1.61 / 1.80) | 0.087 / 0.020 |
| per-window opportunity cost ΔE[max] (CRN) | **≈ 0.0006** E[max]-equiv |

**FORMALLY RETIRED:** the "P(touch 1.44) ≈ 0.18" Gaussian figure
(`learnings/stuck_review_v2_2026-07-23.md` §0) and every Gaussian-z in prior
briefs are superseded. Note the retirement is not cosmetic: under the declared
model the frozen-control touch probability at 1.44 is **0.33**, nearly double the
retired figure — parameter uncertainty fattens the predictive tail in the
direction *favorable* to the filler lottery. The "exploration is nearly free"
claim also survives audit under the same single model (ΔE[max]/window ≈ 0.0006):
both of R19 rl-planning's demanded claims now derive from one fitted model.

## (b) Prospective pooling rule (draws 16+) and trend/changepoint check

**Pooling rule (prospective, generalizing the one-time 0.71 exclusion):**
1. The **frozen control stratum** admits only draws of the frozen-fork filler
   composition, byte-identical (or hygiene-graft-equivalent per (i) below) to
   `canivel/arc3-duck-repro` v3. These update the frozen ledger (n=10 → n+).
2. The **pooled ledger** = frozen stratum + closed-arm strata whose composition
   differs from the filler only by mechanisms subsequently judged
   score-distribution-equivalent (war n=5 qualifies by its closed A9 record).
3. **Exclusion rule:** any draw from a composition with a live mechanism diff
   (an open arm) enters *only its own arm ledger* — never frozen, never pooled —
   until the arm is closed AND a distribution-equivalence memo is filed. This is
   the 0.71 rule (`learnings/daily_brief_2026-07-24.md` "Ledger" §), generalized.
4. Strata are never merged retroactively without a written memo; the pooled n
   used in any rule is stated at rule-fire time.

**Trend/changepoint check on the 15 time-ordered pooled draws (published):**
Mann-Kendall on the frozen n=10 chronological series: S = 9, z = 0.72,
two-sided **p = 0.47**; CUSUM (20k permutations) stat 0.579, **p = 0.72**.
Pooled n=15 under the documented interleave assumption (f1–f5, w1–w4, f6=1.33,
w5, f7–f10): MK z = 0.60, **p = 0.55** (alternative w5-before-f6 order: p = 0.49);
CUSUM p = 0.93. **Verdict: no trend, no changepoint — stationarity is not
rejected on any ordering; the pooled ledger remains legal to use.** (The war
interleave order is assumed, not logged per-draw; both plausible orders agree.)

## (c) Entry-bar amendment (BEFORE exploration draw 2/12) — exact rule text

R19's critique (rl-planning MAJOR, methodology funnel MAJOR) is conceded
(already conceded in `learnings/war_room/sentinel_disposition_2026-07-24.md`,
Bookkeeping). Proposed rule text:

> **A21-E (entry bar, draws 2–12).** An arm may occupy an exploration window only
> if its entry case, filed before the window, contains ALL of:
> (1) build-rail canary PASS and non-harm screen (unchanged from A21);
> (2) **evidence aggregation:** one stated prior over the arm's scored-rail
> effect that cites EVERY prior observation on the composition — eval-rail
> seeds, screens, mechanism verdicts — with the dependence structure stated
> (replicate seeds within one rail count as ONE rail-level signal, per the
> sentinel disposition memo); an arm whose aggregated prior is net-negative is
> barred unless the case pre-registers an explicit rail-transfer hypothesis that
> the scored draw would test;
> (3) **positive right-tail evidence:** at least one of (i) an eval-rail primary
> point estimate > 0 on ≥ 1 certified seed, or (ii) a written mechanism story
> with an identified DEPTH channel — a named mechanism by which the arm completes
> MORE LEVELS (not fewer actions, not efficiency) under the completion-weighted
> scorer.
> Exploration windows buy exceedance probability; non-harm alone is no longer
> sufficient.

Under this bar the sentinel would have been barred at entry (both eval seeds
negative; efficiency observable with no depth channel), saving window 1/12.

## (d) Promotion gate re-derived in exceedance currency

The sealed +0.06 mean-lift promotion bar is unreachable at n ≤ 11 windows
(≈ 53 arm draws would be needed for 80% power at α = 0.05; rl-planning flagged
this twice). Proposed replacement, denominated in the currency the LB actually
pays — P(arm draw > current best) — with error rates under the declared model:

> **A21-P (promotion, replaces +0.06 mean-lift for arm promotion).** Within an
> arm's first ≤ 5 scored windows, the arm is PROMOTED to default nightly draw iff
> **(≥ 2 draws > 1.33) OR (≥ 1 draw ≥ 1.44)** (thresholds frozen at rule-fire
> time as: current LB best; bottom of the wall band). A promoted arm remains
> subject to the harm-pause (e) and reverts to filler if 5 consecutive
> post-promotion draws all fall below the control median.

Error rates (joint predictive, 200k reps): **false promotion (arm ≡ filler)
= 4.9%** over the 5-window course; power = 13.8% at true mean +0.10, 22.5% at
+0.15, 35.5% at +0.20. Two honest notes: (i) power against modest shifts is low —
that is the physics of ≤ 5 draws, and the gate is at least *reachable*, unlike
the sealed bar (power ≈ 0 by construction); (ii) the 4.9% false-promotion rate is
cheap in exceedance currency, because a falsely promoted filler-equivalent arm
draws from the same distribution it displaced (residual cost ≈ left-tail risk,
which the retained harm-pause caps). The gate's asymmetric job — blocking
*harmful* arms — is done by the harm-pause, not the promotion bar.

## (e) Harm-pause calibration and resume path

Recomputed under the declared model (the ~13% Φ-figure is superseded):
- **P(pause | healthy arm ≡ filler) = P(draw < 0.80) = 15.6%** (frozen; pooled
  14.8%). One in six healthy arms will trip the pause on its first draw — the
  pause is exposure control, not inference, and MUST have a resume path.
- **Power:** P(pause | truly harmful arm, mean 0.85) = **38.3%**; mean 0.80 →
  50%; mean 0.75 → 61.7%. A single-draw pause is a weak detector of modest harm;
  its value is bounding exposure, and the entry bar (c) is the real filter.

> **A21-R (resume path, two-draw rule).** A paused arm may be granted at most
> TWO resume draws from the exploration budget, only with a written mechanism
> update explaining the pause-consistent story. RESUME (pause lifted, arm draws
> normally) iff both resume draws ≥ 0.80 AND their mean ≥ 0.90. Otherwise the
> arm is SHELVED — no third draw, ever, without a fresh A21-E entry case.

Calibration: P(resume | healthy) = **65.1%**; P(resume | harmful mean 0.85) =
**28.6%**; harmful mean 0.80 → 17.2%. (The sentinel does not get this path: it
is shelved on sealed eval evidence per the disposition memo, not on the pause.)

## (f) A21 allocation policy (the owed paragraph)

**Max concurrent arms: 1.** One open exploration arm at a time; no arm gets a
second window before its first is analyzed (existing A21 text, retained).
**Priority order for the remaining 11 windows:** (1) the A17-branch arm per the
branching rule below; (2) depth-lane/tr87-class compositions (only non-A17
depth-targeting line; prereg `learnings/war_room/tr87_confirmation_prereg_2026-07-24.md`);
(3) any new arm meeting the (c) bar with a depth channel; efficiency-channel
arms are deprioritized to last (doctrine: price ≈ 0). **Reallocation-from-paused:**
a pause immediately returns the arm's remaining windows to the pool; the next
arm in priority order may claim the next window without a new panel round
(entry case still required). **Front-load vs spread: SPREAD** — at ΔE[max] ≈
0.0006/window the exploration budget is cheap, and information (A17 outcome,
gold-cutoff drift) accrues over time; no more than 3 exploration windows in any
7-day span. **Branching on A17:**
- **GO →** war-v4 (Qwen2.5-VL-72B-AWQ) becomes the next arm. Its entry case must
  be filed BEFORE its first window (methodology Q6) and must contain: named
  non-score observables (per-game levels_completed from the pull, realized
  per-game action counts N₇₂B(g), realized ρ_action as diagnostic, heartbeat/
  liveness log), a null criterion (arm draw distribution ≡ frozen control;
  KILL/pause per (e)), and the v4 25-game × 3-seed ledger obligation from scope
  v2 §3 before any promotion claim.
- **NO-GO →** the depth-budget lane (tr87 re-entry, OBJ-H ratified) becomes
  priority 1; windows not claimed by a qualifying arm revert to filler (which is
  affirmatively a strategy at P(touch 1.44) ≈ 0.33, per (a)).

## (g) Nov-2 target denomination — gold-cutoff forecast and honest rank bleed

**Gold/cutoff series** (from daily briefs + discussions sweeps): 1.44 wall
(07-14 through ~07-20; the wall proper), gold cutoff ≈ 1.47 (07-22, rank 13;
07-23), ≈ **1.49** (07-24, top 13). Leader flat at 1.86 since 07-14.
**Fits to Nov-2** (107 days from 07-18): linear slope 0.0076/day → **2.25**
(exceeds the stationary leader; treated as implausible upper bound); saturating
exponential toward ceiling C: C = 1.61 → **1.61**, C = 1.86 → **1.80**,
C = 2.00 → 1.87. **Published forecast band: Nov-2 gold ≈ 1.60–1.90, central
≈ 1.80** (sensitivity = choice of ceiling; six days of data — wide honest band).
**What it does to "enough lift":** our 1.33 plus one frontier depth event
(+0.19–0.29, d4 reprice) reaches 1.52–1.62 — at or below the BOTTOM of the
Nov-2 band — so a single depth event is necessary but likely insufficient;
gold requires the A17-class jump or ≥ 2 stacked depth events, and filler-only
P(max ≥ 1.61) ≈ 8.7%, P(max ≥ 1.80) ≈ 2.0%.
**Honest rank bleed** (methodology MINOR conceded): rank series 40 (07-20),
44 (07-22), 45 (07-23), 49 (07-24) → OLS **2.1 ranks/day** (R² = 0.95). The
"~4 ranks/day" was a single-overnight delta and is corrected; 4/day is the
worst daily step, not the trend.

## (h) One-sentence items

1. **§6 falsification amendment:** the scored-regime item ("first exploration
   draw fired AND analyzed under its pre-registered rule") is hereby made
   LOAD-BEARING — the §6 disjunction is not satisfiable without it, and "draw
   fired" alone does not count.
2. **A25 prospectivity:** seal-qualifying rounds begin at **R18+**; R16 and R17
   do not count toward the two-consecutive-rounds seal condition.
3. **Seal-termination downgrade logging:** any downgrade of a seal-tracked named
   condition is logged in the amendment ledger with date, owner, and reason at
   the time of downgrade — never retroactively.
4. **Rule-of-three CIs:** the 29/29 mechanism count carries a 95% upper bound of
   **10.3%** on the true failure rate (exact Clopper–Pearson 9.8%); the 49/49
   count carries **6.1%** (exact 5.9%) — "perfect" operational records are
   compatible with ~1-in-10 / ~1-in-17 failure rates and are quoted with these
   bounds henceforth.

## (i) Fork policy — boristown classification and the re-baselining answer

Per `learnings/war_room/fork_diff_boristown_2026-07-24.md`: boristown
`agi-duck-harness-fast-eval` (public 1.47) is **byte-equivalent to our frozen
fork** — 12/22 cells md5-identical including every load-bearing cell, identical
solver dataset bytes (frozen pre-06-12), zero metadata mismatches; the ONLY
functional diff is a 25-line `wait_vllm_ready()` readiness gate that closes a
startup race our fork has. The claimed patches are deliberate no-ops. There is
no depth channel; 1.47 is a right-tail draw of (approximately) our own
distribution plus a slightly raised left tail.

**Classification (answers methodology Q3/Q5(iii), rl-planning Q6, systems MINOR):
adoption is a FILLER-REPLACEMENT with a hygiene graft — NOT a baseline change
and NOT an arm.** Distribution-equivalence is established by the byte diff, so
methodology's n=0 re-baselining scenario does not apply. Proposed policy:
1. **Adopt the vLLM readiness gate as a default hygiene graft** in all future
   builds (frozen-fork lineage, duckwar, sentinel-telemetry kernels), exactly as
   the (f) continuation became default hygiene: free left-tail insurance, zero
   interaction risk, no score-sensitive surface.
2. **Optional fresh-slug byte-fork** of boristown as the new filler slug
   (fresh-slug memory: long-lived slugs accumulate hidden state; forking gives a
   fresh slug for free). Honest EV = our distribution + slightly better floor —
   **+0.14 is NOT budgeted as systematic.**
3. **Control ledger CONTINUES (no n=0 reset)** with (i) a noted caveat that the
   readiness gate raises the left tail (early low draws like 0.82 may partially
   reflect the closed startup race, so the continued ledger is conservatively
   left-skewed vs the new filler), and (ii) the (b) changepoint monitor ARMED
   for the first 5 post-gate draws: MK + CUSUM run on the frozen stratum after
   each of those draws; a CUSUM permutation p < 0.05 forces a stratum split
   (pre-gate / post-gate) before any rule fires on pooled numbers. This is the
   monitored-continuation answer to methodology's re-baselining MAJOR: the diff
   evidence licenses continuation; the monitor bounds the cost of being wrong.

---
*Draft prepared 2026-07-24 for R20. Computation artifacts:
`runs/r19_hygiene/r19_hygiene_stats.{py,json}`.*
