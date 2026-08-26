# LB draw-process generative model - stress test of the 07-18 "variance bombshell"

Date: 2026-07-18. Audience: panel R14 (Q-A window-pricing, Q-B frozen-vs-war allocation).
Code: `runs/lb_process_model/lb_process_model.py` (deterministic, seeded; regenerates this
report). Raw fits/sims: `runs/lb_process_model/fits_and_sims.json`.

**Discipline statement.** This analysis is DESCRIPTIVE / SIMULATION ONLY. It consumes no
gate look; the sealed A5/A8 n=5 war variance look (after tonight's draw #5) is untouched.
All inputs are already-on-record artifacts: the 13 fully-instrumented local seeds
(`runs/null10/` seeds 101-110 + `runs/kernel_pulls/war_eval_v1-3`, warpack but
delta-lc-null-ish) and the LB ledger numbers already published in ITERATION_LOG.md.
Local CPU only; no pushes, no API spend, no submissions.

## 0. Data, scorer, model

- **Scorer:** exact RHAE mirror from `scripts/phase1_gate.py`, re-validated this run:
  max abs error **1.78e-15** over 1000 cross-checks vs Tufa's 500 stored runs.
- **Ledgers (verified vs ITERATION_LOG):** frozen5 {0.82, 0.89, 0.93, 1.02, 0.95} mean
  0.922 sigma-hat 0.0740; frozen6 (+1.33) mean 0.990 sigma-hat 0.1792; war4
  {0.91, 1.08, 0.88, 1.05} mean 0.980 sigma-hat 0.0997; pooled10 mean 0.986 sigma-hat
  **0.1455** (chi2 95% CI on sigma, df 9: **[0.100, 0.266]**).
- **Per-game fit:** empirical 13-draw distributions per public game (score +
  levels_completed), scored offline with Tufa null baselines. 13-seed grand mean 1.594
  (25 games). Heavy tails confirmed: ft09 mean 8.98 sd 8.87 max 28.6 (lc 0-3); vc33
  3.59/4.67/16.7 (lc 0-3); tn36 1.85/3.10/10.7; ar25 3.52/3.05/8.3; r11l 3.36/1.82/4.8
  (lc <= 1 - r11l cannot spike).
- **LB night model:** score = mean over the FIXED official 110-game set, one pass each
  (gap_forensics 2026-07-09). Official games proxied by cloning the public 25 to 110 slots
  (multiplicity 4-5, randomized per world). Difficulty calibration: multiplicative c to hit
  a target mean - c = 0.922/1.594 = **0.58** (out-of-sample target, pre-1.33 mean) or
  0.986/1.594 = 0.62 (all-10-draw mean); this is the ~0.55 ratio of position_analysis refit
  on the 13 seeds. TRIM variant (drop ft09/vc33/r11l/ar25 = "official mix has fewer
  ft09-likes") run as sensitivity. Correlation variants: **IND** (all 110 slots
  independent), **COR** (IND + additive common night effect, ANOVA estimate from the 13
  seeds: sigma_u,local = 0.168 -> ~0.10 at LB scale), **BLOCK** (one seed drives all slots;
  upper bound, degenerate). 20,000 simulated nights per combo.

## 1. Headline 1 - does the local model reproduce the 1.33?

**Only partially, and only with cross-game correlation aboard.** In plain terms: if nightly
LB noise were nothing but independent per-game pass noise over 110 fixed games, a 1.33 draw
was a ~1-in-100 event in our observed window and the bombshell would be evidence of
something the bench doesn't capture. Adding the *measured* local common-night effect
(and/or accepting that the true frozen-fork mean is nearer 0.99 than 0.92) makes 1.33 an
unlucky-but-ordinary right-tail draw.

| model (mean calib, corr) | sd(night) | P(night>=1.33) | P(>=1.33 in 6) | in 10 | in 13 |
|---|---|---|---|---|---|
| A scale->0.922, IND | 0.128 | 0.0011 | 0.007 | 0.011 | 0.014 |
| B scale->0.922, COR | 0.161 | 0.0069 | **0.041** | 0.067 | 0.086 |
| D scale->0.986, IND | 0.137 | 0.0079 | 0.046 | 0.076 | 0.098 |
| E scale->0.986, COR | 0.172 | 0.0251 | **0.142** | 0.225 | 0.281 |
| F trim->0.922, IND | 0.109 | 0.0003 | 0.002 | 0.003 | 0.004 |
| G trim->0.922, COR | 0.195 | 0.0200 | 0.114 | 0.183 | 0.232 |
| C scale->0.922, BLOCK (bound) | 0.284 | 0.135 | 0.580 | 0.765 | 0.837 |
| H scale->0.922, IND, +Tufa20 pool | 0.121 | 0.0006 | 0.003 | 0.005 | 0.007 |

- The **pooled-10 observed sigma-hat 0.1455 sits exactly inside the model bracket**
  (IND 0.11-0.14, COR 0.16-0.20). The model does NOT need new physics to match the ledger's
  dispersion.
- The **sigma-hat5 = 0.074 of 07-12 was a sampling fluke, not a property of the process**:
  under the model, P(5-night sample sd <= 0.074) = 0.05-0.14 (IND) and 0.04-0.07 (COR).
  Unlucky-tight, not impossible. Symmetrically, P(6-night sigma-hat >= 0.179) = 0.02-0.13
  (IND), 0.28-0.53 (COR): the brief's 0.179 point estimate is itself the noisy side of an
  n=6 look.
- **Which games flip (attribution, COR@0.922, nights >= 1.30):** ft09-family clones carry
  **+0.152** of the required ~+0.35 exceedance (44%), with mean ft09 lc rising 1.31 ->
  **2.00** - i.e. a 1.33 night is, first of all, an **"ft09-analogues complete level 2
  efficiently"** night. vc33 adds +0.055 (16%, lc 1.46->1.74), tn36 +0.021, ar25 +0.015,
  re86 +0.009; **r11l contributes almost nothing (+0.007)** - its local distribution is
  capped at lc 1 and cannot spike. R14-question ranking: **ft09 2-level >> vc33 > tn36 >>
  r11l**.

## 2. Headline 2 - regime verdict (R13's load-bearing question)

**No regime-transfer gap is required to explain 1.33; the "8h LB plays deeper than the
bench" hypothesis is NOT supported as a necessary claim - but the pure-independence model
IS refuted, so *something* night-correlated is real.** Quantified:

- The needed depth already exists locally: ft09 lc=2-3 and vc33 lc=2-3 appear in the 13
  bench seeds. Simulated 1.33 nights are built from outcome patterns the bench has produced.
- To make a single night >= 1.33 a >=5% event you need either night-sd >= ~0.19 or true
  mean >= ~1.05. The model delivers sd 0.16-0.20 through the measured common-night effect
  (vLLM/server health, sampling-temperature luck shared across all 110 games in one 8h
  run) - no extra depth needed. What the model **cannot** do is produce 1.33 from
  independent per-game noise at mean 0.922 (p ~ 0.001/night). The honest residual: either
  (a) the common-night effect transfers to the LB rail, or (b) the frozen fork's true
  official mean is ~0.95-1.00 (first five draws mildly unlucky - consistent with
  gap_forensics placing our early draws at the 7th-28th pct), or (c) some combination. All
  three are ordinary; none require "the LB regime plays deeper".
- **Gap quantification for the record:** IND@0.922 under-disperses reality (predicted sd
  0.13 vs pooled observed 0.146, and P(1.33 event) ~1%); COR@0.986 slightly over-disperses
  (0.172). The truth is inside the bracket. Nothing about the ledger, including 1.33, is
  >2 sigma outside the local generative family.

## 3. Headline 3 - honest E[max] table and what a filler window actually buys

The brief's arithmetic replicates (frozen6 point sigma-hat 0.179 -> E[max@107] = 1.444).
But the point estimate is the fragile edge of the honest range. Full table ("postpred" =
posterior predictive integrating sigma AND mu uncertainty, noninformative prior; walls at
107 remaining windows):

| ledger / model | sigma basis | E[max@30] | E[max@60] | E[max@107] | P(max@107>=1.44) | >=1.56 | >=1.61 |
|---|---|---|---|---|---|---|---|
| frozen5 (pre-1.33) | 0.074 point | 1.07 | 1.09 | 1.11 | 0.00 | 0.00 | 0.00 |
| frozen6 | 0.179 point | 1.36 | 1.41 | **1.44** | 0.48 | 0.08 | 0.03 |
| frozen6 | CI [0.112, 0.440] | 1.22-1.89 | 1.25-2.01 | 1.27-2.10 | 0.00-1.00 | 0.00-1.00 | 0.00-1.00 |
| frozen6 | postpred | 1.43 | 1.49 | 1.53 | 0.57 | 0.34 | 0.28 |
| **pooled10** (frozen6+war4) | 0.146 point | 1.28 | 1.32 | 1.36 | 0.09 | 0.00 | 0.00 |
| **pooled10** | CI [0.100, 0.266] | 1.19-1.53 | 1.22-1.60 | 1.24-1.66 | 0.00-0.99 | 0.00-0.81 | 0.00-0.64 |
| **pooled10** | postpred | 1.31 | 1.36 | **1.39** | **0.29** | 0.11 | 0.07 |
| generative model (IND-COR, both calibs) | sim | 1.19-1.34 | 1.23-1.39 | 1.26-1.43 | 0.01-0.39 | - | - |

**On poolability (R14 Q-B input, argued both ways):** For: war4 is the same duck harness +
warpack whose compound gate just FAILED both prongs (no measurable mechanism effect;
delta-lc-null-ish in all three instrumented evals), war mean 0.980 ~ frozen6 mean 0.990,
and Welch t on war4-vs-frozen6 is ~0.1 - statistically indistinguishable, so treating war4
as quasi-control draws of the same process is defensible and buys df 9 instead of df 5.
Against: warpack is formally UNTESTED-IN-REGIME (A9) and its banking/soft-end paths could
in principle truncate right tails (max war draw 1.08 vs frozen 1.33), so pooling may
slightly shrink sigma-hat. Both ledgers are therefore reported; they agree on the
decision-relevant band: **E[max@107] ~ 1.35-1.53, P(touch 1.44 wall) ~ 10-50%, central
~30%.**

**Decision-relevant conclusion (window pricing):**

- The 07-14 ruling "order stats never break the wall" is overturned in its absolute form,
  but the brief's "expected max AT the wall" is the optimistic edge: the honest statement
  is **E[max@107] ~ 1.39 (pooled postpred), P(reach 1.44) ~ 0.3** - filler is a genuine
  wall-path lottery, not a wall-path plan. P(reach the 1.56-1.61 top-5 band) ~ 0.07-0.11;
  P(reach 1.86) ~ 0.01. Volume alone still does not credibly reach the leader.
- **One marginal filler window buys P(new campaign best, >=1.33) ~ 0.025 (pooled postpred;
  0.007-0.025 under the generative model)** and ~0.01 of touching 1.44. Marginal E[max]
  value decays fast: E[max@30]->E[max@107] adds only ~+0.08 over 77 windows
  (~0.001/window at the tail).
- **Break-even for an experimental draw** (same sigma, true official mean lift delta,
  infra-death prob q): to match the filler's P(new-best), an experimental build needs
  delta >= **+0.06** (q=0), **+0.07** (q=0.15), **+0.08** (q=0.30) under pooled10;
  +0.08/+0.09/+0.11 under frozen6. So: **filler strictly dominates null-EV and
  hygiene-only experimental draws, but does NOT dominate any candidate credibly claiming
  >= +0.10-0.12 official** - i.e. the existing A10/track thresholds (+0.12) already price
  windows correctly. "Filler-as-strategy" is the right default for windows with no gated
  candidate; it is not a reason to stop shipping gated candidates.

## 4. Caveats

1. Official-games proxy: 110 unknown games modeled as clones of the public 25 (SCALE) or a
   trimmed subset (TRIM). Truth is a different game population; SCALE/TRIM bracket the
   plausible variance structures, but a genuinely different official tail game would evade
   both.
2. The common-night effect (sigma_u,local 0.168) is estimated from 13 seeds on the pod rail
   (F-ratio not individually significant); its transfer to the Kaggle 8h rail is assumed in
   COR, not proven. IND and COR are presented as a bracket for exactly this reason.
3. The 13 seeds pool 10 vanilla + 3 warpack-eval runs (delta-lc-null-ish); a null10-only
   and a +Tufa-20-pass fit were run as sensitivities and do not change any conclusion.
4. Stationarity assumed across nights (no game-version drift term); the 07-11
   monotone-drift watch was cleared on the record, but drift would inflate observed
   sigma-hat relative to any stationary model.
5. LOO fragility of frozen6 sigma-hat 0.179 is inherited by everything using it - which is
   why the pooled10 and generative-model rows are the recommended pricing basis.

## 5. Three headline answers (plain sentences)

1. **Does the local model reproduce the 1.33?** Yes, but only as a right-tail event and
   only if cross-game night correlation (or a true mean near 0.99) carries to the LB: with
   the measured common-night effect the model gives a 4-14% chance of seeing >=1.33 within
   the observed 6-10 draws; from independent per-game noise alone it is ~1%, effectively
   ruling that reading out. A simulated 1.33 night is primarily an "ft09-analogues clear
   level 2" night (44% of the exceedance), with vc33 second (16%); r11l contributes
   nothing.
2. **Regime verdict:** the 8h LB regime does NOT need to play deeper than the bench - every
   ingredient of a 1.33 night already exists in the 13 local seeds, and the pooled-10
   observed sigma-hat 0.146 falls inside the model's 0.11-0.20 bracket. What is refuted is
   the independence assumption (and the old 0.074 point sigma, which the model shows was a
   ~1-in-10 lucky-tight n=5 sample), not the bench's depth calibration.
3. **Honest E[max]:** the brief's 1.44@110 is the optimistic point-estimate edge. Central,
   honest numbers: E[max@107] ~ 1.39 (pooled-10 posterior predictive; 1.26-1.53 across
   ledgers/models), P(touch 1.44) ~ 30% (10-50%), P(reach 1.56+) ~ 10%. A marginal filler
   window buys ~2.5% chance of a new campaign best; an experimental window beats that only
   if its build credibly claims >= +0.06-0.11 true official lift after infra-death risk -
   so filler dominates null-EV draws, and the +0.12 gate thresholds already separate the
   two correctly.
