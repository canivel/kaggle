# Gate recalibration — the sealed K3 non-harm gate vs the campaign's own null

**Date:** 2026-08-09 · **Cost:** $0, CPU-only, zero Kaggle pushes · **Status:** independent
reproduction + extension of R24 methodology objection 1.

**Artifacts**
- script: `duck_eval/r24_prep/gate_recalibration.py` (deterministic, re-runnable, read-only)
- machine-readable: `runs/gate_recalibration_2026-08-09.json`
- gate under test: `PASS iff mean Δlc ≥ −0.128 AND worst-game Δlc ≥ −1.0`
  (implemented in `runs/kernel_pulls/a22_v2_1/_screen_m1m2.py` L116-119 and its v1/v2 twins)

---

## 0. Headline

1. **The reviewer's numbers reproduce exactly.** 12.2% / 50.0% / 51.1%. Not a single digit off.
2. **The provenance is worse than reported.** −0.128 is not a statistic at all: it is
   `(12 − 15.2)/25`, i.e. "the arm scored 12 levels when the vanilla 10-run mean was 15.2".
   Two unrelated runs on disk (`sentinel_eval_v1`, `war_v2_eval_s1`) both total 12 levels and
   both therefore report mean Δlc = −0.128 exactly.
3. **The pairing asymmetry is real but smaller than the reviewer's 1.78×.** The true
   single-vs-averaged inflation of the decision statistic is **1.28×** (per-game 1.34×, matching
   theory √(2/(1+1/9)) = 1.342). The reviewer's 1.78× compared a *null* sd against a
   *between-arm* sd. **Consequence: the fix is a new NUMBER first, a new procedure second** —
   −0.128 sits at ~10-13% type-I in *every* baseline-size regime, m = 1 … 9.
4. **NEW, and it changes the A22 answer.** Three runs of the *identical* comparator config
   (`war_eval_v1/v2/v3`, label `duck-harness-kaggle-warpack-v1`, ledger-OFF seeds 1/2/3, on
   *identical game instance ids*) score **22 / 15 / 13** levels. Their pairwise mean Δlc spans
   **−0.360 … +0.360**. A22 v2.1's headline harm of −0.360 is *exactly* the war_eval_v3-vs-
   war_eval_v1 difference — two runs of the same config with no compaction whatsoever.

---

## 1. Reproduction of the null calibration

### 1.1 What `runs/null10` actually contains — audited

| property | finding |
|---|---|
| runs | 10 (`vanilla_seed101` … `seed110`) |
| games | 25, identical game-id set across all ten runs |
| config | `phase1_env` byte-identical across all ten; `n_passes = 1` for all ten; elapsed 7974–8003 s |
| only difference | the RNG seed (101–110) |
| cross-check | per-game `levels_completed` in `vanilla_seedNNN.json` matches `seedNNN/benchmark.json` for all 250 cells — **0 mismatches** |
| per-run lc totals | 16, 11, 16, 15, 16, 15, 14, 18, 18, 13 (mean 15.20) |

**Verdict: genuine same-config null.** The reviewer's characterisation is accurate.

*Caveat I found and the reviewer did not report:* 15 of the 25 game **instance hashes** differ
between `null10` (pulled 07-12) and `war_eval_v1` (07-14) — e.g. `sk48-41055498` vs
`sk48-d8078629`. The screens key by 4-char prefix, so this is already campaign practice
(the −0.128 screen itself did it), but a calibration measured on one instance set and applied
to another should say so. It does not affect any conclusion below, because §4's decisive
comparison (`war_eval_v1/v2/v3`) is *within* one instance set.

### 1.2 Regime A — single run vs single run (the regime the gate is APPLIED in)

90 ordered pairs (arm run *i*, baseline run *j*), per-game Δlc, exactly as the screens compute it.

| statistic | mine | reviewer's | agree? |
|---|---|---|---|
| mean Δlc null: mean | 0.000 | 0.000 | ✅ |
| mean Δlc null: sd | **0.1223** | 0.1223 | ✅ |
| mean Δlc null: 5th pct | **−0.200** | −0.200 | ✅ |
| per-game Δlc sd | **0.6994** | 0.699 | ✅ |
| worst-game Δlc histogram | **0: 1, −1: 44, −2: 41, −3: 4** | 0:1, −1:44, −2:41, −3:4 | ✅ |
| **type-I, mean leg (≥ −0.128)** | **12.22%** (11/90) | 12.2% | ✅ |
| **type-I, worst leg (≥ −1.0)** | **50.00%** (45/90) | 50.0% | ✅ |
| **type-I, conjunction** | **51.11%** (46/90) | 51.1% | ✅ |

**Explicit statement of agreement: I reproduce every number in the reviewer's table exactly.**
The sealed K3 gate fails a true null 51.1% of the time. It is a coin flip.

Full null distribution of mean Δlc (n=90, symmetric by construction since ordered pairs are
sign-mirrored): min −0.28, p1 −0.28, **p5 −0.200**, **p10 −0.160**, p25 −0.08, median 0.00,
p75 +0.08, p90 +0.16, p95 +0.20, max +0.28.

### 1.3 Two cross-checks of reviewer reliability (both pass)

- `actions_per_level_completed` across null10 (objection 6): I get **195.5–322.9, mean 234.25,
  sd 40.88, CV 17.45%**. Reviewer: "195.5–322.9, mean 234.2, sd 40.9, CV 17.5%". ✅
- Fisher-z on ρ −0.131 → −0.403 (objection 3): I get z-diff −0.295, SE 0.3015, **p = 0.329**.
  Reviewer: "p ≈ 0.33". ✅

The reviewer is reliable on arithmetic. Where I diverge below is on *interpretation and scope*,
not on numbers.

### 1.4 Bonus: the sign-flip test is fine, the gate is not

The exact two-sided sign-flip p on the 90 null pairs never goes below 0.121 (5th pct 0.1875,
0/90 ≤ 0.05). The campaign's significance machinery is conservative and correctly calibrated.
It is only the two hard-coded threshold legs that are broken.

---

## 2. The pairing asymmetry, measured

The threshold was **estimated** against a 10-run-averaged baseline and is **applied** against a
single baseline run. I measured the null spread of the mean-Δlc statistic as a function of
baseline size *m* (arm = one held-out run; baseline = mean of *m* of the remaining 9; all
combinations).

| m (baseline runs) | draws | sd of mean Δlc | per-game Δ sd | 5th pct | 10th pct | P(worst ≤ −2) | **type-I at −0.128** |
|---|---|---|---|---|---|---|---|
| **1** (as applied) | 90 | **0.1223** | 0.6994 | −0.200 | −0.160 | 0.500 | **12.2%** |
| 2 | 360 | 0.1055 | 0.6056 | −0.180 | −0.142 | 0.219 | 12.5% |
| 3 | 840 | 0.0994 | 0.5709 | −0.187 | −0.147 | 0.127 | 13.0% |
| 4 | 1260 | 0.0962 | 0.5528 | −0.190 | −0.150 | 0.075 | 12.3% |
| 5 | 1260 | 0.0942 | 0.5416 | −0.184 | −0.144 | 0.042 | 11.0% |
| 6 | 840 | 0.0929 | 0.5340 | −0.187 | −0.141 | 0.021 | 10.7% |
| 7 | 360 | 0.0921 | 0.5286 | −0.189 | −0.134 | 0.008 | 10.3% |
| 8 | 90 | 0.0917 | 0.5245 | −0.185 | −0.126 | 0.000 | 10.0% |
| **9** (as estimated) | 10 | 0.0956 | 0.5222 | −0.147 | −0.107 | 0.000 | **10.0%** |

**Variance inflation from the substitution:** mean-statistic sd **1.28×** (0.1223 / 0.0956),
per-game sd **1.34×** (0.6994 / 0.5222), against the theoretical √(2/(1+1/9)) = **1.342**. The
empirical and theoretical values agree; the pairing effect is exactly what independence
arithmetic predicts and nothing more.

**Correction to the reviewer.** The claimed 1.78× (0.392 → 0.699) is not the pairing effect.
0.392 is the per-game sd of the *war-v1 arm vs null10 mean* — a between-arm quantity that
happens to be tight because that arm scored 0 on many games. The like-for-like null quantity is
0.5222, giving 1.34×. (For completeness I recomputed war_eval_v1 vs the null10 mean directly:
per-game sd **0.5264**, giving 1.33×.)

**What this means for the fix.** The −0.128 line sits at **10.0–13.0% type-I in every regime,
m = 1 through 9**. It was never a 5% line anywhere, in any pairing. So:

> The pairing asymmetry is a genuine ~28% variance understatement and should be fixed, but it is
> **not** the reason the gate is broken. The gate is broken because −0.128 was never a quantile
> of anything. **The primary fix is a new number.** The new procedure is a secondary, and
> separately justified, improvement.

### 2.1 The worst-game leg gets *worse*, not better, under an averaged baseline

Naively, averaging the baseline should shrink the worst-game tail — and for the *integer* event
"Δ ≤ −2" it does (0.500 → 0.000 at m = 9). But the leg as written is `Δ ≥ −1.0` against a
*fractional* baseline, and fractional Δ slips below −1.0 trivially. Measured regime-B worst-game
values: −1.222, −1.556, −0.556, −0.778, −1.222, −0.889, −1.556, −1.444, −1.000, −1.556 →
**type-I 60.0%**, and the conjunction fails **60.0%**. The leg is broken in both regimes for
different reasons.

---

## 3. Calibrated replacement gates

### 3.1 Mean Δlc — empirical thresholds

`stat ≥ t` with measured one-sided type-I ≤ α, on the empirical null:

| regime | α = 0.05 | achieved FPR | α = 0.10 | achieved FPR |
|---|---|---|---|---|
| **A: single vs single (n=90)** | **−0.200** | 2.2% | **−0.160** | 7.8% |
| **B: single vs mean-of-9 (n=10)** | −0.187 | 0.0% | −0.098 | 10.0% |
| sweep m=3 (n=840) | −0.187 | — | −0.147 | — |
| sweep m=4 (n=1260) | −0.190 | — | −0.150 | — |

(Regime B has only 10 non-independent draws; the m=3/m=4 sweep rows are the more usable
estimates of the multi-run-baseline null and land on ≈ −0.19 / ≈ −0.15.)

Discreteness note: Δlc is integer at m=1 so the achievable FPRs are lumpy — −0.200 gives 2.2%,
not 5%. −0.16 (7.8%) is the closest line to a true 5%; I recommend the conservative −0.200 and
require the seal to *quote* 2.2%.

### 3.2 Worst-game Δlc — is it salvageable at 25 games? **No.**

Regime-A null: `{0: 1, −1: 44, −2: 41, −3: 4}`. A −2 somewhere is the modal outcome. The only
threshold reaching α ≤ 0.05 is **−3** (P(worst ≤ −4) = 0/90), i.e. a leg that fires only on a
game losing four levels. At 25 games with per-game Δ sd 0.70, the minimum of 25 draws is
structurally ≈ −2 under any true null. **The worst-game leg is structurally uninformative and
must be dropped.**

**Replacement (count/quantile statistic), regime A, n=90:**

| # games with Δlc ≤ −2 | 0 | 1 | 2 | 3 | ≥4 |
|---|---|---|---|---|---|
| null count | 45 | 31 | 10 | 4 | 0 |
| P(≥ k) | 1.000 | 0.500 | 0.156 | **0.0444** | 0.000 |

- `count ≤ 2` → measured type-I **4.44%** ✅ α = 0.05
- `count ≤ 3` → measured type-I **0.0%** (the reviewer's proposal; also valid, more conservative;
  note their citation "null P(≥3) = 4%" is the FPR of `count ≤ 2`, an off-by-one in the write-up,
  not in the number).

I recommend **`count ≤ 2`** if a second leg is wanted at all, but honestly: at 4.4% FPR its
*power* is also near-nil (A22 v1 and v2 each have count = 1, i.e. indistinguishable from the
null median). It adds a 4-point inflation of family-wise error for almost no detection. **My
recommendation is to record it as non-gating/advisory** and let the mean leg be the single
decision statistic, which also discharges the reviewer's objection 6 about ten uncorrected
statistics.

### 3.3 Recommended gate — and why a fixed number is still not enough

A fixed line is only valid for a config whose run-to-run variance matches null10's. §4 shows the
warpack comparator's does *not* (4.8× the variance, F(2,9) = 4.83, p = 0.038). So the durable
recommendation is a **self-calibrating** line:

> **K3′ (recalibrated non-harm), single decision statistic:**
> pair the arm against the **per-game mean of m ≥ 3 runs of the same config as the comparator**;
> PASS iff
> `mean Δlc  ≥  − t(0.95, df = m−1) · s_base · √(1 + 1/m)`
> where `s_base` = sd over the m baseline runs of (run lc total ÷ 25).
> The seal must publish `m`, `s_base`, the resulting line, and its measured type-I rate.

Validation on the one config where a 10-run null exists: null10 gives `s_base` = 0.0860, m = 10 →
line **−0.165**, against an empirical 5th percentile of −0.147 (m=9) to −0.190 (m=4). The
parametric form lands inside the empirical band, so the formula is calibrated where it can be
checked.

**Fixed fallbacks if the multi-run baseline is unavailable:**
- m = 1: `mean Δlc ≥ −0.200` — measured type-I **2.2%** (α 0.10 ⇒ −0.160, 7.8%)
- m ≥ 3, vanilla-like variance: `mean Δlc ≥ −0.190` — empirical 5th pct
- **drop** `worst ≥ −1.0`; optional advisory `#(Δ ≤ −2) ≤ 2` at 4.4%

**Mandatory seal language:** every leg quotes its measured type-I rate and the corpus it was
measured on. A threshold without an operating characteristic is not a gate.

---

## 4. Provenance of −0.128 — worse than reported

`runs/sentinel_eval_v1/screen_report.md` L5: *"PRIMARY paired Δlc: mean −0.128 (sd 0.392, 4W/12L,
exact sign-flip p = 0.9495)"*. Confirmed on disk, and confirmed as a **non-significant** result
(p = 0.9495) promoted to a decision threshold. Three additional findings:

1. **−0.128 is an arithmetic identity, not a statistic.** In that screen the baseline is the
   null10 10-run mean, whose total is exactly 15.2. mean Δlc = (arm_lc_total − 15.2)/25. The arm
   scored 12 ⇒ (12 − 15.2)/25 = **−0.128 exactly**. The "threshold" encodes one fact: *an arm
   that scores 12 levels*. Independently, `runs/war_v2_eval_s1/screen_raw.json` — a completely
   different arm (warpack ledger+escalation) — also totals 12 and also reports mean −0.128, with
   a different sd (0.595) and p (0.860). The number is not identifying anything.
2. **The arm it came from is not the arm it is applied to.** The −0.128 screen's arm is
   `runs/kernel_pulls/sentinel_eval_v1` (label `duck-harness-kaggle-sentinel-v2`, **12** levels).
   Every A22 screen applies it against baseline `runs/kernel_pulls/war_eval_v1` (label
   `duck-harness-kaggle-warpack-v1`, **22** levels). Recomputing war_eval_v1 vs the null10 mean
   gives mean Δlc = **+0.272**, 12W/5L, sign-flip p = 0.0074 — the *opposite sign* and the
   strongest positive screen in the record. The two runs share only the word "war".
3. **The pairing regime changed too**, as the reviewer says (per-game sd 0.526 → 0.699,
   1.33×) — but per §2 that is the smaller half of the problem.

---

## 5. What this does to the A22 death verdict

### 5.1 The claim as it stands

| arm | screen | mean Δlc | worst | #(Δ≤−2) | sign-flip p | inherited verdict |
|---|---|---|---|---|---|---|
| A22 v1 | `runs/a22_compaction_v1/m1m2m3_screen.json` | −0.200 | −2 (sc25) | 1 | 0.234 | FAIL |
| A22 v2 | `runs/a22_v2_seed1/m1m2m3_screen.json` | −0.320 | −2 (sc25) | 1 | 0.0557 | FAIL |
| A22 v2.1 | `runs/kernel_pulls/a22_v2_1/m1m2m3_screen.json` | −0.360 | −2 (ar25) | 2 | 0.0781 | FAIL |

All three paired against the single run `war_eval_v1` (22 levels).

### 5.2 Against the vanilla null10 null (the reviewer's frame) — death survives

- P(null mean ≤ −0.320) = **0/90**; P(≤ −0.360) = **0/90** ⇒ v2 and v2.1 are outside the vanilla
  null entirely. **This half of the reviewer's claim reproduces.**
- v1's −0.200 sits **exactly at the 5th percentile**: P(null ≤ −0.200) = **7/90 = 7.8%**. Under
  the recalibrated α = 0.05 line (−0.200) v1 is a *marginal PASS*, not a FAIL.
- Worst-game leg: all three report worst = −2, which the null produces **50%** of the time.
  Zero evidence. Correctly to be struck from the record — reviewer agreed, and I agree.
- Count leg: v1 and v2 have count = 1 (null P(≥1) = 0.500 — the null median). v2.1 has count = 2
  (P(≥2) = 0.156). Also not evidence.

### 5.3 Against the comparator's OWN null — the death does **not** survive

This is the part neither the R24 proposal nor the methodology review ran, and it is decisive.

`war_eval_v1`, `war_eval_v2`, `war_eval_v3` all carry benchmark label
`duck-harness-kaggle-warpack-v1`, are documented as ledger-OFF seeds 1/2/3
(`learnings/daily_brief_2026-07-16.md`: *"war-eval seed 2 = NULL screen"*), and — verified — run
on **identical game instance ids** (25/25 match). They are a same-config null for the exact
baseline the A22 screens used. `w0_eval_s1` is the fourth member of the R16-sealed control band
(`learnings/panel/r16_circulation.md` L417-418).

lc totals: **war_eval_v1 = 22, war_eval_v2 = 15, war_eval_v3 = 13, w0_eval_s1 = 16.**

Pairwise mean Δlc among the three warpack seeds (no compaction anywhere in this table):

| pair | mean Δlc | worst | W/L | sign-flip p |
|---|---|---|---|---|
| v1 − v3 | **+0.360** | −1 | 9/1 | **0.0195** |
| v1 − v2 | +0.280 | −1 | 9/2 | 0.0654 |
| v2 − v3 | +0.080 | −1 | 6/6 | 0.818 |
| v3 − v2 | −0.080 | −2 | 6/6 | 0.818 |
| v2 − v1 | −0.280 | −1 | 2/9 | 0.0654 |
| **v3 − v1** | **−0.360** | −2 | 1/9 | **0.0195** |

- **A22 v2.1's headline "harm" of −0.360 is bit-for-bit the war_eval_v3-vs-war_eval_v1
  difference.** Same config, same games, no compaction, one day apart.
- That same pair produces sign-flip p = **0.0195** — *more* "significant" than v2.1's 0.0781 and
  v2's 0.0557.
- All three A22 arms lie **inside** the comparator's pairwise range [−0.360, +0.360]:
  P(warpack pair ≤ −0.200) = 2/6, ≤ −0.320 = 1/6, ≤ −0.360 = 1/6.
- The warpack config's run-to-run variance on lc totals is **4.83×** the vanilla null10's
  (F(2,9) = 4.83, one-sided **p = 0.0376**). The vanilla null10 spread — the reviewer's reference
  — **understates** the null the A22 arms should have been judged against. (Caveat: n = 3, so
  this variance estimate is itself noisy; but the vanilla null assigns probability 0/90 to a
  ±0.36 pair, and we observed one in only three warpack pairs.)

**Re-baselining A22 onto multi-run baselines that already exist on disk:**

| arm | vs war_v1 only (as screened) | vs mean(v1,v2,v3) | vs R16 4-run band |
|---|---|---|---|
| A22 v1 (lc 17) | −0.200 | **+0.013** | **+0.020** |
| A22 v2 (lc 14) | −0.320 | **−0.107** | **−0.100** |
| A22 v2.1 (lc 13) | −0.360 | **−0.147** | **−0.140** |
| baseline total | 22 | 16.67 | 16.5 |
| #(Δ ≤ −2) | 1 / 1 / 2 | **0 / 0 / 0** | **0 / 0 / 0** |

Against the recalibrated m ≥ 3 line (−0.190, or the self-calibrating warpack line of −0.412 at
α = 0.10 / −0.637 at α = 0.05), **all three A22 arms PASS.**

### 5.4 Verdict on the A22 death record — stated plainly

> **The A22 death record does NOT stand.**
>
> - The **worst-game leg falls outright** — 50% type-I; already conceded by the reviewer.
> - The **mean leg survives only against the vanilla null10 spread**, which is the wrong
>   reference for a warpack-baselined arm and is measurably too tight (F p = 0.038).
> - Against the comparator's own three same-config seeds, the observed "harm" is
>   **indistinguishable from picking a different baseline day.** The 22-level `war_eval_v1` is
>   the config's own documented high outlier — the campaign said so itself on 07-16 ("seed 1's
>   positive primary was plausibly a 1-seed draw from a noisy panel") and then used that very
>   run as the sole baseline for three consecutive A22 kills.
> - Re-baselined onto the 3-run or R16 4-run band that has been on disk since July, **A22 v1 is
>   +0.013 (no harm at all)** and v2/v2.1 are −0.107/−0.140, inside any calibrated line.
>
> **Correct status: A22 compaction is UNRESOLVED, not dead.** Three K3 strikes were three
> strikes of a broken instrument against an outlier baseline.

This is not a comfortable answer and I am not arguing A22 should be revived — the arms are
directionally negative in every framing (13/14/17 vs a 16.7 baseline mean), the mechanism story
is thin, and there are better uses of a push. But the *record* must say "not shown to help, not
shown to harm, screened against a miscalibrated gate and an outlier baseline", not "died three
times".

### 5.5 §2.2's monotonicity claim — does not survive (agreeing with the reviewer)

- Steps: −0.200 → −0.320 (−0.120) → −0.360 (−0.040).
- SE of a between-arm difference sharing one baseline (the shared baseline cancels, leaving both
  arms' own run noise): √2 × sd(single-run mean lc) = √2 × 0.0860 = **0.1216**. The steps are
  **z = −0.99 and −0.33**. (The reviewer quoted SE ≈ 0.198, derived differently; the conclusion
  is the same and mine is if anything the *tighter* SE — the steps still vanish.)
- ρ(`evicted_chars`, Δlc) −0.131 → −0.403: Fisher-z diff −0.295, SE 0.3015, **p = 0.329**. Not
  significant; "nearly 3× stronger" is not supportable.
- And now additionally: the arms' means themselves are one baseline-day's noise (§5.3), so the
  ordering of the three arms is not established either.

**Supported restatement:** *three single-seed screens of increasing eviction pressure all landed
below a single high-outlier baseline run; neither the level of harm nor its ordering is resolved
at this n.* §2.2 must not be adopted as a standing ordering constraint on this evidence.

---

## 6. Actions for the R24 seal

1. **Do not seal §6.2's K3 as written.** Measured type-I 51.1%. Replace with K3′ (§3.3).
2. **Name one decision statistic** — mean Δlc. Everything else non-inferential.
3. **Baseline must be m ≥ 3 same-config runs.** For any warpack-baselined arm, that band already
   exists (`war_eval_v1/v2/v3`, +`w0_eval_s1` under the R16 seal) at zero pushes. Answering the
   reviewer's question 2: **there is no reason not to, and the runs are on disk.**
4. **Publish the comparator config's own run-to-run sd in the seal.** A fixed threshold is only
   meaningful alongside it. warpack: totals 22/15/13, `s_base` = 0.189 — 2.2× vanilla's 0.086.
5. **Every leg quotes its measured type-I rate and null corpus.**
6. **Amend the A22 record** per §5.4 before it is entered as a formal finding, and restate §2.2
   per §5.5.
7. Consider running 2 more warpack/comparator seeds if any future arm's decision is close — the
   n=3 variance estimate is the weakest link in §5.3 and it is the only thing standing between
   "unresolved" and a real answer. (Not free; not urgent unless A22 is revisited.)

---

## Appendix — where each number came from

| number | source |
|---|---|
| null distributions, all FPRs, thresholds | `runs/null10/vanilla_seed{101..110}.json`, cross-checked vs `runs/null10/seedNNN/benchmark.json` |
| gate definition | `runs/kernel_pulls/a22_v2_1/_screen_m1m2.py` L116-119 |
| −0.128 provenance | `runs/sentinel_eval_v1/screen_report.md` L5, `runs/sentinel_eval_v1/screen_raw.json`, `runs/kernel_pulls/sentinel_eval_v1/benchmark.json` |
| −0.128 duplicate | `runs/war_v2_eval_s1/screen_raw.json` |
| war_eval_v1 = +0.272 | `runs/war_eval_v1/screen_raw.json`, recomputed from `runs/kernel_pulls/war_eval_v1/benchmark.json` |
| warpack 3-seed null | `runs/kernel_pulls/war_eval_v{1,2,3}/benchmark.json` |
| "seed 2 = NULL screen" | `learnings/daily_brief_2026-07-16.md` |
| R16 4-run control band | `learnings/panel/r16_circulation.md` L417-418 |
| A22 arms | `runs/a22_compaction_v1/`, `runs/a22_v2_seed1/`, `runs/kernel_pulls/a22_v2_1/` — `m1m2m3_screen.json` |
| ρ values | `M2.v2_attribution.pearson_evictedchars_vs_dlc` in the v2 / v2.1 screens |
| all of the above, machine-readable | `runs/gate_recalibration_2026-08-09.json` |
