# K3′ fallout — A22 disposition, type-II recalibration, protocol codification, warpack null

**Date:** 2026-08-10 · **Cost:** $0, CPU-only, **zero Kaggle pushes**, read-only w.r.t. every
existing artifact · **Status:** offline analysis of on-disk artifacts only.

**Artifacts**
- script: `duck_eval/r24_prep/k3prime_fallout.py` (deterministic, re-runnable)
- machine-readable: `runs/k3prime_fallout_2026-08-10.json`
- protocol written: `duck_eval/SCREEN_PROTOCOL.md` (new canonical file — see D3)

**Upstream**
`learnings/war_room/r24_minutes_2026-08-09.md` §3.2/§3.3/§3.3a/§5.1 ·
`learnings/sweeps/gate_recalibration_2026-08-09.md` · `runs/gate_recalibration_2026-08-09.json` ·
`learnings/panel/round25/_directives.md` (methodology N1) and `learnings/panel/round25/methodology.md`
(N1, N4, N5).

---

## 0. Headline

1. **D1 — the A22 re-screen the task asked for CAN be computed, and every arm PASSES; but the
   screen is not legal and not informative.** Not legal: `war_eval_v1/v2/v3` are **warpack** runs
   (`warpack: banking` in their logs) and the A22 arms are **no-warpack** runs (`NO warpack/
   ledger-graft/sentinel`, ~1,452–2,780 `COMPACTION ` lines). R17 **sealed** the warpack band
   ILLEGAL as a control on 2026-07-22, before all three A22 screens. Not informative: at warpack
   σ̂ the m=3 screen's 80 %-power detection floor is **0.566 lc/game = 14.1 levels over 25 games**
   against a ~17-level arm.
2. **D2 — the R25 N1 miscalibration REPRODUCES exactly.** Sealed K3′ at m=3 gives **−0.2916**
   versus its own m=1 fallback of **−0.200**: buying two extra baseline runs widens the PASS band
   by 46 %. Measured on `runs/null10`, sealed K3′'s type-I goes **2.0 % (m=1) → 7.1 % (m=3)** while
   power against a −0.20 harm goes **40 % → 45 %** with a line whose sd is **0.131** — the line is
   a random variable. Corrected gate **K3″** proposed, measured, monotone.
3. **D3 — no canonical screen protocol file existed.** Created `duck_eval/SCREEN_PROTOCOL.md`,
   with `m ≥ 3` as a hard precondition, a same-config legality precondition (the defect that
   actually killed A22), and K3″.
4. **D4 — the warpack-specific null is already better answered for free than by buying builds.**
   Pooling the four same-rail control families gives **σ̂ = 0.1417, df = 6** at zero cost — *more*
   precise than 3 extra warpack builds (df = 5) would buy. And the ceiling matters more than the
   df: with warpack σ = 0.189, **an infinite baseline plus one arm seed still gives only 18.9 %
   power** against a −0.20 harm. **RECOMMENDATION: buy no null builds.**

---

## 1. Inventory — what the build rail actually holds *(new; not in the 08-09 sweep)*

`runs/k3prime_fallout_2026-08-10.json → inventory`. 31 distinct 25-game `duck-harness` runs on
disk (33 files; 2 are byte-duplicates: `runs/kernel_pulls/compaction_v2/benchmark.json` duplicates
`runs/kernel_pulls/a22_v2_1/benchmark.json`, and `runs/tmp_pullback_gate_eval_s2v3/benchmark.json`
duplicates `runs/kernel_pulls/gate_eval_v2/benchmark.json` — identical `start_time` and identical
per-game lc). All 31 run the same 25 game **prefixes**; there are exactly two distinct **instance-id**
sets (null10 vs everything else), consistent with the 08-09 sweep's §1.1 caveat.

**Same-`label` families with m ≥ 2** (`inventory.families`):

| label | m | lc totals | mean lc/game | s_base | df |
|---|---|---|---|---|---|
| `duck-null-seed1xx` (null10) | 10 | 16,11,16,15,16,15,14,18,18,13 | 0.640…0.440 | **0.0860** | 9 |
| `duck-harness-kaggle-warpack-v1` | 3 | 22, 15, 13 | 0.88, 0.60, 0.52 | **0.1890** | 2 |
| `duck-harness-kaggle` | 3 | 18, 19, 21 | 0.72, 0.76, 0.84 | **0.0611** | 2 |
| `duck-harness-kaggle-continuation-v1` | 2 | 10, 16 | 0.40, 0.64 | **0.1697** | 1 |
| `duck-harness-kaggle-sentinel-v2` | 2 | 12, 16 | 0.48, 0.64 | **0.1131** | 1 |
| `duck-v2-seedNNN` | 3 | 14, 13, 16 | 0.56, 0.52, 0.64 | 0.0611 | 2 |

*(`duck-null-seedNNN` and `duck-v2-seedNNN` carry the seed inside the label; the script strips a
trailing `seed\d+` before grouping — `inventory()` in `k3prime_fallout.py`.)*

Two findings drop straight out of this table:

- **A third m=3 family exists and was never used:** `duck-harness-kaggle` = {`gate_eval_v1` (18,
  07-31), `gate_eval_v2` (19, 08-01 12:36), `tmp_pullback_duckgate_v1post` (21, 08-01 12:38)}.
  Its s_base is **0.0611 — *below* vanilla's 0.0860.** So "warpack-era build-rail runs are noisier"
  is not a property of the rail. Within the same rail and the same fortnight, a df=2 family estimate
  lands at 0.71× vanilla in one case and 2.20× in another. This is R25 N5's objection, measured.
- **`git_status.txt` is byte-identical (md5 `5bca49fe…`) across ALL of them** — including arms of
  demonstrably different configs (warpack vs no-warpack, compaction vs no-compaction). The R24
  minutes §3.3a cite "byte-identical `git_status.txt`" as evidence that v1/v2/v3 are the same
  config. **That evidence is vacuous**: the file records workstation repo commits, not the arm.
  The same-config claim for v1/v2/v3 still stands, but it rests on `label` + the log banners,
  which is what I used below.

---

## D1 — A22 re-screen under K3′, and the disposition ruling

### D1.1 The screens as they stand (recomputed from raw benchmarks, not read from the screen files)

| arm | lc | baseline as screened | mean Δlc | worst | sign-flip p | old-K3 verdict |
|---|---|---|---|---|---|---|
| A22 v1 (`runs/a22_compaction_v1`) | 17 | `war_eval_v1` (22) | **−0.200** | −2 | 0.2344 | **FAIL** |
| A22 v2 (`runs/a22_v2_seed1`) | 14 | `war_eval_v1` (22) | **−0.320** | −2 | 0.0557 | **FAIL** |
| A22 v2.1 (`runs/kernel_pulls/a22_v2_1`) | 13 | `war_eval_v1` (22) | **−0.360** | −2 | 0.0781 | **FAIL** |

Every `mean_dlc` recomputes to the stored value to <1e-9 (`D1_a22_rescreen.arms.*.old_K3.
recompute_matches_screen = true`). Old K3 = `mean Δlc ≥ −0.128 AND worst ≥ −1.0`, measured type-I
**51.1 %** (08-09 sweep §1.2).

There is only **one** distinct A22 arm per version — `runs/a22_v2_seed1/` contains the v2 seed-1
arm and no additional seed-1 arm; `compaction_v2` is a duplicate pull of v2.1. So the screenable
arm set is exactly {v1, v2, v2.1}.

### D1.2 K3′ exactly as sealed, against the warpack m=3 family

Sealed form (`gate_recalibration_2026-08-09.md` §3.3): `mean Δlc ≥ −t(0.95, df=m−1)·s_base·√(1+1/m)`,
`s_base` = sd over the m baseline runs of (run lc total ÷ 25).

Warpack family: m=3, lc totals 22/15/13 → run means 0.88/0.60/0.52 → **s_base = 0.18903**, df = 2,
t(0.95,2) = 2.9200, √(1+1/3) = 1.15470.

> **K3′ threshold (warpack, m=3, α=0.05) = −2.9200 × 0.18903 × 1.15470 = −0.6374.**
> (α=0.10: t(0.90,2)=1.8856 → **−0.4116**.)

| arm | mean Δlc vs mean(v1,v2,v3) | K3′ line | **K3′ verdict** | margin | old-K3 |
|---|---|---|---|---|---|
| A22 v1 | **+0.0133** | −0.6374 | **PASS** | +0.651 | FAIL |
| A22 v2 | **−0.1067** | −0.6374 | **PASS** | +0.531 | FAIL |
| A22 v2.1 | **−0.1467** | −0.6374 | **PASS** | +0.491 | FAIL |

Also computed for completeness: under the sealed **m=1 fixed fallback (−0.200)** against the
single run `war_eval_v1` as originally screened, v1 is a marginal **PASS** (−0.200 ≥ −0.200) and
v2/v2.1 **FAIL**. Under the corrected gate of D2 with the pooled σ̂ (D4), the m=3 line is
**−0.2863** and all three still PASS.

### D1.3 …but that screen is not legal, and the campaign sealed that fact on 2026-07-22

Verified from the run logs, not from labels (`D1_a22_rescreen.config_audit`):

| run | label | `NO warpack` banner | `warpack: banking` | `COMPACTION ` lines |
|---|---|---|---|---|
| A22 v1 | `…-compaction-v1-continuation-v1` | **yes** | no | 1,452 |
| A22 v2 | `…-compaction-v2-continuation-v1` | **yes** | no | 2,617 |
| A22 v2.1 | `…-compaction-v2.1-continuation-v1` | **yes** | no | 2,780 |
| war_eval_v1/v2/v3 | `…-warpack-v1` | no | **yes** | 0 |
| w0_eval_s1, w0_cont_eval | `…-continuation-v1` | **yes** | no | 0 |

The A22 prereg says so itself: *"Composition: duck baseline + (f) continuation default … +
COMPACTION=1. **NO warpack**, NO ledger graft, NO sentinel"*
(`learnings/war_room/a22_compaction_prereg_2026-08-01.md:77-78`).

And the campaign had already ruled on this comparison:

- `runs/sealed/r17_thresholds.json` (`sealed_at: 2026-07-22`) → `thresholds.control_band`:
  `"verdict": "n=4 pooled band ILLEGAL (config-diff exceeds {(f)}: warpack)"`,
  `"fallback": "2 fresh W0 seeds -> band {w0_s1,w0_s2,w0_s3}, n=3"`, `"gpu_h": 4.4`.
- Same file → `thresholds.legal_control_reanchor`: *"…anchor to the LEGAL control w0_s1 (not the
  illegal warpack baseline) … warpack comparison is DIAGNOSTIC ONLY"*, with
  `legal_control_w0_s1: 1.731` and `illegal_warpack_3seed_baseline_diagnostic_only: 1.454`.
- Restated in prose at `learnings/war_room/sentinel_w2_preregistration.md:58-61`.

All three A22 screens (08-02, 08-05, 08-06) postdate that seal and all three used the illegal
control anyway. So the A22 death record has **four** defects, not two: a mis-calibrated gate
(51.1 % type-I), a single-run baseline, a **high-outlier** single-run baseline, and a baseline of
the **wrong config**, ruled illegal 11 days before the first screen.

**A live seal conflict, recorded:** R16 §11 (`learnings/panel/r16_circulation.md:417-418`) sealed
the n=4 warpack-inclusive band as *legal*; R17 sealed it *illegal* on 07-22. The 08-09
recalibration cited the R16 band (§5.3, §6.3) without noting the R17 override. **R17 is later and
governs.** This should be corrected in the R24 record.

### D1.4 The legal control family, and the precondition failure

The legal same-config comparator for an A22 arm is the `duck-harness-kaggle-continuation-v1`
family: `w0_eval_s1` (16 lc, 07-18) and `w0_cont_eval` (10 lc, 07-24). **m = 2.**

| arm | mean Δlc vs mean(w0_eval_s1, w0_cont_eval) = 13.0 lc | screenable under K3′? |
|---|---|---|
| A22 v1 (17) | **+0.160** | **NO — m=2 < 3** |
| A22 v2 (14) | **+0.040** | **NO — m=2 < 3** |
| A22 v2.1 (13) | **0.000** | **NO — m=2 < 3** |

Against the legal control family every A22 arm is at or **above** baseline. Advisory only: the
m ≥ 3 precondition sealed in K3′ is not met, the family's two runs are 10 and 16 levels
(s_base 0.1697, df=1), and a df=1 variance estimate is not a variance estimate.

**So the task's premise — "an m=3 same-config baseline family already exists, therefore A22 can be
re-screened at $0" — is true for warpack-family arms and false for A22.** The m=3 family that
exists belongs to a different config. The family A22 needs has m=2. A22 is, under the sealed K3′
precondition, **NOT SCREENABLE at $0 today.**

### D1.5 What a PASS would and would not establish — measured

At the m=3 warpack line, with σ̂ = 0.18903 and SE = σ̂·√(1+1/3) = 0.2183:

| screen | line | 80 %-power detection floor | in levels / 25 games | power vs a true −0.20 harm |
|---|---|---|---|---|
| K3″, warpack m=3, σ̂ df=2 | −0.3818 | **0.566 lc/game** | **14.1 levels** | **20.2 %** |
| K3″, warpack m=3, σ̂ pooled df=6 | −0.2863 | 0.424 lc/game | 10.6 levels | 29.9 % |
| K3″, continuation m=2, σ̂ pooled df=6 | −0.2976 | 0.444 lc/game | 11.1 levels | 28.7 % |

(sealed K3′ at the warpack m=3 line −0.6374 is worse still: power vs −0.20 is **2.3 %**.)

A22 v2.1 scored 13 levels against a 16.67-level 3-run baseline mean. To FAIL the screen it would
have had to score roughly **3 levels**. R25 N4 is right in substance and now has a number:
**PASS here means "the instrument cannot see", not "no harm".**

---

## RULING (D1)

> **A22 compaction is re-screened as far as the data legally permit, and its status is:
> FORMALLY OPEN AND UNWORKED — "screened by a broken instrument against an illegal, high-outlier,
> single-run baseline; not shown to help, not shown to harm; not re-screenable at $0."**
>
> **Not** "re-screened and cleared". The PASS in D1.2 is against the wrong config and has 20 %
> power; the +0.160/+0.040/0.000 in D1.4 is against a legal family that fails K3′'s own m ≥ 3
> precondition. Recording either as a clearance would repeat, with the sign flipped, exactly the
> error R24 spent the round exposing.

**What the re-screen DOES establish**
1. The three K3 FAILs are **withdrawn**, not merely downgraded. Every arm passes every calibrated
   line available — warpack m=3 (−0.6374), corrected warpack m=3 (−0.3818), corrected pooled m=3
   (−0.2863), and the m=1 fallback for v1 (−0.200). There is no calibrated line on disk that any
   A22 arm fails.
2. §2.2's monotonicity claim stays dead (08-09 sweep §5.5: step z = −0.99 / −0.33; ρ shift
   p = 0.329) and nothing here revives it.
3. A **new** defect is added to the record: the baseline was the wrong config, and the campaign
   had sealed that as illegal on 07-22 (D1.3).

**What it CANNOT establish**
1. Not that compaction is harmless. Power against a −0.20 harm is 20–30 %; the 80 % floor is
   10.6–14.1 levels. "PASS" is the outcome for an arm that is genuinely neutral **and** for an arm
   that is genuinely 0.30 lc/game harmful.
2. Not a direction. All three arms are directionally below the illegal baseline and at-or-above
   the legal one; the two families differ by 3.67 levels of baseline mean, which is inside a
   single family's own seed spread.
3. Not a re-funding. **Lane (a) state-externalisation was ratified on independent grounds** (R24
   §2.1: three-team convergence, 2608.01326, Tycho, the metric argument) and **keeps the budget.**
   A22 passing a screen changes nothing about lane (a)'s claim on the rail.

**The decision-relevant question, stated plainly: does A22 have positive expected value relative
to lane (a) work, given it costs kernel pushes and the m ≥ 3 requirement?**

Priced, at 2.211 h/build measured (n=31, range 2.202–2.219 h) and 12–13 builds/week (minutes §5.4):

| item | builds | GPU-h | share of a 13-build week |
|---|---|---|---|
| 1 fresh W0 seed → legal control family reaches m=3 | 1 | 2.2 | 8 % |
| 1 A22 re-run on the legal harness (any version) | 1 | 2.2 | 8 % |
| **minimum to produce one legal, m≥3 A22 screen** | **2** | **4.4** | **15 %** |
| …at 29.9 % power against a −0.20 harm | | | |
| to reach 80 % power vs −0.20 at pooled σ̂ (k=m=8) | **16** | 35.4 | **>1 full week** |

**Answer: NO — negative EV. A22 stays open and unworked.** Reasoning:
- The cheap version (2 builds) buys a screen with **~30 % power**, i.e. it re-runs the R10–R23
  failure mode with a nicer threshold: a likely-PASS that says nothing, on a lane with no
  mechanism claim left standing.
- The powered version costs **16 builds ≈ 1.3 weeks of the entire free rail**, for a *non-harm*
  result — an outcome that at best licenses further work rather than producing any.
- Lane (a) currently owes free, zero-push instrumentation (minutes §5.3: latency + matched-action
  prefix, `namespace_reuse_rate` definition, `SAFE_MODULES` gap) and R25's N3 ρ(public,private)
  estimate is the cheapest, highest-VOI item in the campaign (`round25/_directives.md`). Those
  consume no builds. Spending builds on A22 while free lane-(a) work is unfinished is strictly
  dominated.

**Named revival conditions (all must hold before any A22 build is spent):**
- **R1.** Lane (a)'s free instrumentation (§5.3) and R25 N3 are discharged, and lane (a) is either
  advancing on the build rail or blocked on something A22 does not compete with.
- **R2.** A **mechanism** claim exists that survives §2.2's death — a pre-registered, measurable
  statement of what eviction is supposed to *buy*, not merely "does not harm". A non-harm screen
  is not a reason to run an arm.
- **R3.** The legal control family `duck-harness-kaggle-continuation-v1` has reached **m ≥ 3**
  (currently 2; the R17 seal already budgeted this at 4.4 GPU-h for 2 seeds and it was never
  finished — one seed closes it).
- **R4.** The prereg publishes σ̂, its df, the K3″ line, **and the 80 %-power detection floor in
  levels** before the arm runs, per `duck_eval/SCREEN_PROTOCOL.md` §4.
- **R5.** If the honest power number at the affordable m is < 50 %, the prereg says in its own text
  that the arm is an exploratory probe, not a screen, and no PASS may be reported as non-harm.

### D1.6 Two new external results, and whether they move the ruling

Filed today at `learnings/sweeps/research_2026-08-10.md`. Both bear on the EV half only; neither
touches D1's arithmetic. Both carry the standing [SR]/provenance de-rating.

- **2608.07077** (`research_2026-08-10.md:49-80`, verdict row L270) — probes on **Qwen3.6-27B, our
  exact backbone**, find the world model is encoded near-perfectly at prompt end, that failure
  beyond ~3 rings is located in the **decaying representation**, and that re-injecting the
  prompt-time representation partially recovers performance. Read: **the deficit is maintenance,
  not construction.** Discounts carried, both load-bearing: Tower of Hanoi is a fully-observed
  deterministic puzzle and is not ARC-AGI-3; and the causal intervention is **activation-space**,
  which is unreachable on our rail — text externalisation is the *assumed* analogue, not the tested
  one.
- **2608.07429 (TEPA)** (`research_2026-08-10.md:135-162`, verdict row L272) — under reversal,
  append-only memory scores **0.210 vs 0.309 for no memory** (TEPA 0.950); the gain is *entirely* a
  drift/reversal effect. Read: accumulated-but-stale state is actively harmful absent a validity
  lifecycle.

**Does either change the ruling? Partly — it changes the *reason*, not the verdict.**

- Both items make **maintenance** the shared problem statement of A22 and lane (a), which weakens
  the old framing that they are rivals. But 2608.07077's mechanism datum is evidence for
  **externalisation**, and its intervention is re-*injection* — adding state back — not eviction.
  It does not argue for compaction. If anything it argues that whatever compaction throws away is
  the thing the 27B is already losing on its own.
- TEPA cuts *against pure accumulation* and therefore is the one item that is genuinely
  pro-A22-adjacent: it says an unmanaged growing store is worse than no store. But TEPA's answer is
  **revocation with an audit trail keyed on validity** — a *semantic* lifecycle — not
  **token-pressure eviction**, which is what A22 v1/v2/v2.1 implement. The sweep itself routes TEPA
  to **P3**, not to A22 (`research_2026-08-10.md:272`). Adopting TEPA's lesson does not require
  reviving A22; it requires P3 to have a validity lifecycle.
- Net effect on EV: it **raises the value of a future A22-shaped arm** (one that evicts on
  *validity*, per TEPA, rather than on token pressure) and **lowers the value of re-running
  A22 v1/v2/v2.1 as built**. That is a change in what a revived A22 should *be*, and it is
  therefore folded into revival condition **R2** rather than into the disposition.
- **The ruling stands unchanged: A22 formally open, unworked, no builds.** Neither paper is our own
  measured data, and the ruling's binding argument is a power number computed from our disk
  (20–30 %), which no external result can move.

---

## D2 — K3′ type-II recalibration (R25 methodology N1, FATAL)

### D2.1 The reviewer's arithmetic reproduces exactly — CONFIRMED

`runs/null10` (10 same-config vanilla runs × 25 games). Two routes to `s_base`:

- pair sd of mean Δlc = **0.12234** ⇒ implied run-level sd = 0.12234/√2 = **0.086508** (the
  reviewer's 0.0865);
- direct sd of the 10 run means (16,11,16,15,16,15,14,18,18,13)/25 = **0.085997** (the 08-09
  sweep's 0.0860).

Substituting the reviewer's 0.086508 into the sealed formula:

| m | t(0.95, m−1) | √(1+1/m) | multiplier | **sealed K3′ line** | reviewer's | agree |
|---|---|---|---|---|---|---|
| 1 | — (df=0) | 1.4142 | undefined | **fixed fallback −0.200** | −0.200 | ✅ |
| 2 | 6.3138 | 1.2247 | 7.733 | **−0.6687** | — | — |
| 3 | 2.9200 | 1.1547 | 3.372 | **−0.2916** | −0.292 | ✅ |
| 5 | 2.1318 | 1.0954 | 2.335 | **−0.2020** | −0.202 | ✅ |
| 9 | 1.8595 | 1.0541 | 1.960 | −0.1695 | — | — |
| 10 | 1.8331 | 1.0488 | 1.923 | **−0.1663** | −0.166 | ✅ |

> **CONFIRMED, both halves of N1:**
> (i) the sealed m≥3 fallback **−0.190 is NOT reproducible from the sealed formula** (the formula
> gives −0.2916 at m=3; −0.190 is the empirical 5th percentile of the vanilla null sweep, a
> different construction);
> (ii) K3′ at m=3 (**−0.2916**) is **46 % looser** than its own m=1 fallback (**−0.200**). Buying
> two extra baseline runs widens the PASS band.

### D2.2 Measured operating characteristics — not theory

Exact identity used throughout (verified in the script): `mean Δlc = M_arm − mean(M_baseline)`
where `M_r = run lc total ÷ 25`; the per-game structure cancels. So the calibration of the mean-Δlc
statistic is fully determined by the 10 null10 run means. Two instruments:

- **exhaustive** — all (held-out arm run) × (size-m subset of the other 9) draws (90/840/1260/10 at
  m=1/3/5/9). Finite-pool: its m≥2 tails are dominated by which single run is held out.
- **i.i.d. bootstrap** — 400,000 draws per m, arm and baseline runs drawn with replacement from the
  10 observed run means, seed 20260810. Clean m-dependence; this is the primary instrument.

Harm model: location shift — every game's arm lc reduced by δ, so the statistic shifts by −δ.

**Sealed K3′ (bootstrap instrument):**

| m | line mean | line sd | P(s_base = 0) | **type-I** | power vs −0.20 | power vs −0.30 |
|---|---|---|---|---|---|---|
| 1 | −0.2000 | 0 | — | **2.0 %** | 39.9 % | 80.9 % |
| 3 | **−0.2421** | **0.1306** | 4.6 % | **7.1 %** | **44.8 %** | **64.3 %** |
| 5 | −0.1790 | 0.0650 | 0.3 % | 7.4 % | 56.4 % | 84.8 % |
| 9 | −0.1553 | 0.0383 | 0 | 7.3 % | 65.8 % | 94.2 % |

(exhaustive instrument agrees on the pathology: type-I 2.2 / 7.0 / 9.9 / 10.0 % and mean line
−0.200 / −0.264 / −0.194 / −0.168 at m = 1/3/5/9; at m=2 the sealed line averages **−0.544**.)

**Two measured defects, both new relative to the 08-09 seal:**
1. **Non-monotone stringency.** Mean line magnitude runs 0.200 → **0.242** (m=3) → 0.179 → 0.155.
   It gets *looser* going from m=1 to m=3 and only recovers at m≈5. Sealed K3′ therefore penalises
   the very behaviour the protocol is trying to buy.
2. **The line is a random variable, not a threshold.** At m=3 its sd is 0.1306 — 54 % of its own
   mean — and in 4.6 % of draws the three baseline runs happen to tie and the line collapses to
   0.000, i.e. the arm fails on any negative Δ whatsoever. This is why type-I *rises* to 7.1 %
   while power *falls*: the same randomness that produces absurdly wide lines also produces
   absurdly tight ones.

### D2.3 Diagnosis

The sealed K3′ **splices two constructions on two different scales**.

- The **m=1 fallback** −0.200 is an *empirical* 5th percentile. In units of s_base it is
  **2.3257** — and normal theory for a difference of two runs gives 1.645·√2 = **2.3264**. The
  empirical and theoretical m=1 multipliers agree to 3 decimal places. The m=1 leg is fine.
- The **m≥2 leg** is a *parametric t prediction interval* with multiplier `t(0.95,m−1)·√(1+1/m)`,
  which in the same units is **7.733 (m=2), 3.372 (m=3), 2.335 (m=5), 1.960 (m=9), 1.923 (m=10)**.
  That family *is* internally monotone — the break is the splice: at m=3 the parametric multiplier
  (3.372) is **1.45×** the empirically-calibrated m=1 multiplier (2.326).
- **The driver is the t-multiplier at df=2, not the √(1+1/m) inflation.** `t(0.95,2) = 2.920` is
  1.78× the large-sample 1.645, while √(1+1/m) correctly *shrinks* 1.414 → 1.155 → 1.054. The
  inflation term is doing its job; the df=2 variance estimate is not.
- Secondary driver: with df=2 the estimate ŝ is itself so noisy (relative SE = 1/√(2·2) = **50 %**)
  that plugging it in makes the *line* random, per D2.2 defect 2.

### D2.4 The corrected gate — K3″

> **K3″ (corrected non-harm, single decision statistic).**
> Pair the arm against the **per-game mean of m ≥ 3 same-config baseline runs**. PASS iff
>
> ```
> mean Δlc  ≥  − C(m) · σ̂
> ```
>
> where **σ̂** is the run-to-run sd of per-game mean lc for the comparator's **harness family**,
> estimated by pooling within-family sums of squares across same-rail control families
> (**require df ≥ 4**; publish σ̂, its df and the families used), and **C(m)** is the null10-measured,
> monotonised 5th-percentile multiplier:
>
> | m | 1 | 2 | 3 | 4 | 5 | ≥6 |
> |---|---|---|---|---|---|---|
> | **C(m)** | 2.33 | 2.10 | **2.02** | 1.98 | 1.96 | 1.94 |
>
> (measured envelope 2.3257 / 2.0931 / 2.0156 / 1.9768 / 1.9535 / 1.9380; rounded up to 2 dp.)

**Monotone by construction** — C(m) is non-increasing and σ̂ does not depend on m, so the PASS band
can never widen when a baseline run is added. Verified in the JSON
(`corrected_gate_K3pp.monotone_non_increasing_in_m = true`).

**Measured on null10 (independent bootstrap evaluation stream, seed 99260810, 400k draws/m):**

| m | line at vanilla σ̂ | **type-I** | power −0.10 | **power −0.20** | power −0.30 | 80 %-power floor |
|---|---|---|---|---|---|---|
| 1 | −0.2004 | **2.0 %** | 19.0 % | 40.0 % | 80.9 % | 0.297 |
| 3 | −0.1737 | **4.4 %** | 20.8 % | **56.7 %** | 91.8 % | 0.253 |
| 5 | −0.1686 | **4.8 %** | 21.0 % | 60.8 % | 94.2 % | 0.244 |
| 9 | −0.1668 | **5.2 %** | 19.9 % | 64.5 % | 95.2 % | 0.239 |

Type-I is ≤ 5.2 % at every m, stringency is monotone, and power is monotone *increasing* in m —
all three properties the sealed version lacks. Against the sealed gate at m=3 the improvement is
**type-I 7.1 % → 4.4 %** and **power vs −0.20 44.8 % → 56.7 %** simultaneously.

**Lines K3″ produces on the three σ̂'s that exist on disk:**

| σ̂ | source | m=1 | m=3 | m=5 | m=9 |
|---|---|---|---|---|---|
| 0.0860 (df 9) | null10 vanilla | −0.2004 | −0.1737 | −0.1686 | −0.1668 |
| 0.1417 (df 6) | pooled build-rail (D4) | −0.3303 | −0.2863 | −0.2778 | −0.2750 |
| 0.1890 (df 2) | warpack only | −0.4404 | −0.3818 | −0.3704 | −0.3667 |

**Honest limits of K3″, to be quoted in every seal:**
- C(m) is calibrated on a **10-run vanilla** null. It is a *shape* (a multiplier of σ̂), which is
  why it transports across families, but the shape's tail behaviour was measured on one family.
- The whole gate inherits null10's instance set; 15 of 25 instance hashes differ from the kernel
  pulls (08-09 sweep §1.1).
- **Power is bounded by σ̂, not by the gate.** At warpack σ̂ = 0.189, K3″'s 80 % floor at m=3 is
  **0.566 lc/game = 14.1 levels**. A gate cannot fix that; only more seeds or a lower-variance
  endpoint can. See D4.

---

## D3 — the m ≥ 3 requirement, codified

**No canonical screen-protocol file existed.** Searched `duck_eval/`, `scripts/`, `learnings/`:
the gate lived in three places, none of them a protocol — inline in
`runs/kernel_pulls/a22_v2_1/_screen_m1m2.py:116-119` (and its v1/v2 twins), restated per-arm in
each prereg (`a22_compaction_v2_1_prereg_2026-08-06.md` §3 "M1 (primary): inherited VERBATIM"), and
sealed numerically in `runs/sealed/r17_thresholds.json`. `duck_eval/README.md` is a rig description,
not a protocol.

**Created `duck_eval/SCREEN_PROTOCOL.md` and declared it canonical.** It carries:
- **P1 (hard precondition): same-config legality** — the baseline family must share the arm's
  config in everything except the one pre-registered change, verified from run **banners**, not
  labels or `git_status.txt`. This is the precondition A22 violated.
- **P2 (hard precondition): m ≥ 3** same-config baseline runs, per R24 §5.1.
- **P3 (hard precondition): σ̂ with df ≥ 4**, pooled across same-rail control families.
- **K3″** exactly as D2.4, with its C(m) table and measured type-I.
- **P5: the 80 %-power detection floor must be published, in levels, before the arm runs**, and if
  it exceeds the arm's plausible effect the prereg must call the run exploratory, not a screen.
- The worst-game leg formally struck; `#(Δ ≤ −2) ≤ 2` advisory only.

---

## D4 — the warpack-specific null: scoped, and priced

### D4.1 What is free — and it is thinner than 4.83× suggests

**Warpack-only (m=3, df=2):** lc 22/15/13 → σ̂ = **0.18903**. Three same-config runs give three
independent pairs; the 08-09 sweep's six "ordered pairs" are three pairs counted twice.

- relative SE of σ̂ = 1/√(2·df) = **50 %**
- 90 % CI on σ (χ²₂): **[0.1092, 0.8347]** — a **7.6×** range.
- vs vanilla: F(2,9) = **4.832**, one-sided p = **0.0376**, but
  **90 % CI [1.135, 93.7]**, **95 % CI [0.845, 190.3]**.

> **R25 N5 is confirmed with numbers: the 95 % CI on "4.83× vanilla" includes 1.** The point
> estimate is consistent with no inflation at all and with two orders of magnitude of it. It may
> be carried as a *precaution*, never as a measured fact.

The inventory (§1) makes the same point empirically: the same build rail, same fortnight, same 25
instances, `duck-harness-kaggle` m=3 → σ̂ = **0.0611**, i.e. **0.71×** vanilla, while warpack m=3 →
**2.20×** vanilla. Two df=2 estimates, 3.1× apart, both "measured".

### D4.2 The free thing nobody did: pool the families

All four m≥2 build-rail control families run the **same 25 game instance ids** on the **same free
rail**. Pooling their within-family sums of squares:

| family | m | SS | df |
|---|---|---|---|
| `…-warpack-v1` | 3 | 0.071467 | 2 |
| `duck-harness-kaggle` | 3 | 0.007467 | 2 |
| `…-continuation-v1` | 2 | 0.028800 | 1 |
| `…-sentinel-v2` | 2 | 0.012800 | 1 |
| **pooled** | | **0.120533** | **6** |

> **σ̂_pooled = 0.14174, df = 6, 90 % CI [0.0978, 0.2715]** (a **2.8×** range, vs warpack-only's
> 7.6×). Bartlett homogeneity across the four families: **χ² = 1.753, p = 0.625** — no evidence
> against pooling. vs vanilla: F(6,9) = 2.716, one-sided p = **0.086**.

**This free estimate is more precise than 3 extra warpack builds would buy.** Warpack-only at m=6
would have df=5 and a CI-width ratio of 3.11; the pooled estimate available today has df=6 and a
ratio of 2.77. To beat it with warpack seeds alone you need df ≥ 7, i.e. **m ≥ 8 — five more
builds, 11.1 GPU-h, ~38 % of a week's quota.**

### D4.3 Precision is the cheap problem; power is the expensive one

Even with σ known exactly, the achievable power is capped by σ itself. 80 % one-sided power at
K3″ needs `(C + 0.8416)·σ·√(1/k + 1/m) ≤ |harm|` (k arm seeds, m baseline seeds, C(m≥6)=1.94):

| family σ | detect −0.10 | detect −0.20 | detect −0.30 | ceiling: ∞ baseline, 1 arm seed, vs −0.20 |
|---|---|---|---|---|
| warpack 0.189 | k=m=56 (**112 builds**) | k=m=14 (**28 builds**) | k=m=7 (14 builds) | **18.9 %** |
| pooled 0.1417 | k=m=32 (64) | k=m=8 (**16 builds**) | k=m=4 (8) | 29.8 % |
| vanilla 0.0860 | k=m=12 (24) | k=m=3 (6) | k=m=2 (4) | 65.0 % |

Measured build cost: **2.2110 h/build** (n=31 completed 25-game runs on disk, range 2.2021–2.2190 h
— `D4_warpack_null.price.measured_build_hours`), i.e. **13.6 builds/week** at 30 GPU-h, consistent
with minutes §5.4's 12–13.

> **An infinite warpack null buys 18.9 % power against a −0.20 harm.** The binding constraint is
> not the null's df. It is that a 25-game lc endpoint on a warpack-noise config cannot resolve a
> 5-level effect at any affordable n.

### RECOMMENDATION (D4)

> **Do NOT spend build-rail runs on a warpack-specific null. Adopt the pooled build-rail estimate
> σ̂ = 0.1417 (df = 6) as the standing family σ̂ for all warpack-era screens — free, today, and more
> precise than the 3 builds the alternative would cost.**

- **Cost avoided:** 5 builds / 11.1 GPU-h / ~38 % of a week to merely match what pooling already
  gives; 16 builds / 35.4 GPU-h / **>1 full week** to make a warpack-family screen actually
  powered against a −0.20 harm.
- **Opportunity cost named:** at 12–13 builds/week, 16 builds is **1.3 weeks of the entire free
  rail** — the same rail lane (a) needs for its P1/P3 screens, and the same rail R25 has already
  told us not to touch until N3 (ρ(public, private)) is answered
  (`learnings/panel/round25/_directives.md:15`, *"Do NOT build until N3 is answered"*).
- **Standing precaution, not a measured fact:** σ̂_pooled 0.1417 is **1.65×** vanilla's 0.0860 with
  F(6,9) p = 0.086 — suggestive, not established. Use it because it is conservative for screening,
  and say so; do not repeat "warpack variance is 4.83× vanilla" as a fact (95 % CI [0.85, 190]).
- **The one exception worth 1 build, and it is not a null build:** the R17-sealed fallback
  (`runs/sealed/r17_thresholds.json → control_band.fallback`, budgeted at 4.4 GPU-h for 2 seeds)
  is half-finished — the `continuation-v1` legal control family sits at m=2. **One** W0 seed
  closes it and makes the entire `{(f)}`-envelope arm family (A22, and any future compaction/
  memory arm on that harness) legally screenable for the first time. That build is justified *when
  and if* an arm on that harness is actually queued, per revival condition R3 — not before.

---

## Appendix — where each number came from

| number | source |
|---|---|
| run inventory, families, duplicates, build hours | `runs/**/benchmark.json` (31 distinct 25-game duck-harness runs) → `runs/k3prime_fallout_2026-08-10.json → inventory`, `D4_warpack_null.price` |
| A22 arm lc / mean Δlc / old-K3 verdicts | `runs/a22_compaction_v1/`, `runs/a22_v2_seed1/`, `runs/kernel_pulls/a22_v2_1/` → `m1m2m3_screen.json` + recompute from `benchmark.json` |
| warpack / continuation banners, COMPACTION line counts | `arc3-duck-*.log` in each pull → `D1_a22_rescreen.config_audit` |
| "warpack band ILLEGAL", 4.4 GPU-h fallback, legal control re-anchor | `runs/sealed/r17_thresholds.json` (`sealed_at 2026-07-22`) |
| R16 n=4 band (superseded) | `learnings/panel/r16_circulation.md:417-418` |
| A22 composition "NO warpack" | `learnings/war_room/a22_compaction_prereg_2026-08-01.md:77-78` |
| null10 run means, σ = 0.0860, pair sd 0.12234 | `runs/null10/vanilla_seed{101..110}.json` |
| sealed-K3′ lines, type-I, power; C(m); K3″ OCs | `D2_type_II_recalibration` (exhaustive + 400k bootstrap, seeds 20260810 / 99260810) |
| pooled σ̂ 0.1417 df 6, Bartlett, F CIs | `D4_warpack_null.pooled_buildrail_null`, `.free_warpack_null.variance_ratio_vs_vanilla` |
| power-design build counts | `D4_warpack_null.power_design` |
| 2608.07077, 2608.07429 | `learnings/sweeps/research_2026-08-10.md:49-80, 135-162, 270, 272` |
| all of the above, machine-readable | `runs/k3prime_fallout_2026-08-10.json` |
