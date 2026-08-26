# SCREEN PROTOCOL — canonical, binding on every build-rail arm screen

**Status: CANONICAL.** Created 2026-08-10. Before this file existed the screen rule lived in three
places and no protocol: inline in `runs/kernel_pulls/a22_v2_1/_screen_m1m2.py:116-119` (and its
v1/v2 twins), restated per-arm inside each prereg, and sealed numerically in
`runs/sealed/r17_thresholds.json`. That is how three lanes were killed on a gate with a measured
51.1 % false-FAIL rate. **This file is now the single source of truth for the non-harm screen. Any
prereg that says "M1 inherited VERBATIM" inherits THIS.**

**Supersedes:** the K3 gate `mean Δlc ≥ −0.128 AND worst-game Δlc ≥ −1.0` (measured type-I 51.1 %
on the campaign's own null — `learnings/sweeps/gate_recalibration_2026-08-09.md` §1.2) and the
sealed-but-miscalibrated K3′ of R24 §5.1 (looser at m=3 than at its own m=1 fallback — R25
methodology N1, reproduced in `learnings/sweeps/k3prime_fallout_2026-08-10.md` §D2.1).

**Derivation and every measured number:** `learnings/sweeps/k3prime_fallout_2026-08-10.md`,
`runs/k3prime_fallout_2026-08-10.json`, script `duck_eval/r24_prep/k3prime_fallout.py`.

---

## 0. Scope

Applies to any arm whose promotion/kill decision reads a levels-completed (lc) endpoint from a
free Kaggle build-rail run. One decision statistic: **mean Δlc over the games**. Everything else
reported is descriptive and non-inferential.

---

## 1. Hard preconditions — an arm that fails any of these is NOT SCREENABLE

A screen that runs anyway is void, and its verdict may not be entered in the record in either
direction.

### P1 — same-config legality of the baseline family

The baseline runs must share the arm's configuration in **everything except the one
pre-registered change**. Verify this from the **run logs' mechanism banners**, and record the
evidence in the prereg.

- **`label` alone is insufficient** but is a required first check.
- **`git_status.txt` is NOT evidence of anything.** It is byte-identical (md5 `5bca49fe…`) across
  every kernel pull on disk, including arms of demonstrably different configs. Do not cite it.
- Cross-check the banners: e.g. `NO warpack/ledger-graft/sentinel` vs `warpack: banking`;
  `COMPACTION ` event lines present/absent; `SENTINEL v=` present/absent.
- **Standing rulings already in force:**
  - The warpack band is **ILLEGAL** as a control for `{(f)}`-envelope arms
    (`runs/sealed/r17_thresholds.json → thresholds.control_band`, sealed 2026-07-22;
    `thresholds.legal_control_reanchor`). It may be reported as **diagnostic only**.
  - This **supersedes** R16 §11's n=4 warpack-inclusive band
    (`learnings/panel/r16_circulation.md:417-418`), which is later-overturned and must not be
    cited as legal.

*Worked failure:* all three A22 compaction screens (2026-08-02/05/06) paired a
`NO warpack` arm against the `warpack: banking` run `war_eval_v1`, 11 days after R17 sealed that
comparison illegal.

### P2 — m ≥ 3 same-config baseline runs

Pair the arm against the **per-game mean of m ≥ 3 runs of the same config**, not against a single
run. `m = 1` and `m = 2` are **not screenable**; report as advisory and say so in the same
sentence.

Rationale (R24 §5.1): a single baseline run is one draw from the config's own seed spread, and
the campaign has already killed a lane against a documented high outlier (`war_eval_v1`, 22 levels
in a 22/15/13 family).

**Current m by family** (`runs/k3prime_fallout_2026-08-10.json → inventory.families`; refresh
before every screen):

| family label | m | lc totals | screenable? |
|---|---|---|---|
| `duck-null-seed1xx` (null10 vanilla) | 10 | 16,11,16,15,16,15,14,18,18,13 | yes |
| `duck-harness-kaggle-warpack-v1` | 3 | 22, 15, 13 | yes |
| `duck-harness-kaggle` | 3 | 18, 19, 21 | yes |
| `duck-harness-kaggle-continuation-v1` | **2** | 10, 16 | **NO — 1 seed short** |
| `duck-harness-kaggle-sentinel-v2` | **2** | 12, 16 | **NO — 1 seed short** |

### P3 — σ̂ estimated with df ≥ 4

`σ̂` = run-to-run sd of *per-game mean lc* (= run lc total ÷ n_games) for the comparator's
**harness family**, estimated by **pooling within-family sums of squares** across same-rail control
families that run the same game instance set:

```
σ̂ = sqrt( Σ_f SS_f / Σ_f (m_f − 1) ),   df = Σ_f (m_f − 1)
```

A df=2 estimate is not an estimate: its relative SE is 50 % and its 90 % CI spans 7.6×.

**Standing pooled estimate, build-rail era (recompute when a family gains a run):**

> **σ̂_pooled = 0.14174, df = 6**, pooled over `…-warpack-v1` (SS 0.071467, df 2),
> `duck-harness-kaggle` (0.007467, df 2), `…-continuation-v1` (0.028800, df 1),
> `…-sentinel-v2` (0.012800, df 1). Bartlett χ² = 1.753, **p = 0.625** (no evidence against
> pooling). 90 % CI on σ **[0.0978, 0.2715]**. vs vanilla null10 (σ̂ = 0.08600, df 9):
> F(6,9) = 2.716, one-sided **p = 0.086** — carried as a **precaution**, not as a measured fact.

Do **not** quote "warpack variance is 4.83× vanilla" as established: F(2,9) = 4.832, p = 0.0376,
but **95 % CI [0.845, 190.3]**.

---

## 2. K3″ — the non-harm gate

> **PASS iff**
>
> ```
> mean Δlc  ≥  − C(m) · σ̂
> ```
>
> `mean Δlc` = mean over games of (arm lc − per-game mean of the m baseline runs).

**C(m)** — null10-measured, monotonised 5th-percentile multiplier of σ̂:

| m | 1 | 2 | 3 | 4 | 5 | ≥ 6 |
|---|---|---|---|---|---|---|
| **C(m)** | 2.33 | 2.10 | **2.02** | 1.98 | 1.96 | 1.94 |

(measured envelope 2.3257 / 2.0931 / 2.0156 / 1.9768 / 1.9535 / 1.9380, i.i.d. bootstrap over the
10 null10 run means, 400k draws per m. m = 1 and 2 are listed for advisory arithmetic only — they
fail P2.)

**Measured operating characteristics** (null10, independent evaluation stream, 400k draws/m):

| m | line at σ̂ = 0.0860 | type-I | power vs −0.10 | power vs −0.20 | power vs −0.30 | 80 %-power floor |
|---|---|---|---|---|---|---|
| 1 | −0.2004 | 2.0 % | 19.0 % | 40.0 % | 80.9 % | 0.297 |
| **3** | **−0.1737** | **4.4 %** | 20.8 % | **56.7 %** | 91.8 % | **0.253** |
| 5 | −0.1686 | 4.8 % | 21.0 % | 60.8 % | 94.2 % | 0.244 |
| 9 | −0.1668 | 5.2 % | 19.9 % | 64.5 % | 95.2 % | 0.239 |

**Properties guaranteed by construction:** C(m) is monotone non-increasing and σ̂ does not depend
on m, so **adding a baseline run can never widen the PASS band.** The superseded K3′ violated this
(its m=3 line, −0.2916, was 46 % wider than its m=1 fallback of −0.200).

**Ready-made lines at the σ̂'s currently on disk:**

| σ̂ | m=3 | m=5 | m=9 |
|---|---|---|---|
| 0.0860 (vanilla, df 9) | −0.174 | −0.169 | −0.167 |
| **0.1417 (pooled build-rail, df 6)** | **−0.286** | −0.278 | −0.275 |
| 0.1890 (warpack only, df 2 — do not use, fails P3) | −0.382 | −0.370 | −0.367 |

### 2.1 Legs that are struck

- **`worst-game Δlc ≥ −1.0` is STRUCK.** Measured type-I 50.0 % single-baseline / 60.0 %
  averaged-baseline; a −2 somewhere is the modal null outcome at 25 games
  (`gate_recalibration_2026-08-09.md` §3.2). It may not gate anything.
- **`#(Δlc ≤ −2) ≤ 2`** is **advisory, never gating** (measured type-I 4.4 %, near-zero power; and
  it is only defined against a single-run baseline, which P2 forbids anyway).
- **`actions_per_level_completed` may not be a co-primary** — the endpoint is in its denominator,
  and the binding resource on this rail is wall clock, not actions (R24 §3.1: all 25 games ended
  at 7920.2–7939.9 s against `max_actions_per_game=None`).

---

## 3. Sign-flip significance

The exact two-sided sign-flip permutation test is retained and is **correctly calibrated**
(0/90 null pairs ≤ 0.05; 5th pct 0.1875 — `gate_recalibration_2026-08-09.md` §1.4). It reports
significance; it is **not** the gate. Do not promote a p-value to a threshold — that is exactly how
−0.128 (a non-significant observed value, p = 0.9495) became a kill line.

---

## 4. Mandatory seal language — every screen, before the arm runs

A prereg that omits any of these is not sealed:

1. **The baseline family**: every run path, its lc total, and the banner evidence for P1.
2. **m**, and an explicit statement that m ≥ 3 (or that the arm is NOT SCREENABLE).
3. **σ̂, its df, and the families pooled** to produce it.
4. **The resulting K3″ line**, and **its measured type-I rate and the corpus it was measured on.**
   *A threshold without an operating characteristic is not a gate.*
5. **The 80 %-power detection floor, converted to levels over the game set:**
   `floor = C(m)·σ̂ + 0.8416·σ̂·√(1+1/m)`, `levels = floor × n_games`.
6. **The power-honesty clause.** If power at the affordable m against the arm's own hypothesised
   effect is **< 50 %**, the prereg must state in its own text that the run is an **exploratory
   probe, not a screen**, and that **no PASS may be reported as non-harm** — only as
   "uninformative in both directions".

*Worked example (why 5 and 6 exist):* an A22 arm screened at warpack σ̂ = 0.189, m = 3, has an
80 % floor of **0.566 lc/game = 14.1 levels over 25 games**, against a baseline mean of ~16.7
levels. The arm would have to score about 3 levels to fail. Reporting that PASS as "non-harm"
would be false.

---

## 5. Power design — how many runs a real answer costs

`(C + 0.8416)·σ̂·√(1/k + 1/m) ≤ |harm|`, k = arm seeds, m = baseline seeds, C = 1.94 at m ≥ 6.
Measured build cost **2.2110 h** (n = 31 completed 25-game runs, range 2.2021–2.2190), i.e.
**13.6 builds/week** at 30 GPU-h (minutes §5.4 quotes 12–13).

| σ̂ | detect −0.10 | detect −0.20 | detect −0.30 | ceiling (∞ baseline, 1 arm seed) vs −0.20 |
|---|---|---|---|---|
| 0.189 warpack | k=m=56 → 112 builds | k=m=14 → 28 builds | k=m=7 → 14 builds | **18.9 %** |
| 0.1417 pooled | k=m=32 → 64 | k=m=8 → **16 builds** | k=m=4 → 8 | 29.8 % |
| 0.0860 vanilla | k=m=12 → 24 | k=m=3 → 6 | k=m=2 → 4 | 65.0 % |

Read the ceiling column before proposing a screen: **more null runs cannot rescue a screen whose
σ̂ is too large.** On warpack-noise configs, an infinite baseline still gives 18.9 % power against
a −0.20 harm.

---

## 6. Change control

Amendments are **append-only, dated sections**; never edit a number in place. Any change to C(m),
to σ̂_pooled, or to the preconditions must ship with the re-run of
`duck_eval/r24_prep/k3prime_fallout.py` that produced it and an updated
`runs/k3prime_fallout_<date>.json`.

### Revision log
- **2026-08-10** — created. K3″ replaces K3′ (R25 methodology N1). P1 same-config legality and P2
  m ≥ 3 promoted to hard preconditions. P3 df ≥ 4 added. Worst-game leg struck. Seal language and
  power-honesty clause added.
