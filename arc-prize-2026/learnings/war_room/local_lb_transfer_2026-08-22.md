# LOCAL→LB TRANSFER: what our own record says (2026-08-22)

**Trigger.** Forum topic **736578** ("Public vs. Private Discrepancy", Nick Pellegrin, 08-21 15:56Z, 4 votes / 2 comments) reports:
duck harness + qwen3.8 → local **2.1** on the 25 public games, LB **~1.4**; his **own** harness → local **5.0–5.4**, LB **still ~1.4**.
i.e. a **2.5× local gain that bought 0.00 on the LB.** If that is a property of the instrument rather than of his harness, our entire
screening rail (sealed lc bands on the local 25-game benchmark) is inferentially void.

## Method
Local lc recomputed for every retained artifact with the **canonical reader** (`_lc_actions_score`, real key `levels_completed`,
the one repaired this morning) — it reproduces five sealed values **exactly** (field 28/6.173/1639, ArmA 30/5.686/1463,
edge-1 18/2.909/1113, Arm3 18/3.217/1251, graft-confirm 14/1.202), so it is certified against real artifacts, not fixtures.
LB means taken from the 50-row submission history, grouped by the config that produced them.

## The local null distribution (this is the number we never had)
`runs/null10/vanilla_seed10{1..10}` — vanilla duck, **10 seeds**: `[16, 11, 16, 15, 16, 15, 14, 18, 18, 13]`
⇒ **mean 15.20, sd 2.15.** A single-seed local read therefore carries **sd ≈ 2.15 lc (14% of base)**.

## The anchor table (configs with BOTH a local artifact and LB draws)
| config | local lc | z vs null | local ×null | LB mean (n) | LB ×null |
|---|---|---|---|---|---|
| vanilla duck (frozen filler) | 15.20 (n=10) | — | 1.00 | **0.9316** (n=37) | 1.00 |
| war-v1 — **seed 1 only** | 22 | **+3.16** | 1.45 | 0.9360 (n=5) | **1.00** |
| war-v1 — **family mean** (v1/v2/v3 = 22/15/13) | 16.67 | +0.68 | 1.10 | 0.9360 (n=5) | 1.00 |
| duck-sentinel | 12 | −1.49 | 0.79 | 0.7100 (n=1) | 0.76 |
| attempt-scheduler | 17 | +0.84 | 1.12 | 0.9000 (n=1) | 0.97 |
| **field-floor (Q38 xhigh, 08-07)** | **28** | **+5.95** | **1.84** | **1.5850** (n=2) | **1.70** |

## Reading
1. **The poster's failure mode EXISTS in our record — and it is a single-seed artifact, not a transfer failure.**
   `war_eval_v1` scored **22 (+3.16σ)** on one local seed. It bought **+0.004 LB over 5 draws (×1.00).** But the war family's
   *other two* local seeds were 15 and 13; the **family mean 16.67 (×1.10)** predicted the null LB correctly. The local
   instrument was right; the **single-seed read** of it was wrong.
2. **Large local effects HAVE transferred, near-proportionally.** Field-floor: local **×1.84** → LB **×1.70**. The two ratios
   agree to **8%**. This is the only large local effect we have ever submitted, and it transferred.
3. **We have zero internal cases of a large, replicated local gain failing to transfer.** What we have is one within-noise
   effect that correctly read null, and one +6σ effect that correctly read strong.
4. **What our record CANNOT answer** — and where the poster may still be right about himself: every config in the table is
   **duck-lineage**. A from-scratch harness has far more freedom to overfit the 25 public games; ours inherits its
   exploration policy from the fork. His "own harness at 5.4 local" is exactly the untested cell. **Disposition: ADAPT, not IGNORE.**

## Consequence for our sealed gates (filed PRE-OBSERVATION, before the edge-2 number is seen)
The edge-2 band **±5 lc on n=1 seed = ±2.33σ** of the single-seed null sd. That is a defensible screen, and the gate
**stands unchanged** — moving it after reading the forum would be moving goalposts. But the war-v1 precedent is direct
internal evidence that a **+3σ single-seed local SIGNAL can be a draw artifact**, so:

> **A certified single-seed SIGNAL is REPLICATION-PENDING, not established.** It may head the queue (heading is nearly free:
> the public score is a MAX over submissions, so an experimental head risks no banked score), but it may **not** be written
> into the record as a confirmed mechanism until a **second seed** reproduces it.

This costs nothing tonight and prevents the war-v1 error from recurring at the top of the funnel.

---

# ADDENDUM — the war-v1 case is now VERIFIED FROM ARTIFACTS, and it sharpens into the day's real finding

The objection "you pooled three *different* configs into a fake family mean" was tested against the artifacts, not the prose.
**All three war_eval runs carry the identical config label `duck-harness-kaggle-warpack-v1`**, identical `n_passes`/`solver_label`,
run on three consecutive days (07-14 / 07-15 / 07-16). They are **three seeds of ONE config: lc 22, 15, 13.**
(Independently consistent with the 08-10 post-panel overturn, which had already found `war_eval_v3−v1` to be two identical
no-compaction runs.) The pooling is legitimate.

## The measured seed-noise floor (we had never estimated this within-config)
| config | seeds | mean | sd |
|---|---|---|---|
| vanilla duck | 10 | 15.20 | **2.15** |
| warpack-v1 | 3 | 16.67 | **4.73** (range 9) |
| **pooled** | — | — | **2.80 lc** (df 11) |

F = 4.83 (df 2,9) — the two configs plausibly differ in variance, so 2.80 is a *floor*, not a ceiling.

## THE ERROR, stated exactly
war-v1's headline seed of **22** was judged **+3.16σ** — against the **vanilla null**.
Against **its own config's replicates** it is **+1.13σ**. Nothing.
**We compared a single seed to the wrong reference distribution.** The LB then said ×1.00 across 5 draws, and it was right.

## What this does to our sealed ±5 lc single-seed bands
| reference sd | ±5 lc equals | one-tail false-positive |
|---|---|---|
| vanilla-only 2.15 | 2.33σ | 1.0% |
| **pooled 2.80** | **1.79σ** | **3.7%** |
| warpack-only 4.73 | 1.06σ | 14.5% |

So a ±5 band on **n=1 seed** is a **1.1σ–2.3σ** screen depending on which config's noise applies — materially weaker than
the "diff-SD 5.011 context" the prereg cites. Re-reading our single-seed calls against pooled sd 2.80:
- **edge-1 Δ−12 = −4.3σ** — survives comfortably. HARM stands.
- **Arm 3 Δ−10 = −3.6σ** — survives. (Its vehicle/bundle confounds are separate and still binding.)
- **Arm A Δ+2 = +0.7σ** — NULL stands, comfortably.
- **edge-2's pending ±5 gate = 1.8σ** — the weakest call we will have made. A bare +5 is **not** a mechanism.

## Revised consequence (still filed PRE-OBSERVATION — the edge-2 number has not been seen)
The gate **stands unchanged**; moving it now would be moving goalposts, and heading is nearly free under MAX scoring.
But the replication-pending rule is upgraded from prudence to arithmetic:

> **A single-seed SIGNAL in the +5..+8 range (≤~3σ pooled) is a DRAW, not a finding.** It may head the queue. It may not be
> written as a mechanism, entered in the registry as a confirmed effect, or used to justify a follow-on build, until a
> **second seed** reproduces it. A single-seed SIGNAL ≥ +9 (≥3.2σ pooled) may be written as provisional.

**And the standing instrument rule this generalises to:** *judge a seed against its own config's replicates, never against a
different config's null.* Every screen we have run has n=1 seed per arm; only vanilla and warpack-v1 have replicates at all.

## Provenance honesty — what is [V] and what is [INF]
- **[V] VERIFIED from artifacts:** the war_eval v1/v2/v3 config identity (identical `label`/`n_passes`/`solver_label`), their
  lc 22/15/13, the within-config sds, the canonical reader's agreement with five sealed values, and every LB score
  (pulled from the 50-row submission history).
- **[INF] INFERRED, and it matters:** that `runs/null10` (label `duck-null-seed*`, run 07-11/07-12, carrying `PHASE1_EXPLORE_*`
  env) is the *same* config as the frozen fork whose 37 LB draws average 0.9316. It is duck-lineage and was built as the
  campaign's canonical local null, but it is **not proven byte-identical** to the submitted fork.
  ⇒ **The ×null transfer ratios in the anchor table are approximate.** Treat "local ×1.84 → LB ×1.70" as *directionally*
  supported, not as a calibrated mapping.
- **Immune to that caveat:** the seed-noise finding and the war-v1 diagnosis rest entirely on **within-config replicates**
  and need no cross-config baseline at all. That is the load-bearing result, and it stands on [V] evidence.
