# Daily brief — 2026-08-14

Sources merged: `deepdive_2026-08-14.md` (1a), `discsweep_2026-08-14.md` (1b),
`ressweep_2026-08-14.md` (1c). Where they disagreed, the ledger-derived number wins and the
disagreement is recorded rather than smoothed over.

---

## 0. The one thing to take away

**The filler lottery is arithmetically dead as a route to gold, and today is the day we can say so
with a number instead of a feeling.** At our current draw distribution (n=31, mean 0.9368,
s 0.1556), across the ~80 draws remaining to Nov 2:

| quantity | value |
|---|---|
| P(max@80 ≥ gold 1.58) | **0.14%** |
| P(max@80 ≥ prize 1.64) | **0.02%** |
| P(max@80 > our own current best 1.33) | **37%** |

We are more likely than not to *never beat our own best score again* by resubmitting the frozen
fork. The frozen fork remains the correct **eternal fallback** — it is not a strategy.

The escape route is unchanged and still intact: the efficiency reframe. At μ = 1.26 / 1.31 / 1.36,
E[max@80] = 1.636 / 1.686 / 1.736 — **all three clear gold** — *provided σ does not collapse*
(the 08-13 σ-sensitivity table stands: μ=1.26 with σ→0.13 misses). Raising the **mean** is the whole
game. Nothing today changed that; two things today sharpened what it will cost.

---

## 1a. Result deep-dive — the 0.70

**It is a non-event.** z = −1.61, 3rd-lowest draw (record min stays 0.65), 6th sub-0.80. E[max@80]
moves by **−0.0030** — 1.2% of the gap to gold. The mean fell and σ rose; under max-of-k they very
nearly cancel. **Do not re-plan around it.**

**Ledger was stale and is now current:** n=31, mean 0.9368, s 0.1556, trailing-4 0.910, public max
**unmoved at 1.33**, promotion bar **1.0876 → 1.0771**. Two draws had landed, not one (08-13 = 0.78,
08-14 = 0.70). Verified by reproducing 0.9503/0.1513 exactly on dropping the last two, and by
byte-identity of the first 24 elements against the sealed 08-07 series.

**The watch-rule fired, and yesterday we said it hadn't.** Wrong instrument (a threshold R23 retired
for *gated arms only*, applied to the *filler* rule) and wrong arithmetic (trailing-4 *movement* vs
`1.5×ledger-s`, instead of the draw vs `trailing-4 mean − 1.5 × trailing-4 s`). Applied as sealed it
fires on both draws. **Re-run correctly ⇒ STATIONARY** (change-point p = 0.817 — *weaker* drift
evidence than 08-07's 0.757; MK p=0.634; CUSUM −3.582 < h=4). Verdict unchanged, but only by luck.

> **Pre-registered now, before tonight's draw:** GUARD-2's escalation clause reads **three
> CONSECUTIVE** sub-0.80 (0.87 on 08-08 reset the counter). We are at two. **If tonight is also
> < 0.80, escalation fires and a re-baseline is owed.** Written down in advance so it cannot be
> re-read afterwards.

---

## 1b. Discussions + leaderboard

**The board did not move at all.** Every top-20 name and score identical to 08-13; largest delta
anywhere = 0.00. cstl **flat at 2.70** (resubmitted, no gain). Prize line flat 1.64 (2nd day), gold
flat 1.58 (**6th day**).

**★ Our rank is #100, not "below #49".** That figure had been carried unrecomputed since 08-09; the
last real count was #63 on 08-01. Counted directly: 94 strictly above, 9 tied at 1.33 (#95–#103).
**−37 ranks in 13 days on an unchanged score.** The flat gaps to gold/prize are the comfortable
number; the rank is the honest one, and it is the one that says the field is compounding while we
are not.

**★ cstl record correction — this refutes something we believed.** cstl did **not** enter at 2.52.
Our own archived CSVs show it at **1.59 from 08-04 to 08-09** — *inside the duck band* — then
1.59 → 2.52 in a single submission on 08-11, → 2.70, → flat. So the +0.93 is a delta achieved on
what is very likely the same artifact family we run. **That refutes "≈1.26–1.36 is the efficiency
ceiling" as a property of the FAMILY**: the family's demonstrated ceiling is **≥ 2.70**. The ceiling
we measured is a property of **our configuration**, not of the harness.
*(INFERRED, not proven: "same artifact" rests on score proximity to the duck band, not on evidence of
their code. Also, the LB shows best-score-with-latest-date, so cstl's flat scores say nothing about
determinism.)* Mechanism: **no trace found.** Traced to *who* (a competitive-simulation-agent
pedigree, not an LLM-prompting one) but not to *what*, and no mechanism is being proposed.

**3 new posts, 0 ADOPT.** One schedule-only ADAPT: three independent reports of **RTX PRO 6000 queues
at 3–8h today** — relevant because our envelope screen is on that GPU, so a long Queued state today
is *not* a build defect. The "Dynamic Value Model 14% vs 6%" post is IGNORE — its own repo says "no
weight updates, no auxiliary calls, no search branches", i.e. the retired advisory class, n=1 with an
author-acknowledged API-outage confound.

---

## 1c. Research sweep — 16 items, 0 ADOPT, 2 ADAPT

**The item that bit today was a 4-month-old paper we had never read**: the *qs* Inequality
(arXiv:2603.08960). MoE's "double penalty" is **context-length dependent** ⇒ our corrected 1.5×
per-token read edge is a *static* ceiling that **erodes as context grows**. Our canary probes at a
~6k prompt; the deployed regime is `ANALYZER_CONTEXT_WINDOW=32768`. **Registered against the B122 v2
readout at 08:39 EDT with the kernel still RUNNING** — no gate changed, because every optimism it
adds pushes the same way the sealed asymmetry already does (PASS means less; FAIL is more decisive).

**★ The sharpest constraint we acquired is a bound on our own three-day-old doctrine.**
arXiv:2608.04066 built the **maximal deterministic-executive agent on ARC-AGI-3** — LLM files only
typed proposals, code owns all belief — and reports **0 level completions across 52 pre-registered
runs**. So "executive > advisory" needs a narrow boundary: **replay a known-good action; never take
the choosing away.** That bounds EpicStar (2608.12626, executive replay gate, order-of-magnitude
fewer tokens — ADAPT) before we spend anything on it.

**CoAdapt-GUI (2608.11588) is a verdict on our live LoRA lane, not a mechanism for it** — ADAPT.
Unseen apps, no target demos: **policy-only LoRA 37.5% vs context+policy 45.0%**. First quantified
evidence that **R-7 is real** and that the adapter is the *smaller* half of transfer. If the lane is
pre-registered, a Policy-Only cell is now mandatory.

Stated nulls: no ARC-AGI-3 paper in the window (the entire literature is 16 papers); no
action-efficiency paper at all; **nothing anywhere on LoRA over an NVFP4 base with mixed-precision
attention in vLLM** — an engineering unknown only a smoke build settles.

---

## 2. The largest unclaimed finding on the table

**66% of turns emit no world-model update at all.** This came out of an independent replication over
596 archived event files / 50,140 model responses. It is the *same* "the agent FORGOT" root cause we
diagnosed on 08-12 — and it is **~66× larger than the reasoning-channel bug we patched yesterday**.

**No open lane targets it.** Every current lane (122B brain swap, LoRA adapter) is trying to make the
model *better*; this says the harness throws away two thirds of what the model already produces.
→ **Sunday panel, top of agenda.**

## 3. Instrument defect — blocks the memory-channel rider

The 08-13 ADOPT pre-registered a delivery band of **[0.5%, 3%] of turns**, estimated from 4,441 turns
(0.7–1.6%). The 50,140-response replication puts the true loss at **0.32%** — *below the
pre-registered floor*. **The rider would fail its own gate while working correctly.** The symptom
replicates (65.7% of turns have no visible content vs the poster's 66.8%); the harness captures
12,262 of 12,420 labels = **98.7%**.

**Not re-sealing unilaterally.** The rider is not shipping today (neither slot carries it), so the
correct move is to fix the band *before* it can fire, at Sunday's panel, with the better estimate in
hand. Recommended: report the **recovery rate** (158/12,420 = **1.27%**) as the primary quantity —
it is denominator-stable — and re-anchor the band on the 11×-larger sample.
**Blocking: the rider must not ship until the band is corrected.**

---

## 4. Open questions

1. **Does the 122B's edge survive at 32k?** A 6k PASS does not license a 32k deployment. Proposed as a
   pre-registered kill condition for Sunday: if tok/s at 32k falls below the 27B's, the swap dies
   regardless of its batch-1 number.
2. **What is cstl's +0.93 on a duck-band artifact?** The family ceiling is ≥2.70 and our
   configuration reaches 1.33. That gap is now the most valuable unexplained number in the campaign —
   and it is evidence the ceiling is ours, not the harness's.
3. **Why does 66% of turns produce no world-model update** — is it prompt, parser, or schema? Cheap to
   answer, zero GPU.
4. **Does the LoRA lane survive CoAdapt-GUI?** If the adapter is the smaller half of transfer, the
   lane's H1 needs a Policy-Only cell or it will over-attribute.
5. **Rank −37 in 13 days.** Is the field's compounding coming from better agents or more draws? This
   determines whether "raise the mean" is sufficient or merely necessary.
