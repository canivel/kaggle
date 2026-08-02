# Draw deep-dive — 2026-08-01 frozen-fork filler 0.65 (campaign low)

**Analyst note (measurement discipline):** this is a distribution / trigger analysis of a
single control-arm draw. NO capability claim is made or licensed from n=1. The frozen fork
is a byte-identical filler; a low draw carries no information about any experimental arm.

---

## 1. Verification (Task 1)

- Live API: `uvx --from kaggle==2.0.0 kaggle competitions submissions arc-prize-2026-arc-agi-3`.
- Newest row: `submission.parquet`, **2026-08-01 00:07:11Z**, status
  **SubmissionStatus.COMPLETE**, publicScore **0.65**.
- Description (verbatim): *"frozen-fork filler (eternal fallback; band 0.82-1.33, record
  ledger n=17 after 07-31 draw 1.10; A/B control frozen n=15 mean 0.9727 s 0.1343; nothing
  fires tonight — gate-eval BUILDS in flight, entry-gate discharge pending)."*
- Confirmed: this is the byte-identical frozen-fork filler `canivel/arc3-duck-repro` v3
  (upstream `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`). The draw is
  genuine (COMPLETE, not errored) and is a **control-arm / record filler**, not any
  experimental arm. **VERIFIED.**

---

## 2. Leaderboard cross-check — isolated low draw vs platform-wide shift (Task 2)

Full public LB CSV pulled 2026-08-01T12:28Z (2001 teams). Head vs the 07-31 snapshot in
`runs/lb_ground_truth.md`:

| rank | 07-31 snapshot | 08-01 pull | moved? |
|---|---|---|---|
| #1 | YUTO KOJIMA 1.86 | YUTO KOJIMA 1.86 | no |
| #2 | Andy liu 1.69 | Andy liu 1.69 | no |
| #3 | GeniusYY 1.64 | GeniusYY 1.64 | no |
| #4 | Tecnod8.AI 1.61 | Tecnod8.AI 1.61 | no |
| #5–6 | DhanaLakshmiMalla 1.60 | FOYSAL 1.61 / DhanaLakshmiMalla 1.60 | new entrant, churn |
| gold #13 | ≈1.50 | **1.54** (paul) | drifted UP |

**Findings:**
- **The head is frozen.** Every top-4 team holds the exact same score as 07-31. There is
  **no platform-wide rescoring, no game-set change, no eval-infra shift** — those would move
  the top scores, and they did not.
- **Gold cutoff drifted UP** (#13 ≈1.50 → 1.54) purely from normal new-submission churn in
  the dense 1.47–1.61 band (FOYSAL 1.61 new #5, Nkosi Ndwandwe 1.58, paul/Seok 1.54 new).
  This is ordinary competitive movement, unrelated to our 0.65.
- **Our banked 1.33 is byte-for-byte intact.** Team "Canivel" sits at **#63** (58 strictly
  above, 7 tied, ranks 59–65). If a platform rescoring had cut our 0.65, our banked-best
  1.33 would also have been rewritten — it was not. Our rank slipped 07-28 #51 → 08-01 #63,
  but that is **pure competitive drift** (other teams climbing the band), not a change to
  any of our own draws.

**Conclusion: the 0.65 is an ISOLATED left-tail low draw of our own frozen fork.** No
evidence of platform-wide rescoring, game-set rotation, or eval-infra change.

---

## 3. Tail arithmetic (Task 3)

All via `uv run python` (numpy + scipy) on the API-verified ledger.

**Record stats:**
- Before draw (n=17): mean 0.9729, s 0.1332.
- With 0.65 (n=18): **mean 0.9550, s 0.1500** (mean −0.018, s +0.013 from the one low draw).
- Frozen A/B control (n=15, sealed): mean 0.9727, s 0.1343 — unchanged (0.65 does not
  accrue to it, per prereg §3).

**z-scores of 0.65:**
- vs frozen n=15 control: **z = −2.402**
- vs n=17 record: **z = −2.424**

**Tail probability — is −2.4σ tail-consistent or shift evidence?**
- P(single draw ≤ 0.65 | N(0.9727, 0.1343)) = **0.813%** (Gaussian) / **1.78%**
  (t-predictive, ν = 14, sd_pred = 0.1388).
- **P(≥1 of 18 draws ≤ 0.65) = 13.67% (Gaussian) / 27.6% (t-predictive).**

**Reading:** over 18 draws from a stationary frozen distribution, seeing at least one draw
this low is a **1-in-7 (Gaussian) to better-than-1-in-4 (t)** event — i.e. *expected*, not
surprising. A −2.4σ single observation in a run of 18 is **tail-consistent with
stationarity**; it is NOT distribution-shift evidence. (Contrast: a distribution shift would
show up as a *run* of low draws or a mean/level break, not one isolated point — the prior
17 draws sit in 0.82–1.33 with MK/CUSUM stationarity on record.)

For context, the per-draw sub-0.80 floor probability under the frozen null is 9.94%
(Gaussian), so a draw below 0.80 is itself a ~1-in-10 healthy event; 0.65 is 0.15 below
that floor, in the deeper tail, but still inside the envelope a stationary noisy metric
produces over an 18-draw run.

---

## 4. Prereg clause audit (Task 4)

Read `learnings/war_room/boristown_ab_prereg_2026-07-29_DRAFT.md` in full. Relevant clauses:

**(a) The harm-pause / ABORT rule is scoped to GATED ARM B draws only — it does NOT apply
to a control-arm filler.** §3 (verbatim):
> "**ABORT / harm-pause (per draw, sealed A21/C2):** any **gated draw** < 0.80 pauses the
> arm pending panel review. Because the gate's entire mechanism is *left-tail removal*, a
> sub-0.80 draw is **evidence against H1**, not merely exposure control."

The 0.65 is a **frozen-fork control filler**, not a gated arm B draw (arm B is
`canivel/arc3-duck-gate`, unbuilt/unfired — the entry gate is not even discharged, BLOCKER 3).
The harm-pause floor is **not touched** by this draw.

**(b) The control is explicitly SEALED / FROZEN — later draws do not perturb it.** §3
(verbatim):
> "**Control parameters (frozen ledger, n=15, API-verified `runs/lb_ground_truth.md`
> 07-29):** mean **x̄_C = 0.9727**, s **= 0.1343** … Verified by `uv run python` on the
> 15-value list … → mean 0.972667, s 0.134349."

And §1 (verbatim):
> "Control = the banked frozen-ledger draws (**n=15** as of 07-29; new fillers may interleave
> — see §2). **No new draws are commissioned *for* the control; it is already banked.**"

The 0.65 accrues to the *record ledger* (now n=18), **not** to the sealed control (still
n=15). The control parameters are frozen by construction.

**(c) No drift / stationarity / control-invalidation clause is triggered by a control-arm
draw.** The prereg's only per-draw trigger is the gated-arm harm-pause in §3 (scoped to arm
B, above). The changepoint/drift monitor referenced in §3 arms only **PROMOTE ⇒ NC-6** ("the
changepoint monitor arms for the first 5 post-gate draws") — i.e. *after* a promote, on the
gated variant, not on control fillers. There is **no clause that a low control filler
invalidates the control, re-opens the seal, or pauses the program.** The §2 stationarity
statement ("the control is already banked (n=15) and stationary on record (MK p≈0.47, CUSUM
p≈0.72)") is a *justification for consecutive (non-alternating) gated draws*, not a live
trigger — and one isolated tail draw does not overturn MK/CUSUM stationarity (a level break
or run would).

**Plain-English dispositions:**
- **(a) Today's planned seal:** **NOT AFFECTED.** The seal (§7.1, git-commit) fires on the
  fork-diff evidence and the sealed arithmetic — none of which reference control fillers. The
  0.65 changes no input to the seal. Seal may proceed on schedule (Saturday 08-01 → Sunday
  08-02 panel ratification).
- **(b) The frozen n=15 control's validity:** **NOT AFFECTED.** The control is sealed at
  n=15 by §1/§3; new fillers accrue to the record only. The 0.65 does not enter it and cannot
  invalidate it. (It is worth noting the *record* mean fell to 0.9550, but the control's
  0.9727 is the sealed number the decision rule uses.)
- **(c) The promote threshold ≥1.0970 arithmetic:** **NOT AFFECTED.** The threshold is
  `0.9727 + 1.645·0.0756 = 1.0970`, computed from the sealed n=15 control (mean 0.9727,
  s 0.1343). Those inputs are frozen; the 0.65 does not touch them. The threshold stands
  unchanged at **1.0970** (Gaussian, governing) with the t-robust cross-check bar 1.1269.

---

## 5. Verdict & recommendation

**VERDICT: isolated left-tail low draw of the byte-identical frozen fork — tail-consistent
with a stationary distribution, NOT distribution-shift / rescoring / eval-infra evidence.**

Key numbers:
- z(0.65) = **−2.40** (vs frozen n=15 control) / −2.42 (vs n=17 record).
- P(≥1 of 18 draws ≤ 0.65 | frozen null) = **13.7% Gaussian / 27.6% t-predictive** → expected.
- LB head frozen (top-4 unchanged), our banked 1.33 intact at #63 → isolated, not platform-wide.

Prereg clause triggered: **NONE.** The only per-draw trigger (harm-pause <0.80) is scoped to
gated arm B, which has not fired; the control is sealed at n=15; the promote threshold inputs
are frozen.

**Recommendation for the seal decision:** **PROCEED with the planned seal.** The 0.65 does
not block, modify, or otherwise touch (a) today's seal, (b) the frozen n=15 control's
validity, or (c) the ≥1.0970 promote arithmetic. Log the draw to the record ledger (done in
`runs/lb_ground_truth.md`, n=18) and note it as a record low for band-tracking, but take no
program action on n=1. If a *second* sub-0.80 draw lands in the next few control fillers, that
would warrant a stationarity re-check (run vs level break) — a single −2.4σ point does not.

---
*Prepared 2026-08-01. All statistics `uv run python`-verified against the API-verified
ledger. No capability inference from a single draw (measurement discipline).*
