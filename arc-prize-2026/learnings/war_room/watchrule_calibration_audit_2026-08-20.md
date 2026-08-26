# SEALED ADDENDUM — calibration audit of the 0.41 WATCH RULE
**Written 2026-08-20 ~08:45 EDT — BEFORE tonight's 00:07 draw lands. Pre-registered, not post-hoc.**

## Why this exists
`feedback_audit_the_instrument.md`: audit the gate BEFORE the data lands. The coordinator adopted a
watch rule verbatim last night on the strength of `z = -3.44`. This pass re-derives it from
`runs/ledger.json` (n=37, mean 0.9316, s 0.1771) and finds the rule is **loose, and the mechanism it
was written to catch has no support in the data.** The artifact bytes of the rule are NOT edited;
this is an addendum that says how tonight's draw must be READ.

## Finding 1 — the mixture (PARTIAL-RUN DEATH) hypothesis has NO distributional support at n=37
- Filliben normality correlation on all 37 draws: **r = 0.9872**, critical value at n=37, a=.05 is
  **0.9665**. **Normality is NOT rejected.** (Dropping only 0.41 raises it to 0.9954 — the 0.41 is
  the single point degrading fit, and even with it in, the set passes.) Skew = -0.37.
- Draws below 0.65: **observed 1, expected under the pure Gaussian 1.06.** No excess whatsoever.
  A death-mixture firing at any appreciable rate would pile up **more** mass below 0.65 than we see.
- **Read: "occasional partial-run deaths" remains an untested STORY.** One outlier cannot distinguish
  "extreme tail of a Gaussian" from "mixture" — that is exactly what one point cannot do.

## Finding 2 — z = -3.44 overstates the surprise by ~36x (multiplicity)
- P(a single draw <= 0.41 | N(0.9461, 0.1558)) = **2.90e-04** — this is the number the -3.44 implies.
- But 0.41 is the **minimum of 37 draws**, not a pre-designated draw:
  P(min of 37 <= 0.41) = **0.0107** — a **~1-in-94 campaign event**, not a 1-in-3400 one.
- **Read: surprising, worth watching, NOT yet evidence of a broken rail.**

## Finding 3 — the watch rule's threshold is under-powered (THE ACTIONABLE ONE)
The rule: *"a second consecutive draw < 0.65 BREAKS stationarity and reopens structural investigation."*
Against the current band, **P(draw < 0.65) = 0.0559**. So:
- The rule's **false-alarm rate is 5.6% — it fires by chance about one night in 18.**
- It was adopted as if it were decisive. It is approximately a **one-sided a=.056 test on a single
  draw** — i.e. the weakest possible evidence standard, on the noisiest possible instrument, for a
  decision (reopening structural investigation) that costs days of campaign time.

Calibration table (false-alarm rate of "tonight's draw < T"):
| T | false-alarm | verdict |
|---|---|---|
| 0.65 | 5.59% | loose — as sealed |
| 0.60 | 3.06% | decisive |
| 0.55 | 1.56% | decisive |
| 0.50 | 0.74% | decisive |
| 0.45 | 0.33% | decisive |

## PRE-REGISTERED READ FOR TONIGHT (2026-08-21 00:07 draw) — sealed before the data
1. **draw >= 0.65** -> reversion. 0.41 stands as a **one-off extreme tail draw of a stationary band**.
   Fold into the ledger; do NOT reopen structural investigation; do NOT keep the death-mixture story
   alive in the log as though it had evidence.
2. **0.60 <= draw < 0.65** -> the sealed rule technically FIRES, but at a=5.6% this is **NOT decisive**.
   Record as WATCH-CONTINUES, n=38, re-derive. Do **not** spend a build slot on structural work.
3. **draw < 0.60** -> **DECISIVE (a<=3.1%).** Stationarity is broken on the low side. Reopen structural
   investigation and treat the rail — not the arm — as the first suspect.
4. Regardless of branch: the ledger must be re-derived by `scripts/ledger.py` before any prereg reads
   it. The bar moves every night. Never cache it.

## What would ACTUALLY test the death-mixture (none of which we have today)
The public LB gives one aggregate number; it cannot show a partial run. Distinguishing the mixture
needs **per-game scores from a rerun**, which the competition rerun does not expose to us. Until such
an artifact exists, "partial-run death" must be labelled **UNTESTED HYPOTHESIS** in every entry that
repeats it, not carried as a working mechanism.

**Status: no artifact bytes changed. This addendum governs the READ of the 08-21 draw.**
