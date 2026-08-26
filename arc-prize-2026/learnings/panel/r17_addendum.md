

---

## §12. ADDENDUM (2026-07-23, pre-circulation) — post-filing evidence

Filed the morning after the base document. Three pieces of evidence landed
after §1–§11 were frozen; none alters a sealed threshold (all thresholds in
`runs/sealed/r17_thresholds.json` predate these measurements — the seal-before-
measure discipline held), but all three bear on how the panel should rule.

### 12.1 Sentinel W1 live run (seed 1) — the condition-4 measurement

The v2 sentinel ran on Kaggle 2026-07-22 12:47–14:59Z (free build, 25 games,
`canivel/arc3-duck-sentinel-eval`; pull + analysis:
`runs/sentinel_eval_analysis/report.md`). Verdict: **mechanism PASS / score
NULL / behavior STRONG-NEGATIVE** — "fires, doesn't pay" (build-doc Open Risk
#2 realized):

- **Mechanism:** 22 sidecars + 56 stdout `SENTINEL v=2` lines agree exactly;
  every threshold fires ≤ once per game (≤3 events/game cap held everywhere);
  cumulative game-envelope keying proven live (fires at identical cumulative
  actions 75/113/135; no re-arm across attempts — e.g. ar25 crosses 50% in
  attempt 1 and 90% in attempt 2 with no repeat). The v1→v2 re-key is verified
  on the carrier games: ka59/re86 fired 3/3 early where v1 was structurally
  blind. Open Risk #1 (inert-if-uncapped) is cleared: the budget export was
  live and the banner present. Missing sidecars for s5i5/tu93/vc33 are
  EXPECTED (lazy file creation on first crossing; all three ended <75 actions;
  file-exists ⇔ ≥1 fire).
- **Envelope (condition 4):** tokens/game mean 64.3k; 23/25 games inside the
  sealed 63k ±15% band; B=150 requires no re-derivation. **Condition 4 passes
  on this pull.**
- **Score:** sentinel-arm mean 0.855 vs certified war 3-seed baseline mean
  1.454 (−0.60; −0.72 vs the paired seed). The gap is carried almost entirely
  by three high-variance NON-target games (ar25/ft09/sp80); baseline seed
  spread is 1.16–1.73. The honest call is **NULL/underpowered at n=1**, not
  established regression — but the pre-registered positive-lift framing
  (+0.01–0.03/draw) is refuted as optimistic.
- **Behavior:** the warnings did not change play. 1/22 fired games advanced a
  level after the first warning; 21/22 kept grinding (wa30: 560 actions stuck
  on L1 after all 3 warnings); total actions rose +618 vs baseline. tu93's
  efficient 3.97 draw fired ZERO events and must not be claimed as sentinel
  evidence.

**Proposed ruling for the panel:** seal the sentinel as a certified *observable*
(mechanism half sealed; condition 4 discharged), record the score prong as NULL
with the fires-doesn't-pay label, and register **W2 as a $0 confirmatory-null
free build** (pre-registration: mean inside 1.16–1.73, mechanism clean,
behavior unchanged; no W3 unless W2 is positive). The sentinel's certified
function was always warn-only; lift was window pricing, not a gate premise.

### 12.2 Scoring-function dissection (community, verified) — depth ≫ efficiency

Discussion #728299 reverse-engineers the shipped `arc_agi/scorecard.py`
(reproduced to 1e-9): the 115% figure is a **per-level efficiency cap**, and
the game/LB aggregate is **completion-weighted with a completion cap**. Two
consequences the base document's objectives should be read under:

1. **An unreached level costs its weight twice** (it contributes zero AND
   shrinks the completion factor): 4/6 levels ≈ 47.6, not 66.7. Deeper levels
   dominate marginal score.
2. **Overshoot decays quadratically** (2× baseline actions on a completed
   level ≈ 25%, not 50%) — inefficiency on completed levels is cheap relative
   to failing to reach the next level.

This independently explains 12.1's fires-doesn't-pay: a stop-grinding signal
cannot buy score unless the freed actions convert into level *depth*. It
re-points EWM/A17 value at reaching deeper levels, not action-trimming, and it
resolves the long-standing 1.15x-vs-1.0x watch-item from code (1.0x
completion-weighted is confirmed as the right LB mental model). The same post
ships a no-API-key offline scoring atlas of the 25 bundled games — adopted as
a free deterministic local scoring oracle (zero cloud spend).

### 12.3 External literature (2 ADAPTs, both strengthening the gate)

- **arXiv 2607.12227 (Jul 14), "Rethinking the Evaluation of Harness Evolution
  for Agents":** held-out evaluation of the tune-on-public/report-on-public
  pattern shows only +0.6 avg transfer. This is the external charter for the
  base document's gate discipline: **held-out beat-null10, never
  beat-baseline-on-the-tuning-games**, and semi-private weighted over
  public-25. (Also the right lens on Schema's public-only 99%.)
- **arXiv 2606.24842 (Jun 23), "World Models in Pieces: Structural
  Certification":** certification is **transition-local**, not model-global.
  This reframes §1's holdout collapse as the expected outcome (sb26 is the one
  transition-local certificate that generalized) and tightens the EWM v1.1
  wording: BFS-in-sim is sound only over transitions carrying a live local
  certificate. Proposed as wording (not threshold) amendment to the sealed EWM
  measurement config.
- **A17 boundary note (Kamradt critique, Jul 21):** the 27B→72B
  escalate-on-low-score template stays a **serving-cost-only** policy —
  per-game score feedback must never re-enter the agent's context, or the
  public-set leak the critique penalizes is imported. To be recorded in the
  A17″ amendment text.

### 12.4 Ledger as of circulation

Frozen control n=10: mean 0.975, σ̂ 0.156. Pooled n=15: mean 0.962, σ̂ 0.144.
Overnight draw 0.82 (band-typical). LB: field compressed — the 1.44 wall is
now the bottom of a dense 1.44–1.60 band; our 1.33 slid #44→#45 (erosion, not
regression). No new public clones above zoli800's 1.39 byte-identical draw.
