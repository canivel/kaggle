# Result deep-dive — 2026-07-27 (scored window 00:07Z)

## DRAW
- `canivel/arc3-duck-repro` v3 (frozen-fork filler, eternal fallback), fired 07-26
  20:07 EDT by ARCDailySubmit (submission_log.jsonl t=2026-07-27T00:07:12Z, ok=true).
- Public score: **1.02**. Pre-registered expectation: frozen-control draw from band
  0.82–1.33 — **MET** (comfortably interior; note 1.02 exactly repeats the 4th
  chronological frozen draw).

## LEDGER MATH
- Prior frozen stratum (07-26 brief, confirmed vs runs/lb_ground_truth.md + brief):
  n=12, {0.82, 0.89, 0.93, 1.02, 0.95, 1.33, 0.92, 0.93, 1.14, 0.82, 1.05, 0.84},
  mean ≈ 0.970, s ≈ 0.148.
- With today's 1.02: **n=13, mean ≈ 0.974, s ≈ 0.143** (sample sd; recomputed
  numerically). Standardized vs prior stats: t ≈ (1.02−0.970)/0.148 ≈ **+0.34** —
  unremarkable, slightly above mean.
- Band: protocol band stays the observed frozen range **0.82–1.33** (unchanged;
  1.02 is interior, no new extreme). Exact predictive-band recompute belongs to the
  ratified machinery — amendment 2026-07-24 (t-predictive model) was still DRAFT as
  of the 07-26 brief (R20 relaunch pending), so no re-derivation is claimed here.
- `runs/lb_ground_truth.md` NOT edited: precedent is that the daily loop refreshes it
  with live-API verification (last 07-25, still shows n=11); the 07-26 deep-dive also
  computed without editing. Same discipline followed.

## INTERPRETATION
- Clean in-band control draw; third-highest of the 13 frozen draws (ties the earlier
  1.02). Recent frozen tail: 0.82 (07-23), 1.05 (07-25), 0.84 (07-26), 1.02 (07-27) —
  alternation around the mean, no monotone run, no low-draw streak. The 07-24
  MK/CUSUM no-trend verdict stands; nothing here reopens it. No drift signal.
- Mean crept 0.970 → 0.974, s tightened 0.148 → 0.143: the frozen stratum keeps
  behaving like a stationary ~N(0.97, 0.145²) draw process with a 1.33 right-tail.
- Strategic read unchanged: filler rides while A17 is the only lane that can move us
  (LB best 1.33, gold wall ≈ 1.49). This draw carries zero information about A17.

## TRIGGERS
- **A21/C2 harm-pause (<0.80 pauses an exploration arm): NOT applicable** — this is
  the frozen CONTROL arm, and 1.02 ≥ 0.80 regardless. No pause.
- **Sealed W2 rule (r17_thresholds.json `legal_control_reanchor`, w0_s1 = 1.731):**
  governs W1/W2 sentinel EVAL comparisons only — not LB filler draws. No trigger.
- **OBJ-H kill-switch (held-out null10) and §8R16 guard (−0.28 lc/game):** eval-side
  conditions; untouched by a scored control draw. No trigger.
- Net: **no kill-switch, pause, or escalation condition fired today.**

## NEW ARTIFACTS (runs/, since 07-26)
- Nothing newer than the known 07-26 morning pair: `runs/warkit_verify_0726/`
  (07-26 08:42) and `runs/a17_recovery_replay/` (07-26 08:30); newest kernel pull is
  `runs/kernel_pulls/a17_canary_v3/` (07-26 08:31). Only later writes are the
  submission daemon's own logs (daily_submit_stdout.log, submission_log.jsonl).
- In-flight, no artifact yet: canary **v4 ERRORed at push** (model mount dropped
  again → model-mount route declared DEAD) and **v5 boot canary pushed 07-27** on the
  dataset-weights route (ITERATION_LOG 07-26 backfill + 07-27 stub; commit 66bc223).
  Expect its pull under runs/kernel_pulls/ in a later window.
