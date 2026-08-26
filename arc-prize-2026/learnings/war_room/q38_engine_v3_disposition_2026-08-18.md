# q38 engine-eval **v3** — DISPOSITION RULING (2026-08-18)
**Artifact:** `canivel/arc3-q38-engine-eval` v3, COMPLETE 2026-08-17 12:53:11Z, 2h12m08s.
**Provenance:** the 08-17 **misfire** — an unintended push that spent a slot. Never pre-registered.
**Pulled:** `runs/kernel_pulls/q38_engine_v3/` (benchmark.json, summary.txt, log).

## RULING: DESCRIPTIVE / NON-INFERENTIAL. It does **NOT** become PRIMARY-B's n=2.
Three reasons, in order of force:
1. **It was not pre-registered.** PRIMARY-B (actions/window vs B2) was sealed at n=1 with that
   fact stated in advance. Promoting a comparator *after* the run exists is exactly the move the
   seal exists to forbid.
2. **The decision is already contaminated.** The disposition is being ruled with the data pulled
   and read. Even a "fair" ruling made at this point is unfalsifiably post hoc.
3. **Nothing rides on it.** Q38 is already **REFUTE-2× (decisive)** on the sealed v2 read. A second
   draw cannot change a verdict that is not in dispute.

## What it legitimately delivers, free — and one of these matters a lot
**It is a same-engine, same-arm, SAME-SEED replicate.** Both banners read `seed=1
mode=engine-swap-local25 engine=saltb0x/qwen3-8-27b-fp8 … reasoning_effort=PINNED-medium arm=medium`,
identical wheels. So the v2↔v3 difference is **pure run-to-run nondeterminism**, not seed variation.

| | v2 (sealed) | v3 (misfire) |
|---|---|---|
| levels_completed | **21** | **17** |
| total actions | 2857 | 3127 |
| mean score | 2.79 | 2.91 |
| median score | 0.40 | 0.00 |
| runs won | 0 | 0 |

**1. ★ THE SEALED σ̂ SURVIVED AN INDEPENDENT CHECK — and it had never had one.** The entire
SCREEN_PROTOCOL promotion machinery rests on the standing pooled build-rail estimate
**σ̂ = 0.141740 lc/game (df 6)**, which implies **SD(lc_total) = 0.141740 × 25 = 3.5435 levels** and,
for the *difference* of two draws, **3.5435·√2 = 5.011**. Observed replicate gap **|21 − 17| = 4
levels = 0.80σ.** Fully consistent. This is the first time a same-config replicate pair has been
available to test that constant, and **it holds** — so the sealed bands on the live graft-floor arm
(HARM ≤12 · NULL 13–26 · SIGNAL ≥27) are **NOT** revealed as too tight. No threshold moves; none may.
*The honest converse, recorded: ±4 levels of pure nondeterminism is real, so a graft reading that
lands adjacent to a boundary is not decisive, and the prereg already accounts for exactly this.*

**2. An independent replicate reproduces REFUTE-2×.** Sealed scorer, run descriptively on v3:
`levels 17/25 (0.6800/game)` vs baseline m=3 `19.33 (0.7733/game)` → **mean dlc −0.0933/game,
z = −0.57 → REFUTE-2× (decisive)**. v2 refuted it; v3 refutes it again, further from the line. The
engine-generation hypothesis is now refuted on **two** independent draws. *This is corroboration in
a direction already settled — it is not evidence that was needed, and is not scored as such.*

**3. ★ THE FIRST CLEAN DECODE-RATE MEASUREMENT ON THIS RAIL.** Three consecutive model-level lanes
ended with no usable tokens/s because `generated tokens/sec` in summary.txt is total tokens over
*job* wallclock (server load + 25 games), not a decode rate. The prereg §16 probe fired here:
- `concurrency=1  requested=256 generated=256 exact_token_count=True elapsed=6.01s → **42.6 tok/s**`
- `concurrency=8  requested=2048 generated=2048 exact_token_count=True elapsed=6.55s → **312.8 tok/s**`
**Carries its own limit, verbatim from the probe:** *"synthetic fixed-concurrency rate; NOT comparable
to the job-wallclock tokens/sec in summary.txt, and there is no Qwen3.6 point in this series."*
⇒ usable as an **absolute** Qwen3.8 decode figure only; **no cross-engine claim may be built on it.**

**4. Effort pin certified:** `Q38-EVAL effort-pin=medium local-render reasoning_instruction=ABSENT
control(default=xhigh)=PRESENT pinned_chars=1260 control_chars=1469` — the pin bound, the positive
control fired. Consistent with the 08-16 finding that the pin binds.

## Consequence for the campaign
None that reopens anything. Q38 stays REFUTE-2× decisive; PRIMARY-B stays n=1 as sealed. The slot
the misfire cost is **not** recovered by this ruling and should not be described as recovered.
What is genuinely banked: **a validated σ̂**, an absolute decode rate, and a measured statement that
our 25-game build rail carries ~4 levels of replicate noise — which independently echoes forum
topic 735590's caution about how much weight one build-rail eval can bear.
