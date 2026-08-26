# TASK — DAILY SWEEP 2026-08-25 (STEP 1b discussions + STEP 1c research)

ARC-AGI-3 campaign, repo F:\kaggle\arc-prize-2026. ZERO GPU, no Kaggle pushes.
Use WebSearch / WebFetch. Today is 2026-08-25. Cover NEW material since 2026-08-24.

## WHERE WE ARE (so you can judge relevance, not just novelty)
- Our rail: the "duck harness" (TAAF / ARC3-Inference), a LOCAL served
  `Qwen/Qwen3.8-27B-FP8`, offline Kaggle rerun, 25 games, `max_runtime_s_per_game=7920`.
- Certified field-floor config: n=5 draws, mean **1.5760**, sd **0.2713** (draws incl. 1.92 today,
  1.16, 1.63, 1.58, 1.59). 1.92 is an ordinary max-of-5, it licenses nothing.
- Board: our Score 1.92, rank ~#146/2526. Gold line ~2.65, prize ~2.88. #1 cstl 5.99.
- **Every game dies on the 7920s clock** (198142s/25 = 7925s/game). The DECISION BUDGET is
  binding: we get ~1639 actions / 25 games. Efficiency, not raw capability, is the constraint.
- Actions per level cleared: AVO(Opus5) 36.2 / VISTA(Opus5) 41.2 / OURS(Qwen3.8-27B) 56.5.
  We are within 1.56x on per-action quality; we are behind on HOW MANY actions we get.
- Live program: C3 (analyzer yield-seconds) under verification today; P2 (reset/retry +
  memory amortization); S1 (a SUPERVISOR / stagnation-detector arm, being designed);
  C1/C2 (token caps) sealed behind those.
- Our measured defects: (i) the harness writes its carried world model ONLY from the model's
  VISIBLE message, and a reasoning model under tool-calling sends ~97.6% to the HIDDEN channel;
  (ii) median 88% of each game's wallclock elapses AFTER its last level clear; (iii) 45.2% of
  actions are immediate repeats; (iv) our only stagnation guard has NEVER fired in 5255 actions.

## STANDING PRIORS — things already tried and REJECTED (do not re-propose)
- Prompt-side injection / prompt A/B: 0-for-6 class. Prompt tuning is noise on this rail.
- Reverting to Qwen3.6 (cheaper turns): FORBIDDEN, it is weaker (lc<=22 in 18 runs vs our 28/35).
- `analyzer_timeout=120`: FORBIDDEN, it is a request timeout whose abort path discards work.
- JEPA branch: DEAD (3 strikes, always ERRORs on the Kaggle rerun).
- Adding wallclock: elasticity eps=0.17, and extra wallclock is not competition-legal anyway.
- ARChitect / ARC-AGI-1 grid-transduction TTT: DOES-NOT-TRANSFER (no demo pairs in -3).
- Published ARC-AGI-3 headlines are NON-COMPARABLE (e.g. MAP's "22/25" = beat-ReAct rate).

## 1b. DISCUSSIONS SWEEP
Check https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion sorted by RECENT.
Report every post NEW since 2026-08-24. For each: title, author, author's LB score+rank if
visible, one-line content summary, and a verdict ADOPT / ADAPT / IGNORE with a ONE-LINE reason.
Apply the standing priors above ruthlessly -- most posts will be IGNORE and that is fine.
Flag any post that (a) publishes a bundle/config we could byte-audit, (b) reports a mechanism
with an ablation, or (c) claims a score with a REPRODUCIBLE method.

## 1c. RESEARCH SWEEP
Search arXiv + the web for material NEW in roughly the last week on:
  - LLM agents on interactive / long-horizon benchmarks (esp. supervision loops,
    stagnation detection, replanning triggers -- directly feeds our S1 arm)
  - ARC-AGI-3 specifically
  - test-time learning / test-time training for agents
  - agentic harness architecture (memory persistence, context management across turns)
  - anything on REASONING-MODEL memory loss under tool-calling grammars (our defect (i))
Same ADOPT / ADAPT / IGNORE discipline with one-line reasons.
**PRIORITY**: we specifically need AVO (NVIDIA's agentic harness, ~100% on the public set with
Opus 5) and VISTA -- find their arXiv papers and extract, VERBATIM where possible:
  - the supervisor's exact TRIGGER CONDITIONS (what counts as stagnation, over what window)
  - the exact REDIRECT CONTENT injected when it fires
  - any ablation isolating the supervisor's contribution
That is the single highest-value item in this sweep; it is the specification our S1 arm
would otherwise have to invent. If you cannot find the papers, say so plainly.

## OUTPUT
Write `learnings/sweep_2026-08-25.md` with sections 1b and 1c, every item carrying its
verdict + reason, and a final "WHAT CHANGES OUR PLAN" section (may legitimately be "nothing").
Mark claims [V]=verified-from-source or [INF]=inferred. Quote AVO/VISTA trigger text verbatim
in a fenced block if found. Return a compact summary as your final message.
