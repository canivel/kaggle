# Phase1 failure analysis — why "-0.5 RHAE" (2026-07-13)

Data: `runs/null10` (10 seeds, 250 runs), `runs/phase1_ab` (v1, 3 seeds), `runs/phase1_v2` (3 seeds). All arms rescored offline with the validated Tufa formula against null10 baselines (reproduces gate_report_FINAL exactly: v2 mean Δ = -0.542).

## Correction to the premise

**v1 does NOT cost 0.5 vs the clean null.** Rescored against null10: **v1 = +0.13** (+0.42 on the 11 version-stable games). Only v2 shows -0.54. The "-0.5 in both arms" framing was an artifact of comparing v1's tufa-scale numbers to null10.

## Where the v2 -0.54 lives

Net sum Δ = -12.9 over 24 games; the worst 3 sum to **-13.1**. The other 21 games net to ~0.

| game | Δ v2 | Δ v1 | bootstrap p (v2 / v1)¹ | decomposition (v2) |
|---|---|---|---|---|
| ft09 | -7.34 | -0.68 | 0.03 / 0.56 | level-loss -5.4 (lc 0,2,0 vs null 30% zero-rate; 2-of-3 zeros has binomial p=0.22) |
| ar25 | -3.03 | -3.16 | **0.009 / 0.008 — replicated** | v2: level-loss; v1: L1 took 172 acts vs null median 26 |
| vc33 | -2.70 | -2.35 | 0.14 / 0.19 (same direction both arms) | half level-loss, half efficiency |
| tn36 | -1.64 | +5.28² | 0.16 / — | efficiency: L1 78 acts vs null 12 |
| re86 | -1.53 | -0.01 | 0.04 / 0.51 | not replicated |
| sb26 | -0.96 | 0.00 | **0.000** / 1.0 | L1 23 acts vs 11 (null lc=1 in 10/10) |
| su15 | -0.58 | -0.21 | **0.000 / 0.000 — replicated** | L1 30 acts vs 20; 40 animation summaries (max of all games) |

¹ p = P(3 random null seeds score this low); overall v2 Δ sits at the 1st percentile of null-vs-null lottery (sd 0.25), so the arm-level loss is real, but **ft09 alone is 57% of it and is mostly bimodal-mode lottery** (v2 excluding ft09: Δ = -0.25, p = 0.075). ² v1 tn36 was a different game version — noise.

Decomposition of the v2 mean: **-0.33 from completing fewer levels, -0.19 from more actions per completed level** (quadratic RHAE penalty). L1 action inflation on sb26/su15/tn36/ar25 shows unchanged action mix (still 90-99% ACTION6) and more wallclock to first level (ar25 1861→4879s, tn36 1806→4798s) — slower *decisions*, not probe contamination (v2 explore fired ≤3×6 actions and only past 90 stalled actions; sb26/tn36 inflate before that gate opens).

## Hypothesis ranking

| rank | hypothesis | verdict | evidence |
|---|---|---|---|
| 1 | **(a) context pollution distracts the LLM** | **CONFIRMED (moderate-strong)** | Generated tokens/action: null 435 → v1 633 (+46%) → v2 543 (+25%) — model reasons about the injected machinery. Replicated losers are always-on cases: ar25 (p<0.01 both arms) had **zero** explores and **zero** animations in v1 seed1 → damage must come from the ~400×/game REPL-archive status line + `explore_archive` tool-description/global. su15 (p≈0 both arms) is the animation-heaviest game (40 summaries) → animation summaries implicated there. Effect = same clicks, more of them, slower level clears. |
| 2 | seed lottery on bimodal high-weight games | **CONFIRMED (dominant in magnitude)** | ft09 (-7.34, 57% of loss) is a 30%-zero-rate coin flipped 3 times; not replicated in v1 (-0.68). re86/tn36 also unreplicated. ~Half the headline -0.54 is variance from n=3. |
| 3 | (c) eviction hysteresis drops critical history | UNTESTED | No context-length telemetry in either arm; cannot separate from (a). Plausible contributor to ar25 (longest inflation). |
| 4 | (b) token-budget displacement | REFUTED | +25% tokens/action yet actions/game unchanged (null 140 → v2 144) in the fixed ~7800s/game budget; token budget was not binding. |
| 5 | (d) archive line breaks prompt-cache → fewer turns | REFUTED | Treated arms did MORE work per identical budget: actions/game 140→144 (v1: 180), gen tokens/game 61k→78k (v1: 114k), sec/action 77→70. `uncached_input_tokens` is 0 in all arms (dead field), but turn throughput clearly did not drop. |

## Culprits

1. **Always-on REPL-archive injection (status line + REPL global + tool-doc)** — the only component present in both replicated losers; kill-switch it (`PHASE1_ENABLE_REPL_ARCHIVE=0`) in the next A/B.
2. **Animation summaries** on animation-heavy games (su15). Gate or cap them.
3. **n=3 seeds on bimodal games** — ft09 alone can swing ±2.5; any future gate needs ≥5 seeds or ft09-aware stratification. True systematic cost of phase1-v2 is ≈ -0.2 to -0.3, not -0.54.
