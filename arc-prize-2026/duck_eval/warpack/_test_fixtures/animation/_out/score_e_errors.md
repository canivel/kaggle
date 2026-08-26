# animation-awareness v1 — score e_errors

**VERDICT: KILL** — K-A3 animation_tokens_est / total_tokens < 1% -> FAIL; K-A4 animation_errors == 0 -> FAIL

- pull: `F:\kaggle\arc-prize-2026\duck_eval\warpack\_test_fixtures\animation\e_errors`
- log: `F:\kaggle\arc-prize-2026\duck_eval\warpack\_test_fixtures\animation\e_errors\arc3-duck-animation-eval.log` (json-array)
- arm label: `duck-harness-kaggle-continuation-v1-animation-v1`
- prereg: `learnings/war_room/animation_prereg_2026-08-11.md` (SEALED) — canaries §3, metrics §4, seal arithmetic §4.1, kill rules §5
- VOID != FAIL: rebuild, and do not record a verdict in either direction.

## 0. EXTERNAL PRIOR — the feature's own author published a NULL

**Kaggle discussion 734369 'Write Up: Taaf Anim Agent'** — Jakob Bruggen (Helmut AGI, #8 @ 1.61), 2026-08-11T07:55Z (swept in `learnings/sweeps/discussion_sweep_2026-08-11.md §1.1`). Filed as a **pre-result external prior**; the prereg is NOT amended and K-A3 keeps its sealed threshold.

- **Efficacy: NULL: +1.4% mean score, p = 0.92 over 6 games x 4 passes.**
- Harm mechanism: 'Tokens are the real currency, not actions.' His animation arm went 384 -> 449 tokens/action (+17%), and 'in every single game, more tokens per action meant fewer actions'. Every run in both arms hit the 132-min wall-clock cap.
- Tool use: animation() called in 21/24 runs; 2 of 181 calls landed on an informative animation
- Why this does not falsify our arm: his flag carries all three stages; ours is stage 1 only (fixed ~45-token scalar summary, emitted only on animated actions). The retrieval tool and the proactive hint -- where his token inflation lives -- are pre-registered as explicitly OUT (prereg §2.1/§2.2). Locally measured token fraction: 0.00243.
- Consequence for this run: M2 (tokens/action, tokens/lc, wall-clock/action) is the metric that decides whether this arm can ever pay. M0 remains the PRIMARY pre-registered mechanism endpoint; a good M0 is mechanism DELIVERY, never an efficacy claim -- the best available efficacy evidence is his, and it is null.

> **M0 measures mechanism DELIVERY, not efficacy.** A good M0 may not be read as an efficacy claim: the best available efficacy evidence is his, and it is null.

## 1. Canaries (gate everything — prereg §3/§5)

| canary | status | outcome if fail | key numbers |
|---|---|---|---|
| K-A0 banner + ANIMATION_AWARE=1 stamp | **PASS** | VOID (ran VANILLA -- explicitly NOT a FAIL) | banner=yes stamp=yes PATCH FAILED=no seams=4 |
| K-A1 >=1 ANIMATION event line on >=5 distinct games | **PASS** | VOID | 568 event lines on 17 distinct games (need >= 5) |
| K-A2 nonzero invisible on >=1 of ft09/cd82/sc25/ls20 | **PASS** | VOID + audit method back under review | invisible by type-1 game: {"ft09": 9, "cd82": 33, "sc25": 30, "ls20": 29} |
| K-A3 animation_tokens_est / total_tokens < 1% | **FAIL** | KILL (arm killed, module reverted) | tokens_est=25560 / total=2084803 = 0.012260 (bound < 1%) |
| K-A4 animation_errors == 0 | **FAIL** | KILL (a perception patch that raises in the action path is not shippable) | animation_errors=7 |

Raw evidence lines each canary was decided from:

- `K-A0` — `animation v1: ACTIVE (4 seams patched)  -  per-action intermediate-frame summary from GameState.raw.frame (taaf/game.py:170 discards all but frame[-1]; zero prior consumers). Fixed scalar schema, NO raw frames, ~45 tok, emitted only on animated actions. only_invisible=OFF (default); outcome_text=ON (default); NO no-op guard (prereg sec2.2: separately gated, downstream); zero LLM calls, no locks, g`
- `K-A0` — `animation-eval: SEED=1 animation-awareness ON, NO warpack/ledger-graft/sentinel/compaction (pairs with the duck-harness-kaggle-continuation-v1 family); ANIMATION_AWARE=1; NO no-op guard`
- `K-A0` — `animation v1: graft applied from /kaggle/input/arc-war-kit (applied=True)`
- `K-A1` — `ANIMATION v=1 kind=motion game=sk48-d8078629 action=MOUSE(row=50, col=52) frames=2 unique=4 board_unchanged=0 transient_cells=97 bbox=[3, 11, 43, 60] run_actions=318 run_multi=1 run_invisible=0`
- `K-A2` — `ANIMATION CANARY v=1 version=v1 actions=6466 multi=568 invisible=101 summaries=568 errors=7 games_with_events=17 games_with_invisible=4 audit_type1_engaged=cd82,ft09,ls20,sc25 tokens_est=25560 token_fraction=`
- `K-A2` — `sc25-635fd71a action=MOUSE(row=55, col=12) frames=2 board_unchanged=1 transient_cells=289`
- `K-A3` — `ANIMATION CANARY v=1 version=v1 actions=6466 multi=568 invisible=101 summaries=568 errors=7 games_with_events=17 games_with_invisible=4 audit_type1_engaged=cd82,ft09,ls20,sc25 tokens_est=25560 token_fraction=`
- `K-A4` — `ANIMATION CANARY v=1 version=v1 actions=6466 multi=568 invisible=101 summaries=568 errors=7 games_with_events=17 games_with_invisible=4 audit_type1_engaged=cd82,ft09,ls20,sc25 tokens_est=25560 token_fraction=`

- per-game jsonl sidecars: per-game jsonl sidecars absent -- EXPECTED, see the emitter defect note; K-A1 is decided on the stdout event lines per prereg §3

## 2. M2 (DECIDING — external prior 734369) — tokens, and what they cost in moves

| run | actions | actions/game | lc | gen tokens | **tok/action** | tok/lc | wall-clock s/action |
|---|---|---|---|---|---|---|---|
| duck-harness-kaggle-continuation-v1-animation-v1 | 6466 | 258.6 | 31 | 2084803 | **322.4** | 67252 | 30.64 |
| duck-harness-kaggle-continuation-v1 | 4290 | 171.6 | 16 | 1549056 | **361.1** | 96816 | 46.17 |
| duck-harness-kaggle-continuation-v1 | 5162 | 206.5 | 10 | 1639111 | **317.5** | 163911 | 38.36 |
| duck-harness-kaggle-continuation-v1 (m=2, pooled) | 9452 | 189.0 | 26 | 3188167 | **337.3** | 122622 | 41.90 |

- **tokens/action delta: -4.41%** vs the `duck-harness-kaggle-continuation-v1` family. External reference: his stage-1+2+3 arm was **+17.0%** (384 → 449 tok/action). Ratio to his: -0.259×.
- arm / family ratios: tok/action 0.9559, tok/lc 0.5484, wall-clock/action 0.7312

**Wall-clock / actions coupling** (his causal path from tokens to lost levels):

- **the arm executed MORE OR EQUAL actions per game than the family (258.6 vs 189.0, +36.82%)**
- arm executed fewer actions than the family: **no** (6/25 games individually below the family mean)
- wall clock per game: arm 7925.3 s vs family 7921.7 s
- our scored rail projects 32,267 s against a 32,400 s cap and all 25 games end on wall clock, not on an action limit (max_actions_per_game=None) -- so his tokens -> fewer actions -> fewer levels path transfers to us
- tokens are generated tokens (uncached_input_tokens is 0 on this rail); wall clock is final_wallclock_seconds per game

## 3. M0 (PRIMARY pre-registered mechanism endpoint — DELIVERY, not efficacy)

`invisible_actions / executed_actions` = 101/6466 = **0.01562**; `multi_frame_actions / executed_actions` = 568/6466 = 0.08784.

Offline pre-build audit (`F:\kaggle\arc-prize-2026\runs\animation\frame_audit.json`, the pre-registered expectation): 17/25 games multi-frame, 401/11104 = 3.6% INVISIBLE, multi-frame 23.2%. Registered expectation: nonzero on ft09/cd82/sc25/ls20, ~0 elsewhere (prereg §4 M0).
**Expectation check: MET** (misses: none; surprises: none)

| game | audit type | exec | multi | invis | invis rate | multi rate | audit invis% (comb / probe-A) | audit multi% | expectation | status |
|---|---|---|---|---|---|---|---|---|---|---|
| ar25 | single | 374 | 0 | 0 | 0.0000 | 0.0000 | 0.0% / 0.0% | 0.0% | ~0 | MATCH |
| bp35 | type2 | 299 | 37 | 0 | 0.0000 | 0.1237 | 0.0% / 0.0% | 100.0% | ~0 | MATCH |
| cd82 | type1 | 290 | 48 | 33 | 0.1138 | 0.1655 | 3.8% / 3.6% | 18.4% | nonzero | MATCH |
| cn04 | single | 405 | 0 | 0 | 0.0000 | 0.0000 | 0.0% / 0.0% | 0.0% | ~0 | MATCH |
| dc22 | single | 239 | 0 | 0 | 0.0000 | 0.0000 | 0.0% / 0.0% | 0.0% | ~0 | MATCH |
| ft09 | type1 | 80 | 13 | 9 | 0.1125 | 0.1625 | 79.8% / 0.0% | 80.7% | nonzero | MATCH |
| g50t | type2 | 421 | 52 | 0 | 0.0000 | 0.1235 | 0.0% / 0.0% | 52.6% | ~0 | MATCH |
| ka59 | type2 | 122 | 15 | 0 | 0.0000 | 0.1230 | 0.0% / 0.0% | 0.8% | ~0 | MATCH |
| lf52 | type2 | 408 | 51 | 0 | 0.0000 | 0.1250 | 0.0% / 0.0% | 100.0% | ~0 | MATCH |
| lp85 | type2 | 114 | 14 | 0 | 0.0000 | 0.1228 | 0.0% / 0.0% | 0.3% | ~0 | MATCH |
| ls20 | type1 | 254 | 42 | 29 | 0.1142 | 0.1654 | 3.6% / 8.2% | 6.2% | nonzero | MATCH |
| m0r0 | single | 407 | 0 | 0 | 0.0000 | 0.0000 | 0.0% / 0.0% | 0.0% | ~0 | MATCH |
| r11l | type2 | 179 | 22 | 0 | 0.0000 | 0.1229 | 0.0% / 0.0% | 74.5% | ~0 | MATCH |
| re86 | single | 100 | 0 | 0 | 0.0000 | 0.0000 | 0.0% / 0.0% | 0.0% | ~0 | MATCH |
| s5i5 | single | 393 | 0 | 0 | 0.0000 | 0.0000 | 0.0% / 0.0% | 0.0% | ~0 | MATCH |
| sb26 | type2 | 438 | 54 | 0 | 0.0000 | 0.1233 | 0.0% / 0.0% | 35.9% | ~0 | MATCH |
| sc25 | type1 | 259 | 43 | 30 | 0.1158 | 0.1660 | 18.9% / 18.8% | 21.3% | nonzero | MATCH |
| sk48 | type2 | 318 | 39 | 0 | 0.0000 | 0.1226 | 0.0% / 0.0% | 45.2% | ~0 | MATCH |
| sp80 | type2 | 268 | 33 | 0 | 0.0000 | 0.1231 | 0.0% / 0.0% | 8.0% | ~0 | MATCH |
| su15 | type2 | 98 | 12 | 0 | 0.0000 | 0.1224 | 0.0% / 0.0% | 51.1% | ~0 | MATCH |
| tn36 | type2 | 359 | 44 | 0 | 0.0000 | 0.1226 | 0.0% / 0.0% | 5.4% | ~0 | MATCH |
| tr87 | single | 82 | 0 | 0 | 0.0000 | 0.0000 | 0.0% / 0.0% | 0.0% | ~0 | MATCH |
| tu93 | type2 | 208 | 26 | 0 | 0.0000 | 0.1250 | 0.0% / 0.0% | 52.2% | ~0 | MATCH |
| vc33 | type2 | 185 | 23 | 0 | 0.0000 | 0.1243 | 0.0% / 0.0% | 0.3% | ~0 | MATCH |
| wa30 | single | 166 | 0 | 0 | 0.0000 | 0.0000 | 0.0% / 0.0% | 0.0% | ~0 | MATCH |

- executed actions: canary=6466, benchmark=6466
- event lines vs canary counters consistent: yes (event lines: 568 multi / 101 invisible)

## 4. M1 (SECONDARY — DESCRIPTIVE ONLY, NOT A SCREEN)

**NOT SCREENABLE (SCREEN_PROTOCOL §1 P2)** — family `duck-harness-kaggle-continuation-v1`, m = 2.
**M1 verdict: uninformative in both directions.** prereg §4.1.6 / SCREEN_PROTOCOL §4.6: power at m=2 is far below 50%; this run is an exploratory mechanism probe, not a screen. No PASS may be reported as non-harm; no FAIL may be reported as harm. The ONLY legal M1 verdict string is 'uninformative in both directions'.

- baseline `F:\kaggle\arc-prize-2026\runs\kernel_pulls\w0_eval_s1\benchmark.json` — label `duck-harness-kaggle-continuation-v1` (matches family: yes), lc total 16
- baseline `F:\kaggle\arc-prize-2026\runs\kernel_pulls\w0_cont_eval\benchmark.json` — label `duck-harness-kaggle-continuation-v1` (matches family: yes), lc total 10
- family SS re-check from those two benchmark.json files: SS=0.0288 df=1 vs sealed 0.0288 → matches: yes

- σ̂ re-derived from the SCREEN_PROTOCOL §1 P3 pooled SS table = 0.14174 (sealed 0.14174, match yes); df = 6 (sealed 6, match yes)
- **ADVISORY** K3″ line at m=2: −C(2)·σ̂ = −2.1×0.14174 = **-0.29765** lc/game (sealed -0.2977, match yes)
- **ADVISORY** 80%-power floor at m=2: C(2)·σ̂ + 0.8416·σ̂·√(1+1/2) = **0.44375** lc/game = **11.09 levels** over 25 games (sealed 0.4437 / 11.09, match no/yes)
- ADVISORY ONLY -- C(1) and C(2) are listed in SCREEN_PROTOCOL §2 'for advisory arithmetic only'; C(2) was never measured (interpolated between m=1 2.0% and m=3 4.4% type-I). Not a gate.
- seal arithmetic all-match: **no**

- arm lc total 31 vs baseline totals [16, 10] (family mean 13.0 levels)
- paired mean Δlc = **0.72000** lc/game over 25 games (sd 0.7916, 17W/2L, sign-flip p = 0.0004 — descriptive significance only (SCREEN_PROTOCOL §3: not a gate))
- vs the ADVISORY line -0.29765: mean Δlc is above it — ADVISORY ONLY -- may not be reported as PASS or FAIL

| game | arm lc | baseline lcs | baseline mean | Δlc |
|---|---|---|---|---|
| ar25 | 2 | [1, 0] | 0.5 | 1.5 |
| bp35 | 2 | [1, 1] | 1.0 | 1.0 |
| cd82 | 1 | [0, 0] | 0.0 | 1.0 |
| cn04 | 1 | [0, 0] | 0.0 | 1.0 |
| dc22 | 2 | [0, 0] | 0.0 | 2.0 |
| ft09 | 0 | [2, 0] | 1.0 | -1.0 |
| g50t | 0 | [0, 0] | 0.0 | 0.0 |
| ka59 | 2 | [1, 0] | 0.5 | 1.5 |
| lf52 | 0 | [1, 1] | 1.0 | -1.0 |
| lp85 | 2 | [1, 1] | 1.0 | 1.0 |
| ls20 | 1 | [0, 0] | 0.0 | 1.0 |
| m0r0 | 1 | [0, 0] | 0.0 | 1.0 |
| r11l | 1 | [1, 1] | 1.0 | 0.0 |
| re86 | 2 | [1, 1] | 1.0 | 1.0 |
| s5i5 | 2 | [0, 0] | 0.0 | 2.0 |
| sb26 | 2 | [1, 1] | 1.0 | 1.0 |
| sc25 | 0 | [0, 0] | 0.0 | 0.0 |
| sk48 | 1 | [0, 0] | 0.0 | 1.0 |
| sp80 | 1 | [1, 0] | 0.5 | 0.5 |
| su15 | 1 | [1, 1] | 1.0 | 0.0 |
| tn36 | 2 | [0, 1] | 0.5 | 1.5 |
| tr87 | 1 | [0, 0] | 0.0 | 1.0 |
| tu93 | 1 | [2, 0] | 1.0 | 0.0 |
| vc33 | 2 | [2, 2] | 2.0 | 0.0 |
| wa30 | 1 | [0, 0] | 0.0 | 1.0 |

## 5. M3 (descriptive) — repeated identical no-ops on the type-1 games

Definition: action i counts as a repeated identical no-op iff board_changed is False AND action_display == action_display[i-1]; read from the vanilla harness per-game viewer events (artifacts/*_events.jsonl, type=action, 1:1 with benchmark history)

| run | games | actions | no-ops | repeated identical no-ops | rate |
|---|---|---|---|---|---|
| ARM | 4/4 | 883 | 488 | 238 | 0.2695 |
| w0_eval_s1 | 4/4 | 345 | 43 | 6 | 0.0174 |
| w0_cont_eval | 4/4 | 725 | 119 | 64 | 0.0883 |
| family pooled | — | 1070 | — | 70 | 0.0654 |

- arm / family rate ratio: 4.1200

## 6. P1 legality (prereg §4.1.1, SCREEN_PROTOCOL §1 P1)

| run | continuation-v1 banner | forbidden tokens | status |
|---|---|---|---|
| ARM e_errors | yes | none | **PASS** |
| baseline w0_eval_s1 | yes | none | **PASS** |
| baseline w0_cont_eval | yes | none | **PASS** |


_Generated by `duck_eval/warpack/animation_score.py` at 2026-08-11T08:38:07._
