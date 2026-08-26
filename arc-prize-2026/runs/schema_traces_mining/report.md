# Schema traces mining — vs our latent-state audit

**Date:** 2026-07-22. **Task:** ADOPT item from `learnings/war_room/research_2026-07-22.md`.
**Dataset:** `schema-harness/arc-agi-3-schema-traces` (HuggingFace, 737 MB) -> `kaggle-data/schema_traces/`.
**Integrity:** their bundled `score_trajectories.py` PASSES on our copy — all 50 `events.jsonl` logs
contiguous, scores reproduce exactly (claude collection 25/25 wins, 183/183 levels, mean RHAE **98.98%**;
gpt collection 24/25, 182/183, 95.35%; only loss = ka59 GPT run STOPPED at 6/7, 65.34%).
**Engine versions match ours exactly** — all 14 audited-game IDs (`wa30-ee6fef47`, `sc25-635fd71a`,
`cd82-fb555c5d`, `m0r0-492f87ba`, ...) are identical to `runs/latent_state_audit/report.md` rows, so their
trajectories are directly comparable evidence for our aliased-game class. Zero cloud spend; all local CPU.
Mining script: `runs/schema_traces_mining/mine.py`; raw stats: `stats.json`.

## Event/artifact anatomy (what a trajectory contains)

Per game: `run.json` (meta), `events.jsonl` (every LLM turn, tool call, committed plan, env action with
full 64x64 grid, and — crucially — `model_mispredicted` events), per-level `snapshots/cleared_level_N.py`
(the certified world model at each level clear), `notes.md`, and the final `world_model_v5.py`.
Harness tools: `commit_actions` (plan), `run_python`, `write_file`/`edit_file` (recompile+install model),
`run_backtest` (**certification: replay model vs full interaction history**), `run_bfs` (search in-sim),
`read_history`, `run_shell`. Runtime contract: when a committed plan's next step mispredicts, the harness
emits `model_mispredicted`, **drops the rest of the plan**, and tells the agent to `run_backtest` and fix
the model before planning again. `max_actions` = 3000 per game.

## Per-game table (claude collection = the 98.98% headline)

cls = our audit verdict (A=ALIASED-RESOLVABLE, U=ALIASED-UNRESOLVED, C=CLEAN). probe1 = actions from
1-step plans; p10+ = actions from plans >=10 steps (BFS-certified execution). mp = runtime certification
failures (`model_mispredicted`); mp@0 = at step 0 of a plan; bt(F) = `run_backtest` calls (with >=1
mismatch); wmW = world-model file writes/edits (revisions); ctr% = share of mp events whose grid diff is
confined to the counter rows (row 0/63) — a lower bound on hidden-phase-caused divergence.

| game | cls | model kept | lv | RHAE | actions | turns | probe1 | p10+ | mp | mp@0 | bt(F) | wmW | resets | hrs | ctr% |
|------|-----|-----------|----|------|--------:|------:|-------:|-----:|---:|-----:|------:|----:|-------:|----:|-----:|
| ar25 | C | opus-4-8 | 8 | 100.00 | 269 | 33 | 12 | 242 | 20 | 0 | 41(12) | 26 | 0 | 3.4 | - |
| bp35 | C | fable-5 | 9 | 93.51 | 566 | 146 | 31 | 275 | 110 | 0 | 94(26) | 125 | 10 | 13.8 | - |
| cd82 | A | opus-4-8 | 6 | 100.00 | 121 | 121 | 121 | 0 | 0 | 0 | 0(0) | 0 | 3 | 2.0 | n/a |
| cn04 | A | opus-4-8 | 6 | 100.00 | 479 | 62 | 19 | 395 | 42 | 0 | 62(15) | 67 | 3 | 4.6 | 17 |
| dc22 | A | fable-5 | 6 | 98.70 | 1205 | 218 | 51 | 861 | 140 | 1 | 130(52) | 223 | 3 | 8.0 | 22 |
| ft09 | C | fable-5 | 6 | 100.00 | 78 | 15 | 0 | 57 | 9 | 1 | 17(2) | 29 | 0 | 0.8 | - |
| g50t | U | opus-4-8 | 7 | 100.00 | 544 | 82 | 26 | 429 | 51 | 0 | 29(28) | 79 | 17 | 9.1 | 20 |
| ka59 | A | opus-4-8 | 7 | 100.00 | 431 | 104 | 20 | 335 | 88 | 0 | 55(19) | 61 | 3 | 4.1 | 16 |
| lf52 | C | fable-5 | 10 | 100.00 | 1030 | 250 | 20 | 794 | 212 | 1 | 54(38) | 62 | 1 | 9.9 | - |
| lp85 | C | opus-4-8 | 8 | 100.00 | 134 | 68 | 55 | 59 | 54 | 0 | 29(5) | 40 | 1 | 2.6 | - |
| ls20 | C | opus-4-8 | 7 | 100.00 | 642 | 115 | 22 | 500 | 75 | 0 | 100(26) | 220 | 7 | 13.4 | - |
| m0r0 | U | opus-4-8 | 6 | 100.00 | 221 | 26 | 4 | 193 | 19 | 0 | 38(9) | 37 | 0 | 2.5 | 37 |
| r11l | C | opus-4-8 | 6 | 100.00 | 83 | 22 | 3 | 57 | 15 | 0 | 32(11) | 51 | 0 | 5.6 | - |
| re86 | A | opus-4-8 | 8 | 100.00 | 615 | 59 | 7 | 562 | 45 | 0 | 84(22) | 115 | 2 | 4.4 | 20 |
| s5i5 | A | opus-4-8 | 8 | 89.87 | 643 | 376 | 225 | 209 | 328 | 1 | 39(26) | 72 | 34 | 32.0 | 16 |
| sb26 | A | fable-5 | 8 | 98.63 | 135 | 12 | 3 | 126 | 3 | 0 | 10(2) | 13 | 0 | 0.6 | 0 |
| sc25 | A | fable-5 | 6 | 100.00 | 334 | 150 | 11 | 117 | 135 | 0 | 75(37) | 126 | 5 | 5.0 | 0* |
| sk48 | U | fable-5 | 8 | 100.00 | 443 | 34 | 1 | 406 | 24 | 1 | 38(9) | 74 | 1 | 3.7 | 0* |
| sp80 | C | fable-5 | 6 | 100.00 | 283 | 42 | 10 | 248 | 30 | 0 | 49(15) | 54 | 1 | 5.3 | - |
| su15 | C | fable-5 | 9 | 100.00 | 158 | 63 | 22 | 51 | 45 | 0 | 80(60) | 106 | 4 | 8.2 | - |
| tn36 | C | fable-5 | 7 | 94.74 | 210 | 39 | 10 | 126 | 23 | 0 | 28(5) | 64 | 0 | 2.8 | - |
| tr87 | A | opus-4-8 | 6 | 100.00 | 138 | 38 | 23 | 99 | 20 | 0 | 23(19) | 27 | 0 | 1.2 | 10 |
| tu93 | C | opus-4-8 | 9 | 100.00 | 243 | 96 | 12 | 148 | 77 | 0 | 57(40) | 87 | 7 | 11.6 | - |
| vc33 | A | fable-5 | 7 | 99.10 | 342 | 55 | 10 | 279 | 35 | 0 | 50(8) | 80 | 1 | 2.9 | 69 |
| wa30 | A | opus-4-8 | 9 | 100.00 | 956 | 318 | 32 | 567 | 262 | 0 | 90(78) | 144 | 7 | 12.2 | 44 |

*sc25/sk48 render their counters inside the playfield (cost bar / divider row), not rows 0/63, so ctr% under-counts there.

Totals (claude, 25 games): **10,303 actions** (vs 17,135 human-baseline actions — **0.60x human**),
2,544 LLM turns, 1,862 runtime mispredicts (18% of actions), 1,304 backtests (564 with mismatches),
1,982 world-model revisions, ~170 wall-hours (mean 6.8 h/game, max 32 h on s5i5).
Class aggregates: ALIASED-11 mean 491 actions, mispredict/action 0.203; UNRESOLVED-3 mean 403, 0.078;
CLEAN-11 mean 336, 0.181 — aliasing does NOT make games disproportionately expensive for their method.

## (a) How Schema handles our hidden-phase games

**Headline: every one of our 14 ALIASED/UNRESOLVED games was fully cleared by the Claude collection
(13 at RHAE >= 98.6; s5i5 at 89.9), and in 13/14 the certified world model carries an EXPLICIT latent
action-counter in its model state.** The mechanism our audit inferred statistically (parity/mod3/4/5
phase of actions-since-RESET) is, in their symbolic models, a first-class variable `n` with a per-level
**affine-modular law** mapping it to an on-screen quantized "budget/move bar" — the bar is a floor-divided
projection of `n` (e.g. `filled = (mult*n + off)//D`), so the sub-tick residue `n mod D` is exactly our
hidden phase. Three implementation patterns, all in released `world_model_v5.py` code:

1. **Explicit counter + online law fitting (wa30, our mod4).** State dict `{'n':0,'off':off0,...}`;
   per-level rates lvl0=1/3, lvl2=2/3, lvl4=1/2, lvl5=6/7...; `_fit_bar()` brute-forces `(D, mult, off0)`
   over observed `(n, filled)` pairs, needs **>=3 observations**, prefers the slowest fitting rate; the
   `off` term self-corrects (`off += actual - predicted`) so an irregular skip costs **exactly one
   mispredict, then re-syncs**. Their notes.md warns: "A WRONG RATE CORRUPTS PLANNING, NOT JUST
   PREDICTION... PIN THE RATE OVER >=8 FRAMES (through at least one SKIP)". So revision cost ~= 3-8
   counter observations per level, +1 action per residual anomaly.
2. **Inverse-observation resync every step (ka59, our parity).** `predict()` FIRST reads the real bar
   from the observed grid (`_barcount(grid)`) and inverts `_bar_for(n,N)` to recover/advance latent `n`,
   simultaneously pruning the per-level budget hypothesis set `Ns`. The latent state is re-derived from
   the observation on every prediction — belief never free-runs.
3. **Self-healing phase state (sk48, our UNRESOLVED).** `init_state` = `{'acts':0, 'synced':False,
   'phase':-1, 'last_bar':0}`, comment: "last_bar **resyncs from each REAL input grid, so anomalies
   self-heal after one mispredict**". Ticking is action-class-dependent ("clicks AND a7 are free — no
   budget tick") — precisely the no-op-advances-phase trap that broke our N5 prune_trace.

Other aliased games: s5i5 has per-level adversarial counters with laws as gnarly as "+3 per 7 clicks
(2,2,3 gap pattern)" and "+1 per 3 clicks; stall (gap 4) each time t4 reaches k with k%8==5" — each
annotated "exact vs history" (certified by replay); sc25 models "counter c; bar od=2*floor(c/2);
od==64 -> DEATH"; tr87 "row 63 = budget bar: floor(n_actions/2)"; g50t threads `n_actions` through a
state dict with per-level tick rates (/2 or /4); vc33's notes: "Timer row0: per-level (a*n+b)//c
click-indexed; **refit by brute force when broken**"; dc22's agent even wrote `fit_ticks.py` that parses
its own event history to fit the tick law; cn04 treats its bar as a per-level budget gauge with interval
narrowing ("just refit whenever a new level's bar disagrees"). **The counterexample is cd82**: solved at
RHAE 100 with NO world model at all — 121 turns, one action per turn, zero backtests, pure note-driven
incremental play (its notes still hand-track the tick pattern: "counter ticks on EFFECTIVE actions").
So the harness does not force model-building; on small games direct careful play is competitive.

**Do certification failures show up on these games, and how many observations does revision take?**
Yes, massively, and they are cheap: on the aliased-11, 20.3% of all actions triggered a runtime
`model_mispredicted` (vs 18.1% on clean — aliasing is NOT an outlier failure mode for them, it is the
same revise-loop), and 10-69% of those divergences are confined to the counter rows (wa30 44%, vc33 69%,
m0r0 37%, dc22 22%). Each mispredict costs one action (plan truncated, replan from the real frame);
`run_backtest` against full history then localizes the law error; the fix lands in 1-3 file edits.
**Critically, mp@0 ~= 0 (5 events in 10,303 actions): a freshly committed plan essentially never diverges
on its first step**, because plans are always re-rooted in the just-observed real frame and latent state
is recovered by replay/inverse-reading — the exact failure our EWM line dies on (step-0 aborts at rates
of 0.5-0.9 on lp85/sb26/s5i5) is engineered away.

## (b) Turn/action spend: what ~99% RHAE costs

- **Interaction budget:** 78-1,205 actions per game (median 283, mean 412), under a 3,000-action cap.
  Total 10,303 actions = **0.60x the human baseline** — RHAE >=100 is reached by being MORE action-
  efficient than humans on most levels (the 115% per-level cap makes over-efficiency bankable).
- **Split:** only 7.3% of actions come from 1-step probe plans; 23.5% from short 2-9-step plans;
  **69.3% from >=10-step plans executed inside a certified model** (BFS/A* in-sim, zero env cost per
  simulated state). Observation is mostly free-ridden: every executed action is also a certification
  sample, so "observing" and "executing" are not separate phases after the first few turns.
- **The real budget is turns/wall-clock, not actions:** 2,544 frontier-LLM turns and ~170 wall-hours
  (0.6-32 h per game) for 25 games, plus 1,304 backtests and 1,982 model edits. Per action that's ~0.25
  LLM turns; per game a mean of 102 turns.
- **Vs our regime:** our ~150-action sentinel budget would cover their cheapest quartile (ft09 78,
  r11l 83, cd82 121, lp85 134, sb26 135, su15 158) but is 3-8x short for the games that need model
  revision under aliasing (wa30 956, dc22 1205, s5i5 643, re86 615). Their mispredict tax alone (18% of
  actions) would consume ~27 actions of a 150-action budget. Uncapped eval is the right frame for this
  method; a hard 150-action cap forces cd82-style direct play, not EWM+BFS.

## (c) Fallback/escalation pattern

`run.json`/dir names record only the KEPT run (model+effort), confirming the release's stated rule
(first-line run, rerun games scoring <80 with the stronger config, keep per-game max):

- **Claude collection:** 14 games kept from first-line **claude-opus-4-8**; **11 games escalated to
  claude-fable-5**: bp35, dc22, ft09, lf52, sb26, sc25, sk48, sp80, su15, tn36, vc33.
- **GPT collection:** 16 kept from **sol-xhigh**; **9 escalated to sol-max**: bp35, dc22, ka59, lf52,
  sc25, sk48, sp80, su15, tn36.
- **8 games needed escalation under BOTH stacks** (bp35, dc22, lf52, sc25, sk48, sp80, su15, tn36) —
  a stable "hard core" that is NOT our aliased class (only dc22, sc25 + unresolved sk48 are in it; our
  worst aliased games wa30/s5i5/m0r0/g50t were all closed first-line by Opus). Latent state is not what
  drives their escalations; long-horizon level structure (bp35 9 levels/217-action level, lf52 10 levels)
  is. Even escalation does not guarantee closure: ka59 GPT sol-max still STOPPED at 6/7 (65.34), the
  release's only non-win. The failed first runs themselves are not in the release (only the kept max),
  so per-game deltas are not recoverable — but the cheap-first/strong-on-<bar template for A17
  wall-closer economics is confirmed as their actual operating rule, with ~44% (11/25) escalation
  frequency on the Claude side.

## Implication for EWM contract v1.1 and R17

Schema's traces settle the resync-before-abort question with running code on our exact engine versions:
their contract makes the world model **stateful** (`init_state(entry_grid)` at RESET, state threaded
through every `predict()` — full replay from RESET, which independently confirms our audit's
FULL-REPLAY-ONLY banking rule), makes the hidden phase an **explicit variable with a brute-force-fitted
affine-modular law** (3-8 observations to pin, our audit's parity/mod3/4/5 resolvers are exactly the
right hypothesis class and we already know which game needs which), and treats every divergence as a
**one-action-cost resync signal** (drop plan tail, re-root in the real frame, inverse-read the counter
from the observation where possible, backtest against full history, edit the law) — never a step-0
abort. Contract v1.1 should therefore specify: (1) sim state = frame + explicit phase register
initialized at RESET and advanced per action-class (no-ops tick, clicks may not — learn the tick set per
game); (2) on mispredict, truncate-and-replan with phase re-derived by replaying the action history
through the current step function (certification-as-resync), with EnvProbe-style probe spend targeted at
the phase variable; (3) abort only after certification still fails post-revision (Schema's own m0r0/g50t
wins show our UNRESOLVED class is resolvable with a better hypothesis language, so demote abort-and-
fallback from default to last resort on those three). For tonight's R17 panel: the wall-closer relevance
is the escalation economics (11/25 games re-run, max kept — same shape as our 27B->72B plan) and the
budget reality check — Schema's 99% costs 0.6x human actions but ~7 LLM-hours/game, so under our
zero-budget offline runtime the liftable piece is the CONTRACT (stateful sim + counter law + resync
ordering), not the harness; and any sub-200-action sentinel budget structurally forbids the revise-loop
that produced these numbers, which should be stated when the sentinel budget is next litigated.
