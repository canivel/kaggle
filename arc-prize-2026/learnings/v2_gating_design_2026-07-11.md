# Phase1-v2 gating design — trajectory mining report (2026-07-11)

Analysis: `scripts/mine_trajectories.py` (stdlib, CPU-only). Data: 500 vanilla runs (`runs/tufa_example_run/benchmark.json`), 3 phase1 seeds (~73 runs), seed1 per-action events + transcripts (only seed with artifacts), `gate_report_final.md`.

## Headline discovery (changes the whole question)

**The explore trigger almost never fired in the v1 A/B.** In seed1, scripted explores ran in 2/25 games only — bp35 (1, excluded-flaky) and ft09 (4). All five win-games (cn04 +1.40, cd82 +0.81, lp85 +0.57, sc25 +3.11, tn36 +5.25) received **zero** explore injections. The v1 deltas therefore cannot be attributed to explore; they come from the always-on components (archive status line ~400×/game, animation summaries 0–40×/game) plus version noise — 15/24 games including ar25, ka59, cn04, sc25, tn36 were served under a **different game version** than the null arm. A static per-game allowlist built from these deltas would be fitting version noise.

## Q1 — Mode signatures (vanilla, first 30 actions, good = lc≥1)

Per-feature AUC, good vs bad mode, pooled over the 10 bimodal games:

| feature | good mean | bad mean | pooled AUC |
|---|---|---|---|
| action diversity (uniq id+coords) | 0.266 | 0.263 | 0.49 |
| repeated-action ratio | 0.43 | 0.41 | 0.52 |
| tokens/action | 559 | 625 | 0.47 |
| sec/action (median) | 7.1 | 11.0 | 0.43 |
| mouse fraction | 0.26 | 0.26 | 0.46 |

**No behavioral feature separates modes early.** Per-game AUCs are strong but *contradictory in direction* (repeat_ratio: g50t 0.95 vs ls20 0.16; tokens/action: ls20 0.91 vs s5i5 0.00), so any fixed detector on these features anti-generalizes.

The one game-agnostic signal is **time-to-first-level**. Pooled over 113 vanilla good runs: median 32 actions, p75 = 65, p90 = 94. Bad-mode runs never level at all. So the runtime detector is simply: **`levels_completed == 0 AND action_num ≥ 90`** → ~10% false-positive rate on eventually-good runs, ~100% true-positive on bad-mode runs. Bonus: by action 90 level-1 efficiency `(base/actions)²` is already near zero for most games (median base ≈ 30–50), so probe-action contamination at that point is nearly free.

## Q2 — Intervention timing (seed1)

- **Win-games:** zero explores; level-ups cannot have followed injections. cd82/sc25/tn36 had 24/6/20 animation summaries; cn04/lp85 had none. Animation count does not separate wins from losses (loss ft09: 36; neutral su15: 40).
- **Loss-games:** ar25 and ka59 had **zero explores and zero animations** — their −1.65/−1.03 deltas are version-diff/variance, not injection damage. ft09 (same version both arms, so the only unconfounded loss) fired 4 explores at turns 327/339/350/368 of 375 — pure desperation phase; counter showed 10 consecutive stalled turns before each, so nothing mid-progress was displaced. The seed1 ft09 run was mode-dead from the start (0 levels in 169 actions vs null median first level at action 18). The real explore cost is **action inflation**: 4×8 = 32 probe actions land in `actions_per_level` of whatever level is in progress; the RHAE penalty is quadratic — a level finished at budget b=30 with 8 probe actions added scores (30/38)² = 0.62× that level's weight (−38%).

## Q4 — Trigger threshold (tracker counter, seed1, ~10.7k analyzer turns)

Streak length when organic progress arrives (n=295 events): median 1, p75 = 2, p90 = 4, p95 = 6, **max = 9**. Streak on the turn a level-up lands (n=19): **median 0, max 0** — level-ups happen while progress is flowing, never after long stalls. Implications:

1. Threshold 10 is already above the observed organic max (9) — it interrupts organic progress with probability ≈ 0 (0/295). **Raising 10→20 is wrong**: the counter never organically exceeded 10 in any of 25 games, so 20 would disable explore entirely.
2. The trigger is *sufficient but not the binding gate*: it fired only where the state archive saturated. Keep it at 10; add the level-based gate from Q1 as the AND-condition that carries generalization.

## Q5 — RHAE weight exploitation

Triangular weights (level i weight i+1, cap 115): per-level value = min(115,(base/actions)²·100)·w/W. Across 25 games:

| move | mean RHAE gain |
|---|---|
| polish level-1 from observed median eff to 115 cap | +1.73 |
| complete one more level at 2× over budget (eff 0.25) | +1.43 |
| complete one more level at budget (eff 1.0) | **+5.73** |

Late levels dominate: for ft09 one more level at eff 1.0 = +14.3; r11l +9.5; vc33 +10.7. **Budget reallocation:** never spend turns re-optimizing a completed level or restarting for efficiency; a marginal level at even 2× over budget matches a full early-level polish, and at budget it is worth 3.3× more. Practically: suppress any "restart to redo level 1 cleanly" behavior; allow explore's action cost freely once a level is stalled ≥90 actions (its efficiency term is already forfeit); protect *early, in-budget* levels from probe inflation (the −38% case above).

## V2 rule (chosen: hybrid dynamic, option c-modified)

Static allowlist (a) **rejected**: deltas confounded by version drift and not attributable to explore. Pure Q1 detector (b) adopted as the primary gate; stricter streak threshold (c's 10→20) **rejected by Q4 data**. Fire scripted explore only when ALL hold:

1. `no_progress_turns ≥ 10` (unchanged; = p100 of organic streaks),
2. `levels_completed == 0 AND action_num ≥ 90` **or** `actions_on_current_level ≥ 90` (mode detector; probe cost ≈ free past this point),
3. no level-up within the last 20 analyzer turns (momentum guard; level-ups land at streak 0),
4. `explores_done < 3` (was 6) and budget 6 (was 8) → worst-case contamination 18 actions/run (was 48).

## Patch spec — `duck_eval/phase1/` v2

`phase1_patch.py` `Phase1Config`:

- `explore_after_turns`: keep 10 (`PHASE1_EXPLORE_AFTER_TURNS`).
- NEW `explore_min_level_actions: int = _env_int("PHASE1_EXPLORE_MIN_LEVEL_ACTIONS", 90)` — actions taken on the current level (track via level-up action indices) must reach this before explore may fire.
- NEW `explore_levelup_cooldown: int = _env_int("PHASE1_EXPLORE_LEVELUP_COOLDOWN", 20)` — `ProgressTracker` records `turns_since_levelup` (reset in `update()` when `level > _last_level`); trigger requires `turns_since_levelup ≥ 20` or no level-up yet.
- `explore_probe_budget` default 8 → **6**; `max_explores_per_game` default 6 → **3**.
- `explore_max_level`: leave 99 (condition 2 supersedes it; keep as override knob).
- Trigger conjunction in the analyze wrapper (~line 338) gains the two new checks.
- Kill switches preserved verbatim: `PHASE1_ENABLE_EXPLORE`, `PHASE1_ENABLE_ANIMATION`, `PHASE1_ENABLE_REPL_ARCHIVE`, `PHASE1_ENABLE_EVICT_HYSTERESIS`; each new gate is inert when `PHASE1_ENABLE_EXPLORE=0`.
- Animation + status line unchanged (no evidence of harm; not the tested variable).

## Pre-registered v2 A/B thresholds

`MIN_LEVEL_ACTIONS=90` (p90 of 113 vanilla time-to-first-level), `AFTER_TURNS=10` (p100 of 295 organic streaks), `COOLDOWN=20`, `MAX_EXPLORES=3`, `BUDGET=6`. Primary metric: mean paired RHAE delta vs null, same-version baselines, α=0.0125; secondary: levels_completed. No per-game parameters anywhere — private-set safe.
