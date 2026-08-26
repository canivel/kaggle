# Attempt scheduler (R1) — patch notes

**Version:** `v1` (`scheduler_patch.VERSION`)
**Spec:** `learnings/path_forward_v2_2026-07-13.md` §R1 (survived panel rounds 5+6; resolves RL-M1/LA-M4). EV +0.10 ± 0.05 local.
**Pattern:** minimal monkeypatch in the phase1 style (`duck_eval/phase1/phase1_patch.py`), applied from the duckfork notebook customization-hook cell (`hook_cell.py`) after `bm` is unpickled.

## Semantics

| Rule | Implementation |
|---|---|
| **Trigger** | Within a game, if `levels_completed == 0` when the per-attempt action counter reaches `SCHED_RESTART_AT` (=90, null10 p90 of time-to-first-level) → execute `RESET` (fresh episode; `ONLY_RESET_LEVELS=true` restarts the current level, which at lc=0 is a fresh episode). Checked post-action, so it fires on exactly the 90th action of the attempt. |
| **Per-attempt counter** | `session.action_count − attempt_start`; `attempt_start` moves forward **only** on a scheduler restart. Harness auto-resets after `GAME_OVER` do **not** reset it (a game-over-thrashing attempt still counts toward the trigger) and do not count against the cap. |
| **Cap** | `SCHED_MAX_RESTARTS` (=2) restarts per game (3 attempts), counted by a cumulative counter that **never resets** — no thrash loop is possible. |
| **Park** | At the trigger point with the cap already spent (still lc=0), the session is parked: `should_stop()` returns `True`, the play loop exits, the game finishes and stops consuming analyzer turns / vLLM throughput. Precedence: **park dominates restart**. Dead games are bounded at 272 actions (90 + RESET + 90 + RESET + 90) vs. today's grind-to-wall. |
| **Progressing games** | If `game_run.levels_completed ≥ 1` (monotonic), the scheduler never fires — no restart, no park. |
| **Thresholds** | Game-agnostic null10 percentiles. Zero game-ID-keyed logic; private-set-safe. |

## What gets patched (only the game-session loop level)

1. `inference.framework.solver._HarnessGameSession._execute_action` — post-action hook: per-attempt counting + restart/park decision. The injected `RESET` goes through the original `_execute_action`, so it lands in the ordinary action history exactly like the harness's existing auto-reset after `GAME_OVER`.
2. `inference.framework.solver._HarnessGameSession.should_stop` — parked sessions report `True`.

## NOT patched — no context injection whatsoever

- `ToolAgent.analyze`, `_build_user_prompt`, `_compact_action_result`, `_trim_messages_for_context` — untouched.
- Python tool sandbox, tool descriptions, prompts, message history — untouched.
- The model sees a restart only as a `RESET` row in its normal action history plus the fresh frame, identical in kind to the existing game-over auto-reset path.

## Config (env vars, read at `SchedulerConfig` construction)

| Var | Default | Meaning |
|---|---|---|
| `SCHED_ENABLE` | `1` | Kill switch. `0` → `apply()` installs nothing; vanilla duck. |
| `SCHED_RESTART_AT` | `90` | Per-attempt no-level trigger (actions since episode start). |
| `SCHED_MAX_RESTARTS` | `2` | Cumulative restart cap per game; park after. |
| `SCHED_STRICT` | unset | Hook-cell only: `1` → re-raise patch failures instead of falling back to vanilla. |

## Interop

- Composable with `phase1_patch` (both wrap `_execute_action`; whichever applies second wraps the other). The R1 arm ships **scheduler-only** — explore/context-injection stays closed per §R3 (ME-M5).
- Idempotent `apply()`; failure policy in the hook cell falls back to vanilla duck (score = baseline, never 0).

## Deviations / notes

- The plan text says "bounded at ≤270 actions"; the exact bound is 272 (the two RESET actions themselves occupy history rows). Same compute class.
- "A parked game resumes only if all non-parked games are finished": with per-game wall clocks and a 28-way concurrent pool, parking = finishing the run early is the faithful minimal implementation ("stop spending on that game"); freed throughput automatically flows to still-running games via the shared vLLM server.
- RESET actions increment `actions_per_level[level_0]`; that column only affects scoring for **completed** levels, and a level-1 clear after a restart pays for its own restart actions in the score formula — accounted for in the R1 EV derivation (flip-game bimodality).

## Smoke test

```
uv run python duck_eval/scheduler/smoke_test.py
```

Covers: (a) restart fires exactly at 90 actions with 0 levels; (b) cap 2 then park at 272 actions, post-park `step_env` refuses; (c) no restart when `lc ≥ 1` (from start and after a mid-run level-up); (d) kill switch inert (no patches installed, behaviorally verified); (e) real taaf-bundle module patching + end-to-end `_HarnessGameSession.play()` against a real local engine (`kaggle-data/environment_files`, ls20) with a scripted no-LLM analyzer.

Mandatory before any Kaggle push: `scripts/preflight.py` + this smoke test green.
