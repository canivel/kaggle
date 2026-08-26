# Ledger graft v2 — patch map (R2: Hypothesis Ledger + Goal-Family Escalation)

**v2 (2026-07-16, panel R12 N6 fix):** the store is now keyed PER GAME by the
runtime-state filename stem (`<artifact_stem(game_id)>_p<pass>`), persisted to
`<runtime_dir>/ledger_<stem>.json`. v1 keyed by `state_path.parent`, which in
the LIVE harness layout (`solver.py:1219`: every game's
`<stem>_runtime_state.json` lands in ONE shared `artifacts/` dir, games run
concurrently in a ThreadPoolExecutor) silently shared a single store + a
single `ledger.json` across all 25 games — cross-game digest contamination.
Stage-1/2 rigs used per-game tmpdirs, so it went unnoticed; test rigs with a
bare `runtime_state.json` fall back to the (unique) parent-dir name.
`install()` now prints a stdout banner `ledger v2: store keying =
per-game:runtime-state-stem; flags=...` so the build log proves which dataset
version ran. Registry lock holds dict ops only (I/O outside the lock).
War-v2-eval smoke I7 covers two concurrent games in one shared dir: zero
contamination, per-game files, canary TOTAL `stores==2`.

Spec: `learnings/war_room/intervention_plan.md` R2 + test protocol stages 1–2.
Style: taaf_grafts flagged composite — `install(bm, flags)` with flags
`ledger`, `escalation`; ALL flags default OFF (no-op guarantee); `escalation`
implies `ledger`. `apply(bm)` reads `LEDGER_FLAGS` (comma list) for the hook
cell. Any install error → stock behaviour. No game-id logic anywhere.

## What gets patched (all class/module-level monkeypatches, like phase1)

| # | Target | Change |
|---|--------|--------|
| 1 | `ToolAgent.analyze` | Binds the per-game `Ledger` (registry keyed by the runtime-state filename stem = per game, mirrored to `<runtime_dir>/ledger_<stem>.json`) onto the agent before the turn; persists it after. The store lives OUTSIDE the 14-message window, so it survives eviction, GAME_OVER restarts, and level transitions. |
| 2 | `ToolAgent._update_summarized_knowledge_from_assistant` | Extraction tap: every assistant text is regex-scanned for `GOAL:` / `RESULT:` / `FACT:` prompt fields (`ledger_core.extract_goal_result`). GOAL → HYPOTHESIS(id, statement, family, status untested/executing/refuted/confirmed, evidence); RESULT refuted → refute best-matching active hypothesis (content-word overlap, newest wins); FACT → FACT record (action-effect observations, e.g. "SPACE only decrements the timer"). |
| 3 | `ToolAgent._build_user_prompt` | Appends (a) the ledger digest — refuted list (newest first, older aggregated per family) + facts + actives, hard-capped at **≤600 tokens** — re-injected EVERY turn; (b) the ledger protocol lines that define the `GOAL:`/`RESULT:`/`FACT:` reply fields; (c) when the `escalation` flag is on and armed: the ONE-SHOT 4-family enumeration prompt (execution-order/program, transfer-between-structures, merge/physics, spatial-alignment — "pick the one your refuted set least resembles"). Escalation arms when **N=3** hypotheses in the same family are fully executed and refuted since the last fire; `consume_escalation()` fires exactly once per trigger. |
| 4 | `_HarnessGameSession._execute_action` | Harness-side FACT feed (works with non-LLM analyzers too): level-completion facts, GAME_OVER facts ("this ledger persists across the restart"), and known-no-op coordinate facts (a MOUSE display re-observed with `board_changed=False` ≥2×). Saves the ledger after each action. |

## NOT changed
No solver replacement, no new tools, no message-window change, no env
stepping — the graft never executes actions, so scripted-policy action counts
are byte-identical with flags on (stage-2 non-interference gate).

## Files
- `ledger_core.py` — pure logic: `Ledger` (records, digest, escalation,
  JSON persistence), `extract_goal_result` (new prompt contract),
  `HeuristicExtractor` (legacy-transcript replay mode for stage-1).
- `ledger_patch.py` — the graft (`install` / `apply`), patch layers 1–4.
- `hook_cell.py` — cell-12 additions (dataset marker discovery, env flags,
  vanilla-on-failure policy, `LEDGER_FLAGS` rollback switch).
- `replay_test.py` — stage-1 unit/replay over the 13 recorded runs.
- `noninterference_test.py` — stage-2 scripted-policy non-interference on
  arcengine local engines (sb26/su15/lp85 still clear L2, identical counts).
- `policies/record_policies.py` — records the engine-verified scripted
  policies' action sequences for replay through the grafted harness.

## Test gates (protocol stages 1–2)
Stage 1 (replay, no LLM): sb26 seed1 ledger accumulates ≥20 refuted
ordering-variants and escalation would have fired by action ~60; su15's two
self-disproved goals reach `refuted` with the agent's own arithmetic as
evidence; the `SPACE=timer-only` FACT persists past message 14 (and past the
action-140 GAME_OVER); every digest ≤600 tokens across all 13 runs.
Stage 2 (non-interference): sb26/su15/lp85 scripted policies replayed through
the grafted `_HarnessGameSession` on arcengine 0.9.x still hit
`levels_completed=2` with action counts identical to the ungrafted baseline,
and the ledger contains the correct level-completion action-effect FACTs.
