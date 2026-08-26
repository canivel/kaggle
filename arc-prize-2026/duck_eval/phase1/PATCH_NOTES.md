# Phase-1 Exploration Substrate — Patch Notes

**Version:** `v2` (`phase1_patch.VERSION`)

## Changelog

### v2 (2026-07-11) — gated explore trigger
Spec: `learnings/v2_gating_design_2026-07-11.md` ("Patch spec"). Trajectory mining showed
the v1 explore trigger almost never fired (2/25 games in seed1) and the v1 A/B deltas
were version-noise, so v2 replaces the bare streak trigger with a conjunction that
carries generalization. A scripted explore now fires only when ALL hold:

1. `no_progress_turns >= 10` (unchanged — p100 of 295 organic streaks; raising it
   would disable explore entirely),
2. `levels_completed == 0 AND action_num >= 90` **or** `actions_on_current_level >= 90`
   (mode detector; p90 of vanilla time-to-first-level. Bad-mode runs never level; by
   action 90 the current level's RHAE efficiency term is already near-forfeit, so probe
   contamination is ~free),
3. no level-up within the last 20 analyzer turns (momentum guard; level-ups land at
   streak 0, never after stalls),
4. `explores_done < 3` (was 6) with probe budget 6 (was 8) → worst-case contamination
   18 actions/run (was 48).

Implementation: `Phase1Config` gains `explore_min_level_actions` (90,
`PHASE1_EXPLORE_MIN_LEVEL_ACTIONS`) and `explore_levelup_cooldown` (20,
`PHASE1_EXPLORE_LEVELUP_COOLDOWN`); `explore_probe_budget` default 8→6;
`max_explores_per_game` default 6→3. `ProgressTracker` (phase1_core) now tracks
`turns_since_levelup`, `levelups`, and `actions_on_current_level` via level-up action
indices (the first observed nonzero level is the baseline, not a level-up). The trigger
conjunction lives in `explore_gate_open()` (pure, smoke-tested directly).
`explore_max_level` left at 99 (condition 2 supersedes it; kept as override knob).
All v1 kill switches preserved verbatim; every new gate is inert when
`PHASE1_ENABLE_EXPLORE=0`. Animation summaries, archive status line, and eviction
hysteresis unchanged. `hook_cell.py`/`run_ab.py` env `setdefault`s updated to the v2
pre-registered arm (they would otherwise pin the old 8/6 values over the new defaults).

### v1 (2026-07-08) — initial substrate
Everything below this line: dedup archive, scripted explore on a bare 10-turn
no-progress streak, animation summaries, REPL archive, eviction hysteresis.

**Target:** Tufa duck harness (taaf bundle, `duck_eval/taaf_bundle/src/`), applied at
runtime via the duckfork notebook's customization-hook cell (cell 12), which runs after
`bm` (the pickled `Benchmark` with `bm.solver = HarnessSolver`) is loaded and the bundled
sources are on `sys.path`. All patches are **class/module-level monkeypatches** — the
pickled solver instance picks them up because Python resolves methods at call time.
No taaf-bundle file is edited on disk.

**Spec provenance:** `learnings/tufa_writeup_review_2026-07-08.md` ("Concrete adjustments
to Phase-1 build" 3–6) + `learnings/winning_solution_FINAL.md` (Phase-1 section).

## Files in this module

| File | Role |
|---|---|
| `phase1_core.py` | Pure-stdlib CPU core: signature, archive, frontier, animation summarizer, scripted explore, summary renderer. No harness imports — unit-testable without GPU. |
| `phase1_patch.py` | The monkeypatch layer. `apply(bm)` installs everything. Env-var config, per-feature kill switches for ablations. |
| `hook_cell.py` | Verbatim source for notebook cell 12. Locates the module (Kaggle dataset via `phase1_patch.py` marker, or local `./phase1`), applies patches, falls back to vanilla duck on failure unless `PHASE1_STRICT=1`. |
| `smoke_test.py` | CPU smoke suite (run with `uv run python duck_eval/phase1/smoke_test.py`). |

## Patch map — exactly what is modified

### 1. `inference/agent/tool_agent.py :: ToolAgent.analyze` (wrapped)
Pre-hook before the original body:
- Creates/loads per-game `Phase1State` (keyed by the runtime-state dir, i.e. one per
  game session, matching `_ensure_session` semantics).
- Reads `tool_runtime_state.json` (`load_runtime_state`), observes the current frame
  into the dedup archive, and updates the no-progress counter — **one count per
  `analysis_step`** (retries and 60-s control-yields re-enter `analyze` with the same
  step and are not double-counted).
- **Scripted explore trigger (v2 conjunction, `explore_gate_open`):** when
  `turns_without_progress >= N` (default N=10, `PHASE1_EXPLORE_AFTER_TURNS`) AND the
  run is stalled on its current level (`levels_completed == 0 AND action_num >= 90`,
  or `actions_on_current_level >= 90`; `PHASE1_EXPLORE_MIN_LEVEL_ACTIONS`) AND no
  level-up landed within the last 20 analyzer turns (`PHASE1_EXPLORE_LEVELUP_COOLDOWN`),
  the harness itself executes up to `PHASE1_EXPLORE_BUDGET` (6) real actions through
  the session's `step_env` callback — **never as an LLM-electable tool** (Tufa's
  tools-hinder-creativity finding). Probe order: untried simple actions at the current
  state signature, then MOUSE clicks on salient-component centroids, then already-tried
  actions; replans when the state signature changes. Skips when fewer than
  `PHASE1_MIN_TIME_REMAINING` (300 s) remain or after `PHASE1_MAX_EXPLORES` (3)
  explores per game.
- Stores a curated report (≤ `PHASE1_SUMMARY_CHAR_CAP` = 1500 chars ≈ 500 tokens at
  duck's ~3 chars/token estimator) for injection into the **next user message**, and
  updates `agent._last_step_summary` via the agent's own `_summarize_step_sequence` so
  the prompt's "last executed sequence" narration stays truthful.
- Sets a thread-local pointer to `Phase1State` for the duration of the original
  `analyze` call (used by patch 5). One game = one worker thread, so thread-local is
  the correct scope at `concurrency=16+`.
- The whole pre-hook is wrapped in try/except: a Phase-1 failure can never break a turn.

### 2. `ToolAgent._build_user_prompt` (wrapped)
Appends to the original prompt: (a) the pending exploration report, once; (b) a one-line
archive status (`X unique states, Y turns since last new state`). Both land at the tail
of the newest user message, so the cached message prefix is untouched.

### 3. `ToolAgent._compact_action_result` (wrapped)
Whitelists the new `animation_summary` field through to the model-visible compact action
result (the original drops unknown keys).

### 4. `ToolAgent._trim_messages_for_context` (wrapped) — prefix-cache hysteresis
Original behavior trims **one oldest block per turn** once the 32k analyzer budget is
hit — invalidating the vLLM prefix cache **every turn** at steady state. Patched: when a
trim is needed, evict down to `PHASE1_EVICT_LOW_FRAC` (0.5) of the budget in one batch
(the "64k server / evict-to-32k" idiom at message granularity). Between evictions the
message list is append-only ⇒ long prefix-cache-hit streaks. Under budget, behavior is
byte-identical to the original. The vLLM server already launches with
`--enable-prefix-caching` (verified by the hook cell); the system prompt is stable.

### 5. `tool_agent.run_sandboxed_python` (module-level rebind)
The REPL resets per toolcall, so the dedup archive is delivered the same way duck
delivers `history`/`transitions`: as a **pre-loaded REPL variable**. The wrapper injects
`explore_archive` (JSON-safe snapshot: unique_states, current_state_untried, frontier
scored `novelty/(1+return_cost)`, last_explore_probes) into the sandbox's initial state
and — by wrapping `action_handler` — into every post-`action()` state refresh.

### 6. `tool_agent._PYTHON_TOOL_DESCRIPTION` (string append)
Documents `explore_archive` in the `python` tool description (read at call time by
`ToolAgent._tools`). Stable across turns ⇒ no prefix-cache impact.

### 7. `python_tool_sandbox._SANDBOX_BOOTSTRAP` (string splice)
Adds one line to the sandbox's `_refresh_state`, directly after the
`runtime_globals["last_action_result"] = action_result` anchor:
`runtime_globals["explore_archive"] = state_payload.get("explore_archive") or {}`.
The bootstrap string is read at `Popen` time, so rebinding the module global suffices.

### 8. `inference/framework/solver.py :: _HarnessGameSession._execute_action` (wrapped)
After every real action:
- **Animation-diff summary** (attacks the named sb26/tn36 blindness): when the engine
  returns intermediate frames (`state.animation_frames`, e.g. sb26 ACTION5 returns 42),
  computes a pixel-delta **text** summary across `[before, *intermediate, after]`
  (per-frame delta median/peak, change-region drift, top color flips, net change) capped
  at `PHASE1_ANIMATION_CHAR_CAP` (240 chars). **No multi-image injection** — Tufa showed
  it fails on 27B. Attached as `payload["animation_summary"]`, which patch 3 surfaces in
  the tool result and `run_explore` folds into probe reports.
- **Per-action archive feed:** observes the post-action grid into the dedup archive and
  marks the executed action as tried at the pre-action state signature — so
  model-initiated `action()` batches populate the archive too, not just explores.
- Guarded try/except; a failure degrades to vanilla behavior for that action.

## What is deliberately NOT changed (Tufa-proven keeps)
- No UNDO primitive.
- Single 4×-upscaled current-grid image per turn (`MULTIMODAL_CONTEXT=current_grid`,
  `MULTIMODAL_UPSCALE=4`).
- 64k `max_model_len` / 32k analyzer window and the drop-oldest eviction **scheme**
  (only its watermark granularity changes, patch 4).
- The single `python` tool; `explore()` is not exposed as a tool.
- Segmentation view, world-model note protocol, prompts (except the appended tail lines).

## Dedup hash definition
`state_signature(grid)` = sha1 over the segmentation graph: per 4-connected component
(≥ `PHASE1_NOISE_FLOOR` cells): (color, size, translation-invariant shape hash,
top-left position), plus the component adjacency pairs. Positions included ⇒ distinct
object placements are distinct states; sub-floor components excluded ⇒ stable under
1-px animation sparkle (floor is per-game tunable; default 1 = exact).

## Config reference (env vars, read at `apply()` time)
| Var | Default | Meaning |
|---|---|---|
| `PHASE1_EXPLORE_AFTER_TURNS` | 10 | N no-progress analysis steps before explore |
| `PHASE1_EXPLORE_BUDGET` | 6 | real actions per scripted explore (v2: was 8) |
| `PHASE1_MAX_EXPLORES` | 3 | explores per game cap (v2: was 6; 18-action worst case) |
| `PHASE1_EXPLORE_MIN_LEVEL_ACTIONS` | 90 | v2 mode detector: min actions on current level (or total, pre-first-level-up) before explore |
| `PHASE1_EXPLORE_LEVELUP_COOLDOWN` | 20 | v2 momentum guard: analyzer turns after a level-up during which explore is suppressed |
| `PHASE1_MOUSE_CANDIDATES` | 4 | MOUSE probe targets per explore |
| `PHASE1_SUMMARY_CHAR_CAP` | 1500 | explore report cap (≈500 tokens) |
| `PHASE1_ANIMATION_CHAR_CAP` | 240 | animation summary cap |
| `PHASE1_MIN_TIME_REMAINING` | 300 | skip explore under this many seconds left |
| `PHASE1_EXPLORE_MAX_LEVEL` | 99 | v2: explore only while best observed level <= this (1 = pre-first-clear only; 99 = v1 behavior) |
| `PHASE1_EVICT_LOW_FRAC` | 0.5 | eviction low watermark fraction |
| `PHASE1_NOISE_FLOOR` | 1 | signature component-size floor |
| `PHASE1_ENABLE_EXPLORE/ANIMATION/REPL_ARCHIVE/EVICT_HYSTERESIS` | 1 | ablation kill switches |
| `PHASE1_STRICT` | 0 | hook cell: raise instead of falling back to vanilla |

## Known interactions / risks
- Explore actions are **real** scorecard actions (no save-state exists); the report
  tells the model explicitly that they already happened.
- During explore, `analyze`'s runtime-state reads race nothing: session play-loop,
  analyze, step_env and the sandbox host all share one worker thread (RLock used for
  the archive because the explore loop re-enters the per-action hook on that thread).
- If `arc_agi`'s auto-reset fires mid-explore (game over), `run_explore` stops on the
  terminal flag; the session loop's auto-reset then proceeds normally.
- The `analysis_step`-keyed progress counter means N counts *fresh* analyzer turns, not
  60-s yield slices.
