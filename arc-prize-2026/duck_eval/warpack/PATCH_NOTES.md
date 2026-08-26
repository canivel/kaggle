# warpack v1 — fork-band adoption pack (intervention_plan.md R1)

Own port of the `taaf_grafts.composite` flag family (fork_band_audit.md), phase1-patch pattern.

## Components
- **Fast-submit gate** (`fastsubmit_cells.py` + `build_notebook.py` -> `duck_warpack.ipynb`):
  cells 4/6/8/10 wrapped in `if RUN_HEAVY:` (`RUN_HEAVY = KAGGLE_IS_COMPETITION_RERUN or
  WARPACK_FORCE_OFFLINE_BENCH=1`); Save Version writes a dummy `submission.parquet` in seconds.
  Build-time offline evals (duck_eval README rig) still available via `WARPACK_FORCE_OFFLINE_BENCH=1`.
- **soft_end**: real rerun gets `soft_end = start + 11h20m` (vanilla: None -> hard-kill risk).
- **warpack_patch.py** (env flags, kill switches, VERSION="v1", `apply(bm)` from cell 12 hook):
  - `banking` (WARPACK_BANKING): on run end with >=1 level done, open a NEW play on the same
    card (RESET full-resets when engine action-count==0 or WIN; two-RESET dance guarantees it)
    and replay the pruned winning trace via `GameAPI.env.step` directly. Card score = MAX over
    plays; per-level score = (baseline/actions)^2, so the pruned replay can only raise the max.
    Divergence (levels_completed or frame hash) aborts — free.
  - `recovery` (WARPACK_RECOVERY): REFRESH (clear `_history_messages` + hypothesis-graveyard
    line in `_summarized_knowledge`) after 30x same no-op action or 3 GAME_OVERs w/o progress.
  - `shortcircuit` (WARPACK_SHORTCIRCUIT): stops a homogeneous batch at the first confirmed no-op.
  - `retry_guard` (WARPACK_RETRY_GUARD): report-only counters, logged every 50 actions.
  - Master kill switch `WARPACK_ENABLE=0`; hook cell falls back to vanilla duck on any failure.

## Local-engine gotcha (test-relevant)
If `ONLY_RESET_LEVELS=true` is set before `arcade.make`, the make-time RESET level-resets and
play 1 never registers on the OFFLINE scorecard. GameAPI sets it *after* make on purpose; don't
pre-set it in local rigs. (Competition gateway registers plays server-side; unaffected.)

## war-v2 (ledger-ON) + eval build (panel R12 N6, prereg §4 design lock)
- `build_notebook.py --v2` -> `duck_warpack_v2.ipynb` = war-v1 + cell-2 gate
  detection-signal record + cell-12 ledger graft (`duck_eval/ledger/hook_cell.py`,
  flags default `ledger,escalation`, gated on RUN_HEAVY) + cell-12 **LEDGER
  CANARY** (observability only: per-game attempts / injected digests / skips /
  aborts + escalation firings; `LEDGER_CANARY=0` disarms; `LEDGER_FLAGS=""`
  no-ops graft+canary) + cell-14 greppable summary
  (`LEDGER CANARY game=<g> attempts=N digests=N skips=N aborts=N escalations=N`
  + a `TOTAL ... stores=` line — the keying tripwire: under ledger v2 per-game
  keying stores == n_games; stores=1 with many games = shared-store regression).
- `build_eval_notebook.py --v2` -> `notebooks/duckwar-v2-eval/` =
  `canivel/arc3-duck-war-v2-eval` (NOT pushed): duck_warpack_v2 + eval line
  (`WARPACK_FORCE_OFFLINE_BENCH=1`) + `WAR_EVAL_SEED=1` stamp/banner (ledger-ON
  seed 1, pairs with ledger-OFF war-eval seed 1; seed N = push N, as for the
  ledger-OFF seeds). Runtime modules come UNCHANGED from the `arc-war-kit`
  dataset (already carries ledger_core/ledger_patch since 07-13) — arms differ
  only by the notebook's flags, per the A/B design lock.
- `war_v2_eval_smoke.py` (39/39 PASS vs `_kaggle_dataset/` staged copies):
  structural S1-S6, cells 2/12/14 exec'd end-to-end, sb26 scripted L2 clear
  through the notebook-grafted harness, real-ToolAgent prompt path (digest +
  one-shot escalation + canary counts), fast-submit path, kill switches, and
  I7 two-games-CONCURRENT in one shared artifacts dir (zero cross-game digest
  contamination, per-game ledger files, canary TOTAL stores==2).
- `_kaggle_dataset/` = staged `canivel/arc-war-kit` version (NOT pushed):
  warpack_patch.py + ledger_core.py byte-identical to live v1;
  ledger_patch.py bumped to **ledger v2 per-game store keying** (see
  duck_eval/ledger/PATCH_NOTES.md); ledger_hook_cell.py fixed (live copy was
  a packaging mixup: a duplicate of the WARPACK hook — runtime-inert either
  way, nothing imports it). Push order: dataset version FIRST, verify files
  live + runtime banner `ledger v2: store keying = per-game:runtime-state-stem`,
  THEN kernel (feedback_kaggle_dataset_code_sync).

## Tests (`smoke_test.py`, 48/48 PASS, CPU-only)
W0 config/kill switches; W1 shortcircuit; W2 recovery refresh; W3 prune semantics;
W4-6 banking replays the war-room scripted winning traces (policies.py: sb26 24 actions,
su15 24, lp85 13) VERBATIM on new plays of the real local engines — engine card shows
2 plays, both levels_completed=2; W7 fast-submit dry-run (gate false -> dummy parquet, schema OK).
