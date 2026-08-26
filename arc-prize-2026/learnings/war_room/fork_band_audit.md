# Fork-Band Audit: what the 1.44-1.61 teams change vs vanilla duck
Date: 2026-07-13. Base for diffs: `notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb` (Jeroen Cottaar, 1.21 milestone).

## LB snapshot
Tecnod8.AI 1.61 (07-12) > Mathurin Ache / anngle / NoOneAhead 1.56 > paul 1.54 > ... > 1.44 wall (Lonnie, Tshithihi, Figuring out ARC AGI). **Nearly every 1.44+ team has a submission dated 07-12/07-13** — daily resubmission of a stochastic solver, score = max of N runs (order statistics). The 1.44 wall is consistent with "vanilla duck, resubmitted daily"; 1.56+ is more resubmissions and/or grafts.

## Concrete diffs found (pulled + cell-diffed vs base)

### 1. thtennant/kevin250304 "duck vN" lineage (v7..v13, v13 ran today)
Cell-identical to vanilla duck EXCEPT two cells:
- **Customization cell**: `from taaf_grafts.composite import install; install(bm, flags={...})`, blanket-guarded to fall back to stock HarnessSolver. Flags evolve: v7/v10 `{efficiency, retry_guard}` → v12 `+shortcircuit` → v9b `+recovery, banking`.
- **Run cell "safety pack"**: real rerun gets `soft_end = start + 11h20m` (vanilla had `soft_end=None`) so the solver drains and the shared scorecard closes before Kaggle's hard kill. Plus an offline-only dup-game gate (same env twice via `external_game_id`) to test their transfer/fingerprint family store cheaply.

They swap the source bundle dataset `jeroencottaar/taaf-kaggle-source-share` → **`thtennant/taaf-kaggle-source-share-fork` ("banking", updated 07-08, 152 downloads)**. Pulled it; `src/taaf-grafts/taaf_grafts/` contains:
- **banking_solver.py** — scorecard exploit: card score = MAX over plays; RESET while in WIN state opens a NEW play on the SAME card. On WIN they prune the winning trace (drop no-op actions) and replay it on a fresh play, driving `GameAPI.env` directly. Any divergence aborts (free — recorded win keeps the max).
- **recovery.py** — fixes the two dominant duck failure modes seen in forensics: GAME_OVER confusion loop (m0r0 scored 0.00, 137/172 turns stuck) and hypothesis lock-in (SPACE x71). R1 REFRESH clears chat history + writes a "hypothesis graveyard"; R2 bounded probe burst; R3 cross-level solved-mechanic handoff.
- **shortcircuit_solver.py** — stops a homogeneous repeated-action batch at the first confirmed no-op.
- retry_guard / efficiency analyzer (report-only), family_store, transfer_solver.

### 2. Fast-submit forks (submission-frequency enablers)
- `junjin2/tufa-duck-original-fastsubmit`, `maxingkong733/arc3-duck-dead-signature`: gate ALL heavy setup on `KAGGLE_IS_COMPETITION_RERUN`; interactive Save Version writes a dummy `submission.parquet` in seconds. This makes daily submissions nearly free of GPU quota — the mechanism behind the resubmission wall.
- `rokaiyasomapti/...-share-resubmission`: byte-identical vanilla duck, literally titled "Resubmission" (78 votes).

### 3. Model: NO public duck fork swaps the model
All duck forks keep `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` (Qwen3.6-27B FP8) + `driessmit1/arc3-vllm-h100-wheelhouse-v3`. Only model-adjacent signal: **`jcole75/arc3-qwen36-runtime-wheels` ("Qwen3.6 NVFP4 Runtime Wheels", updated 07-10, 16 downloads)** — someone is privately running an NVFP4 quant (faster tokens → more turns per 11h budget). Worth watching.

### 4. Non-duck lineages (all below the band)
- Gemma-4-31B VLM line: mbmmurad 0.86 single-file agent, ko0kip reflection agent, chrispaulwalker G1b packaging. maxingkong733 grafts dead-click-signature pruning (GEMMA_DEADSIG) onto the Gemma line, not the duck.
- wethepeople918 "Exact Non Replay": pure-Python symbolic flood-fill agent, no weights.

## Tecnod8.AI (1.61)
**Zero public kernels or datasets** (search "tecnod" = Not found). Submitted 07-12. No public evidence of their delta. Most probable given the ecosystem: private duck fork with grafts-class fixes (recovery/banking address exactly the losses that separate 1.21 from ~1.6) plus daily resubmission. The 1.56 trio's fresh daily dates say order stats alone plausibly reaches 1.56.

## Actionable
1. Adopt the fast-submit gate (dummy parquet outside rerun) — free daily order-stats.
2. Graft-equivalents to test: 11h20m soft_end cap, GAME_OVER refresh, win-replay banking (scorecard max-over-plays is engine-verified by thtennant).
3. Track `thtennant/arc3-duck-v13` (ran 07-13) and `taaf-kaggle-source-share-fork` updates; watch jcole75 NVFP4 wheels.
