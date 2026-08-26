# PREREG — Q38 FIELD-FLOOR ADOPTION ARM (`arc3-q38-field-eval` v1), sealed 2026-08-20 BEFORE the push
**Slot:** 2026-08-20 slot 1 (coordinator ruling per `rethink_2026-08-20.md` §4 ARM 1 / §5; principal-ordered rethink).
**Lane:** graft-lane agent operates both slots today; lock updated in `runs/lane_locks.json`.

## 0. WHAT THIS ARM IS
**An ADOPTION arm: byte-faithful rebase of the field's best-evidenced public artifact** — `foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b` (139 votes; author's team **2.23, #23, board-verified 08-20**; near-byte fork replicated at **1.91 on 10 lifetime subs**; two public testimonies: Ya Xu "xhigh > low", Scott "regularly >2, 50% >3 locally"). **Change NOTHING**: notebook content byte-identical (code sha `7227f3286cf60b25`, 11 cells / 10 code); only kernel-metadata retargets (fresh slug `canivel/arc3-q38-field-eval`, `is_private: true`); dataset/model/docker/machine fields byte-identical to FOYSAL's.
**Compound by design, declared:** engine (Q38 repacked Kaggle Model) × effort (UNPINNED ⇒ template `xhigh`) × harness generation (Jakob 08-07 anim bundle, tool_agent 108,927 B, animation NOT enabled — `MULTIMODAL_CONTEXT=current_grid`) × packaging move together, exactly as the field ships them. This arm answers "does the field's floor reproduce on our rail?", NOT "which component carries." The sealed Q38 REFUTE-2× stands, re-scoped to effort=medium on the June-30 harness (rethink §6.1).

## 1. BUILD-TIME CERTIFICATION (done at seal, recorded here)
- FOYSAL notebook byte-audit: `reasoning_effort` 0 · `taaf_grafts`/`install(` 0 · `banking` 0 · guarded wheels install (`if wheelhouse.exists():` — skip, not death, on the migrated layout) · dual-candidate mount resolvers for datasets/kernels + rglob bundle-marker fallback · gateway TRUE-SUBMISSION branch + `submission.parquet` writer present. **Layout-tolerant by construction.**
- Anim bundle FRESH download at seal: 75 files, manifest sha **`cbf26b9b913cc040b500cc52ad69f5ae`**, `setup_commands.json` sha **`650c2cfe94b972c3`**: `reasoning_effort` ABSENT (⇒ xhigh), yield **60**, `VLLM_MAX_MODEL_LEN 65536`, ctx 32768, temp 0.6, `preserve_thinking` true, `tool_agent.py` = 108,927 B. (Kaggle attaches LATEST; a run-time content change is caught by runtime certification below.)

## 2. RUNTIME CERTIFICATION (from the log; any failure ⇒ INFRA DEATH, never NULL/ADOPT)
1. Kernel reaches COMPLETE with `benchmark.json`, n_games = 25.
2. vLLM serve banner shows served model **`Qwen/Qwen3.8-27B-FP8`** (the repacked Kaggle Model — `feedback_kaggle_model_attach`: model attach is the silent-drop trap; ALSO asserted at pull-back).
3. Bundle identity: setup log lines show the anim-bundle setup commands (MODEL_OWNER/SLUG patch lines per the artifact) and NO `reasoning_effort` anywhere in the log.
4. No stock-fallback/ModuleNotFoundError-class death.

## 3. SEALED READING — dual endpoints, fresh ledger read at seal (n=37, mean 0.9316, s 0.1771, trailing-4 0.8425 — re-derived THIS MORNING with the 0.41 folded in)
- **PRIMARY (capability): lc_total over 25 games** vs baseline family `duck-harness-kaggle` m=3 (lc 18/19/21) — same sealed lines as the graft seal (K3″ + mirror): **HARM ≤ 12 · NULL 13–26 · SIGNAL ≥ 27**. Context recorded: our Q38-at-MEDIUM seeds measured lc 21/17 (NULL-class); the field config claims 2.2-class play, which should clear 27 if the capability story is real on our rail.
- **SECONDARY (recorded, NON-INFERENTIAL): mean_score** vs Q3.6 baseline spread 1.427/1.939/3.420 and Q38-medium's 2.795; expectation if the field class reproduces: **≥ 3.4** (clears the spread max). No verdict from score alone.
- **Pre-registered expectation:** SIGNAL on levels (P ≈ 55%) with mean_score 3–5. If levels land NULL but score is high, that is the graft-confirm lesson repeating on the engine side and will be READ as NULL-on-capability + descriptive score (no post-hoc conversion).
- **QUEUE-HEAD GATE (coordinator-ruled, separate from the science verdicts): COMPLETE + runtime certification §2 by 18:00 EDT ⇒ queue head tonight as the A21 exploration draw** (message cites rethink, verified 2.23, A21 budget). The lc/score bands do NOT gate the queue head — the external evidence (2.23 board-verified) carries the draw decision; our build run is its certification, not its audition. If not certified by 18:00: filler one more night.

## 4. WHAT THIS ARM CANNOT SETTLE
Component attribution (compound by design — the xhigh-vs-medium isolation and harness-generation isolation are FOLLOW-UP arms if this reproduces); anything about grafts (none present — tennant v21 compound is next window per ruling); the nightly rerun degradation (field-wide, applies to whatever we submit).

## 5. ARTIFACTS
`notebooks/q38-field-eval/` (content byte-identical to the pull at `rethink0820/foysal_lb9/`), pushed with pull-back verify incl. **model_sources EXACT match** `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1`. Results → `runs/kernel_pulls/q38_field_v1/`. Slot 2 (graft-confirm v5 per its own seal) pushes only after this arm is verified in flight.
