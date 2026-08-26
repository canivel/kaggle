# boristown readiness-gate — 2-seed entry-gate eval READINESS (2026-07-30)

**Status: STAGED, build-rail only. NOT PUSHED, NOT QUEUED, $0 cloud.** This note
discharges the *build/staging* half of entry-gate #1 (2-seed eval canary) and
prepares entry-gate #2 (non-harm screen vs `runs/null10`) for the boristown
readiness-gate A/B (arm B = `canivel/arc3-duck-gate`). The orchestrator holds the
push slots; nothing below fires without a push it performs.

Authorities: `learnings/intents/boristown_ab_intent_2026-07-28.md`
(§"Entry gates", §"Built artifacts"), `learnings/war_room/boristown_ab_prereg_
2026-07-29_DRAFT.md` (BLOCKER 3). NO sealed / prereg / amendment file is modified
by this note or by any script it references.

---

## 1. What is staged (all local, unpushed)

| artifact | path | role |
|---|---|---|
| scored canary (arm B) | `notebooks/duckgate/arc3-duck-gate.ipynb` (+ `kernel-metadata.json`) | the SCORED A/B draw kernel (`canivel/arc3-duck-gate`), built 07-28, smoke 47/47 |
| eval build script | `duck_eval/a17/build_gate_eval_2seed.py` | emits the 2 seed variants (fork-never-build: derives from the staged scored canary) |
| eval seed-1 | `notebooks/duckgate-eval-s1/arc3-duck-gate-eval.ipynb` (+ meta) | entry-gate live-firing build 1/2 |
| eval seed-2 | `notebooks/duckgate-eval-s2/arc3-duck-gate-eval.ipynb` (+ meta) | entry-gate live-firing build 2/2 |
| eval smoke | `duck_eval/a17/gate_eval_2seed_smoke.py` | validates the 2-seed derivation (83/83 PASS) |
| non-harm screen | `scripts/gate_eval_screen.py` | entry-gate #2, prepared (no-ops until pulls exist) |
| pin byte-span | `runs/fork_diff_boristown/cells/boris_16_wait_body.txt` | contiguous `wait_vllm_ready` body for the single-diff T3 `--pin` |

**Eval slug is DISTINCT from the scored slug** — `canivel/arc3-duck-gate-eval`
vs scored `canivel/arc3-duck-gate` — exactly as the sentinel used
`arc3-duck-sentinel-eval` distinct from the scored sentinel. Eval builds must
never burn the scored slug's version history nor risk a `KAGGLE_IS_COMPETITION_
RERUN` mode collision.

## 2. Design — why a plain BUILD *is* the eval (no force-offline graft)

The war/sentinel evals fork the *duckwar* baseline and need
`WARPACK_FORCE_OFFLINE_BENCH=1` (`build_eval_notebook.py` `EVAL_LINE`) to run the
offline bench at build time. **The gate canary forks the frozen duck fork**,
whose run cell (cell 15) branches on `TRUE_SUBMISSION = KAGGLE_IS_COMPETITION_
RERUN`: when UNSET (any ordinary kernel BUILD) it plays the bundled competition
environments OFFLINE via `_offline_games()`, writes a dummy `submission.parquet`,
and is never scored. So a plain build of the gate canary already IS the offline
eval that fires the gate. **No eval-mode flag is grafted** — that would fork the
audited single-cell-graft invariant the A/B rests on. (Smoke E6 asserts no
`WARPACK_FORCE_OFFLINE_BENCH` / `RUN_HEAVY` leaked and the offline branch is
byte-preserved.)

The eval build therefore differs from the scored arm B in EXACTLY a greppable
seed-provenance stamp (mirrors the sentinel's `SENTINEL_EVAL_SEED`), following
the sentinel/`build_v7_seed2.py` convention "seed N = push N of the identical
notebook". Cell 2 gains, after the existing A17-GATE canary banner:

    os.environ["DUCK_GATE_EVAL_SEED"] = "<N>"   # greppable provenance only
    print("A17-GATE-EVAL seed=<N> mode=readiness-gate-ab-B-eval ... (offline bench, NOT scored)")

Seed-1 and seed-2 differ ONLY in the two seed substrings (`= "1"`/`"2"` and
`seed=1`/`seed=2`); reverse-substitution proves byte-identity of everything else.

## 3. Smoke + invariant results (all re-run 2026-07-30, CPU-only, $0)

- `boristown_gate_smoke.py` (scored canary) → **47/47 PASS** (unchanged from 07-28).
- `gate_eval_2seed_smoke.py` (2-seed eval derivation) → **83/83 PASS**
  (E1 compile; E2 each eval nb differs from scored canary in cell 2 only, gate+run
  cells byte-identical; E3 seeds differ in seed substrings only, reverse-sub proof;
  E4 GATE armed / GATE fired+latency / "vLLM server ready" survive on both seeds;
  E5 fresh eval slug, env byte-matched to scored canary, no model_sources, NO extra
  dataset; E6 no force-offline graft, offline branch intact).
- Single-diff-invariant **T3** (`scripts/preflight.py --mode trusted-fork
  --max-diff-cells 1 --pin runs/fork_diff_boristown/cells/boris_16_wait_body.txt`),
  run on the SCORED canary and BOTH eval seeds vs the frozen fork upstream:
  **T3 = OK** on all three (1 inserted gate cell carrying the pinned
  `wait_vllm_ready` body `sha256=9755ac54...`, 0 deleted, 0 rewritten, 1 banner-only
  additive edit = the cell-2 banner). Overall verdict **WARN** only because T4
  (build-status leg) is SKIPPED for an unpushed fork — this is the intent's
  "ALLOW", and matches the 07-28 read.
  - **Pin subtlety (record for the orchestrator):** the stock pin
    `boris_16_code.txt` (whole cell 16) is NOT a contiguous substring of the
    grafted gate cell (our latency-timing banner splits the trailing bare
    `wait_vllm_ready()` call from the def). Per `preflight.py`'s own docstring the
    pin must be the **contiguous function-definition body**; that is the derived
    `boris_16_wait_body.txt` (build-rail artifact, NOT a sealed file). It is the
    executed audited boris body, byte-preserved.

## 4. Push commands (ORCHESTRATOR ONLY — 2 free kernel BUILDS, counts vs 2/day; $0)

Do NOT submit. These are kernel BUILDS, not competition submissions.

    cd F:\kaggle\arc-prize-2026
    # seed 1
    uvx --from kaggle==2.0.0 kaggle kernels push -p notebooks/duckgate-eval-s1
    # seed 2 (only after seed-1 build COMPLETE, to keep 1 push in reserve/day)
    uvx --from kaggle==2.0.0 kaggle kernels push -p notebooks/duckgate-eval-s2

Both target slug `canivel/arc3-duck-gate-eval` (seed-1 = version 1, seed-2 =
version 2 — the two-seed proof is two versions of the same eval slug, exactly the
sentinel `arc3-duck-sentinel-eval` v1/v2 pattern). Expected build cost ≈ 2 × ~2.2
GPU-h against the weekly ~30 GPU-h quota (feedback_arc_zero_budget).

## 5. Pull-back verification (per push; feedback_kaggle_dataset_code_sync)

    uvx --from kaggle==2.0.0 kaggle kernels status  canivel/arc3-duck-gate-eval
    uvx --from kaggle==2.0.0 kaggle kernels pull    canivel/arc3-duck-gate-eval -p <tmpdir> -m
    # metadata round-trip MUST show: 3 dataset_sources (arc3-vllm-h100-wheelhouse-v3,
    #   taaf-kaggle-source-share, vrfai-qwen3-6-27b-fp8-hf-snapshot) — NO arc-war-kit,
    #   NO model_sources, machine_shape NvidiaRtxPro6000, enable_gpu true.
    # pulled ipynb MUST carry: A17-GATE-EVAL seed=<N>  (confirms the right seed built)

Then pull the outputs (benchmark.json + the kernel .log) into
`runs/kernel_pulls/gate_eval_v1/` and `.../gate_eval_v2/`:

    uvx --from kaggle==2.0.0 kaggle kernels output canivel/arc3-duck-gate-eval -p runs/kernel_pulls/gate_eval_v1

## 6. What the orchestrator MUST check in the logs (entry-gate #1, BOTH seeds)

Build status COMPLETE (not ERROR) on BOTH seeds, AND grep each kernel log for:

    A17-GATE-EVAL seed=<N> mode=readiness-gate-ab-B-eval   # right seed ran
    A17-GATE ... : GATE armed                              # gate armed
    vLLM server ready                                      # boris's own readiness line
    A17-GATE observed-firing vllm_ready_latency_s=<X> : GATE fired   # X <= 180.0

Entry-gate #1 discharges iff ALL of the above appear on BOTH seeds with latency
≤ 180 s. (GPU-parity spot-check, LOW per prereg NOTE 4: grep the log for the
RTX PRO 6000 GPU string; not expected to bind.)

## 7. Non-harm screen (entry-gate #2) — prepared, run AFTER the pulls exist

`scripts/gate_eval_screen.py` mirrors the sentinel screen (`war_eval_screen.py`,
same validated RHAE scorer 0e+00, same paired-Δlc exact sign-flip vs `null10`),
adding an explicit non-harm VERDICT. It NO-OPS with a clear message until
`runs/kernel_pulls/gate_eval_v1/benchmark.json` exists (tested today). After the
pulls:

    uv run python scripts/gate_eval_screen.py gate_eval_v1
    uv run python scripts/gate_eval_screen.py gate_eval_v2

PASS iff (a) mechanism fired (armed + fired + boris-ready, latency ≤ 180 s, read
from the pulled log) AND (b) Δlc not materially negative — harm-tail sign-flip
p ≥ 0.05 AND no game collapses > 1 level vs null (the sentinel-precedent bar; the
sentinel itself was admitted at Δlc = −0.128). Writes
`runs/gate_eval_v{1,2}/screen_report.md`.

## 8. Ready-to-push verdict

**YES — ready to push** (the two eval builds + their pull/screen tooling are
staged, runtime-tested, and invariant-clean). Caveats the orchestrator must honor:
1. These are FREE BUILDS, not scored submissions — do NOT `kaggle competitions
   submit`, do NOT touch `submission_queue.json`.
2. Entry-gate #1 is only *build-staged* here; it discharges only after BOTH seed
   builds are COMPLETE and the four log markers (§6) are grepped on both.
3. Entry-gate #2 (non-harm screen) requires the output pulls first; the script is
   prepared but unrun.
4. The single-diff T3 relies on the derived contiguous pin `boris_16_wait_body.txt`;
   BLOCKER 2 (the governance ratification that this local single-diff ALLOW
   substitutes for the strict pushed-fork T3) is SEPARATE and is NOT touched here.
5. The A/B itself remains gated behind the DRAFT prereg's seal + Sunday-08-02
   ratification; this note advances only the entry-gate build rail.

---
*Prepared 2026-07-30, build-rail only. No push, no submission, no queue change,
$0 cloud. Orchestrator holds the push slots.*
