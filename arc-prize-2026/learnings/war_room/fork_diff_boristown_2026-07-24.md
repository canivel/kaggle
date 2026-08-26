# Fork diff: boristown/agi-duck-harness-fast-eval vs frozen fork (canivel/arc3-duck-repro v3)

Date: 2026-07-24. Analysis-only ($0, no pushes/submissions). Pulled artifact: `runs/fork_diff_boristown/pull/` (cell dumps in `runs/fork_diff_boristown/cells/`).

## Provenance

- Correct slug is **`boristown/agi-duck-harness-fast-eval`** (the prompt's `duck-harness-fast-eval` 403s — title carries a 【暗黑AGI】 prefix). 144 upvotes, last run 2026-07-22, public LB 1.47.
- Same duck-harness (TAAF/tufa-labs) family as our frozen fork. 22 cells vs our 17.
- **12 of boristown's 22 cells are byte-identical (md5) to our frozen fork**, including every load-bearing cell: env detect (`TRUE_SUBMISSION`, `ONLY_RESET_LEVELS`), wheel install, source-path setup + solver setup commands (vLLM launch), benchmark pickle restore, deadline/budget cell, the entire 4.4KB run cell (gateway wait, game-list swap, `bm.run(...)`, submission mechanics), and the results cell.
- The solver itself lives in the mounted dataset `jeroencottaar/taaf-kaggle-source-share`, **last updated 2026-06-12 — before our fork froze**. Both kernels mount the identical solver bytes. No hidden diff carrier.

## Cell-by-cell diff (everything not byte-matched)

| boris cell | vs our fork | Category | Semantic content | Rerun-time effect |
|---|---|---|---|---|
| 0 (md) | replaces fork[0-1] intro | docs | input links + section header | none |
| 8 (md) + 9 (code) | NEW | harness logic (claimed) | md describes an ACTION7 round-trip patch + animation-metadata patch; **code is a deliberate no-op** — prints `BASELINE_RUNTIME_STATUS` dict with every `*_changed: False` | none |
| 10 (md) + 11 (code) | NEW | harness logic (claimed) | md describes rolling the patch back; **code is a no-op** (`rollback_required: False`) | none |
| 15 (code) | replaces fork[12] (130-char comment stub for one-off `bm` tweaks) | harness logic | another no-op status print (`BASELINE_CUSTOMIZATION_STATUS`, all False). Explicitly forgoes trajectory memory, loop detector, budget override, prompt hints | none (both versions are no-ops) |
| **16 (code)** | **NEW — the only functional diff** | efficiency/robustness | `wait_vllm_ready()`: polls `http://127.0.0.1:1234/v1/models` every 5 s, up to 180 s, **before** the benchmark cell; raises if vLLM never comes up | closes a startup race our fork has: fork waits for the **gateway** (`_wait_for_gateway`, 600 s) but never for the **vLLM server**, which is launched async by the solver setup commands |
| 21 (code) | NEW trailing cell | local diagnostics | HTML "LOCAL FAST EVALUATION" score card parsed from `taaf.diagnostics.run_summary_text(bm)`; **gated off when `TRUE_SUBMISSION`**; reads `LOCAL_FAST_EVAL_SECONDS` which is **never defined anywhere** → always shows "full local budget" | none in rerun |

Notes:
- The "fast-eval" branding is local-eval convenience only. The advertised shortened local per-game budget hook (`LOCAL_FAST_EVAL_SECONDS`) is not even set in the saved version. The cell's own footer states the gateway, model, sampling, concurrency, game list, and submission path are unchanged for official reruns.
- Cells 8–11/15 look like patch archaeology: the author tried an ACTION7 + animation-metadata monkey patch, it was score-sensitive, they rolled it back, and finally made all patch cells inert while keeping the markdown "for traceability." **The markdown lies about what the code does; the code is what counts, and it does nothing.**

## Metadata diff

**Zero mismatches.** `kernel-metadata.json` is field-for-field identical to ours except identity fields (`id`, `id_no`, `title`, `is_private`):
- `enable_gpu: true`, `enable_tpu: false`, `enable_internet: false`
- `dataset_sources`: same 3 (wheelhouse-v3, taaf-source-share, qwen3-6-27b-fp8 snapshot) — same order, no version pins
- `docker_image`: same sha256 (`...be4cb13c`)
- `machine_shape: NvidiaRtxPro6000`, `competition_sources: [arc-prize-2026-arc-agi-3]`, empty `model_sources`/`kernel_sources`

kaggle_env_match discipline: fully satisfied out of the box.

## (a) +0.14 mechanism hypothesis

At rerun time boristown ≡ vanilla duck baseline + a vLLM readiness gate. There is **no depth channel**: no budget, prompt, model, sampling, action-selection, retry, or reset change; no mechanism for completing more levels. Two components, in order of weight:

1. **Run-to-run variance (primary).** Our own byte-identical frozen fork spans 0.82–1.33 across reruns (mean ~0.975). 1.47 is ~1 band-width above our mean — a plausible right-tail draw of the same distribution. 144 upvotes measure popularity, not reproducibility; the sibling fork `simosc/agi-duck-harness-fast-eval` (byte-family) exists and its score should be checked as a second draw.
2. **vLLM readiness gate (secondary, floor-raising).** Efficiency-channel, small. If any of our low-band reruns (0.82) burned early wall-clock or early game actions on LLM calls issued before the vLLM server finished loading the 27B FP8 model, the gate removes that failure mode. It raises the left tail of the distribution more than the mean; it cannot plausibly account for +0.5 over our mean on its own.

Net: expect a boristown byte-fork to draw from approximately our fork's distribution, plus a slightly better floor. **Do not budget +0.14 as systematic.**

## (b) Fork viability

**GREEN — trivially forkable, lowest-risk class we've seen.**
- No API keys (only the standard local `ARC_API_KEY=test-key-123` placeholder, byte-identical to ours), no external network deps (`enable_internet: false`; the only HTTP is localhost vLLM + gateway), no litellm or any banned package, no new pip installs, no new datasets.
- Submission mechanics byte-identical (same run cell md5). Same competition source. Same TRUE_SUBMISSION gating; the new score card explicitly hides itself in real reruns.
- It literally *is* our frozen fork + 5 no-op cells + a 25-line health check + a display card. A byte-matched fork is equivalent to grafting one robustness cell onto code we already trust.
- Fresh-slug consideration (feedback_fresh_kernel_slug): forking gives us a fresh slug for free.

## (c) Graft compatibility ((f) continuation, sentinel telemetry)

**Ports cleanly.** All of our graft anchor cells (env detect, setup-commands cell, pickle-restore cell, run cell) are byte-identical between boristown and our frozen fork, so (f)-continuation and sentinel-telemetry diffs apply unchanged. One caveat: the one-off customization stub (our fork[12], the natural injection point for `bm`/solver tweaks) was replaced by boristown's no-op status cell — any graft keyed on that cell's literal text needs its anchor updated (content is a no-op either way). Conversely, boristown's `wait_vllm_ready` cell can be grafted **into our existing lineage** (duckwar, sentinel) as a standalone 25-line cell with zero interaction risk — it runs before the benchmark and only polls localhost.

## Recommended action for panel

1. **Adopt the vLLM readiness gate everywhere** (frozen-fork lineage, duckwar, sentinel kernels): free left-tail insurance, zero interaction risk, no score-sensitive surface.
2. **Byte-match fork boristown as the filler-replacement candidate** if the panel wants the floor-raise move: risk is nil (env identical, code = ours + no-ops + gate), and it refreshes the slug. Gate it through the normal `runs/null10` comparison — the honest EV is "our distribution + slightly better floor," not 1.47.
3. **Do not chase the +0.14 as a mechanism.** There is no depth channel in this artifact; treating 1.47 as reproducible would be public-LB luck-chasing (feedback_arc_generalization_first).
4. Optional cheap evidence: pull `simosc/agi-duck-harness-fast-eval` LB score as a second independent draw of the same bytes to bound the variance hypothesis.
