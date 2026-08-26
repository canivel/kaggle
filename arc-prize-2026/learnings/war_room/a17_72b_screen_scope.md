# A17 — war-v4 72B capability screen: scoping doc

Filed 2026-07-19 (scoping only — NO kernel pushes, NO downloads, NO cloud spend).
Discharges the scoping half of pre-registration amendment A17
(`learnings/preregistration_amendment_2026-07-18b.md` §A17) and the Aug-1 v4 row of
`grinder_cracking_design.md` §5. Incorporates the four REQUIRED items from panel R15
(gate boolean, comparator statistic, hardware SKU verification, quota ledger +
resync caution). Blocking: the screen must RUN pre-Aug-1.

**One-line thesis.** A model swap is the campaign's only registered wall-closer, but
on this single-GPU rail the 72B pays a ~2.5–3× decode penalty that costs it 2–3 of
the 4-game level budget before it thinks a single better thought. The screen exists
to measure whether the capability gain clears that self-inflicted hole. This doc
seals the arithmetic that decides it, before the numbers are observed.

---

## 0. Hardware SKU — verified from kernel logs (R15 item 3)

Both rails are the **same physical SKU** — verified, not assumed:

| rail | accelerator | count | memory | source |
|---|---|---|---|---|
| free build/eval rail (this screen) | **NVIDIA RTX PRO 6000 Blackwell Server Edition** (sm_120, cc 12.0) | **×1** | ~96 GB | `runs/kernel_pulls/w0_eval_s1/arc3-duck-w0-continuation-eval.log`: `CUDA GPU check passed for rtx-pro-6000 x1: ['NVIDIA RTX PRO 6000 Blackwell Server Edition']` |
| scored competition rail | same SKU | ×1 | ~96 GB | `notebooks/duckwar/kernel-metadata.json`: `"machine_shape": "NvidiaRtxPro6000"`; harness `assert_expected_cuda_gpu()` hard-asserts `rtx-pro-6000 x1` (setup_commands.json) |

The build rail = the scored rail for this competition (both single RTX PRO 6000 96GB),
so a build-time throughput number transfers to the scored budget with no cross-SKU
correction. **R15 mandate honored:** the tokens/s probe (§2) MUST run on this exact
SKU before N₇₂B is computed; if a future kernel prints any other GPU name the entire
null (§3) and action-count gate (§1) are recomputed. The 27B baseline used here (192
tok/s, `w0_eval_s1/summary.txt`) was itself measured on this SKU, so ρ is a same-SKU
ratio.

Memory ledger on the 96GB card (from the 27B log, our extrapolation for 72B):
- 27B FP8 baseline: model load **33.66 GiB**, KV cache **43.62 GiB** at
  `gpu_memory_utilization 0.92`, `max_model_len 65536`. (`w0_eval_s1/vllm-openai-server.log`.)
- 72B AWQ (4-bit) weights ≈ **43.0 GB** (§1) → at util 0.92 (~88 GB usable) leaves
  ~40–42 GB for KV + VL vision-tower activations + CUDA-graph pool. That is *below*
  the 27B's 43.62 GB KV headroom, so **`max_model_len` must drop from 65536 to
  32768** (which equals the harness's `ANALYZER_CONTEXT_WINDOW=32768` — the analyzer
  never uses more than 32k anyway, so this costs nothing behaviorally). This is a
  REQUIRED bench-config change, not optional (§2, §4-risk-B).

---

## 1. WEIGHTS — which 72B-tier 4-bit model

**Decisive constraint the A17 text omits:** the current harness is **multimodal**.
It renders the current grid as a 4× upscaled image and feeds it to the analyzer
(`setup_commands.json`: `MULTIMODAL_CONTEXT=current_grid`, `MULTIMODAL_UPSCALE=4`),
and the 27B model is a **vision-language** model — the server log resolves
architecture `Qwen3_5ForConditionalGeneration` and loads `Qwen2VLImageProcessor`.
**A text-only 72B would silently delete the visual channel** and would not be a
like-for-like swap — it would confound the capability screen with a modality
regression. The swap model MUST be VL.

### Top candidate (recommended): Qwen2.5-VL-72B-Instruct-AWQ

| field | value |
|---|---|
| Kaggle ref | `qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1` (official QwenLM model, verified present via `kaggle models instances versions files`) |
| modality | **vision-language** (matches the harness's `current_grid` image path) |
| quant | AWQ W4A16, 4-bit |
| size | **43.0 GB** (11 safetensors shards; sum verified = 43,023,138,387 B) — under the ~45 GB envelope |
| attach | as a **Kaggle Model source** (`model_sources` in kernel-metadata) OR re-snapshot into a private dataset mirroring the 27B pattern (`vrfai-qwen3-6-27b-fp8-hf-snapshot` is a *dataset*). Model-source attach is simpler and needs no upload/quota. |
| arch | `Qwen2_5_VLForConditionalGeneration` — supported by vLLM 0.19; same VL family as the 27B, so the harness's image-processor path is unchanged |

### Secondary candidate (text-only fallback, only if VL-72B won't serve single-GPU): Qwen2.5-72B-Instruct-AWQ
- Kaggle ref `qwen-lm/qwen2.5/transformers/72b-instruct-awq/1` (verified present), ~43 GB (10 shards summed 39.1 GB + shard 11 ≈ 43 GB), AWQ 4-bit.
- **Text-only** — would require also disabling `MULTIMODAL_CONTEXT` to be a fair test, changing the harness contract. Carries a modality confound; use ONLY if VL-72B fails to fit single-GPU and record the modality change explicitly.

### Not available / not chosen
- **No Qwen3-tier 72B AWQ exists on Kaggle** (searched models + datasets: "Qwen3 72B", "72B AWQ" → *No models/datasets found*). The A17 text's "Qwen3.6-72B-tier" has no attachable Kaggle artifact today; `Qwen2.5-VL-72B-AWQ` is the closest attachable 72B-tier VL model. If a Qwen3-VL-72B AWQ appears on Kaggle before the push, prefer it (same swap procedure).
- NVFP4 variants (e.g. `Qwen3.5-397B-A17B-NVFP4`) are MoE and/or multi-GPU — out of the single-card envelope.

**Weights verdict: `qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1`, 43.0 GB, attached as a Kaggle Model source.**

---

## 2. BENCH KERNEL DESIGN — minimal diff to the proven eval builder

### What a model swap actually touches (audited)
The vLLM server is NOT launched from the notebook. It is launched by the TAAF bundle's
`setup_commands.json` (in the `jeroencottaar/taaf-kaggle-source-share` dataset;
local copy `duck_eval/taaf_bundle/setup_commands.json`). That embedded PYSETUP script
hard-codes:
```
MODEL_OWNER='driessmit1';  MODEL_SLUG='vrfai-qwen3-6-27b-fp8-hf-snapshot'
SERVED_MODEL_NAME='vrfai/Qwen3.6-27B-FP8';  VLLM_MAX_MODEL_LEN=65536
ANALYZER_CONTEXT_WINDOW=32768;  VLLM_TENSOR_PARALLEL_SIZE=1
```
and builds the launch cmd: `vllm ... --model $MODEL_PATH --served-model-name ...
--tool-call-parser qwen3_coder --reasoning-parser qwen3 --enable-prefix-caching
--default-chat-template-kwargs '{"preserve_thinking": true}' --max-model-len 65536`.
`inference.json` mirrors `gpu_memory_utilization: 0.92`, `dtype: auto`.

So a 72B bench = a **war-v4-eval builder variant** (extend
`duck_eval/warpack/build_eval_notebook.py` with a `--v4` mode) that:
1. **Swaps the attached model source** in kernel-metadata (drop the 27B dataset, add
   `qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1` as a model_source), AND
2. **Ships a patched `setup_commands.json`** (or a graft that rewrites those four
   constants at build time, mirroring the continuation-patch graft pattern in the
   W0 builder) with:
   - `MODEL_SLUG` / `MODEL_PATH` → the VL-72B-AWQ mount
   - `SERVED_MODEL_NAME` → `Qwen2.5-VL-72B-Instruct-AWQ`
   - `--quantization awq_marlin` (explicit; on sm_120 AWQ resolves to the Marlin
     W4A16 kernel — see §4-risk-C)
   - `--max-model-len 32768` (memory, §0; = analyzer context anyway)
   - keep `gpu_memory_utilization 0.92`, `tensor_parallel_size 1`, prefix-caching ON
   - **tool/reasoning parsers:** Qwen2.5-VL uses the `hermes` tool-call parser and has
     **no `qwen3` reasoning parser** — `--tool-call-parser qwen3_coder
     --reasoning-parser qwen3` will error on this model. This is a REQUIRED change:
     set `--tool-call-parser hermes`, drop `--reasoning-parser`, and drop
     `preserve_thinking`/`enable_thinking` (Qwen2.5 has no thinking mode). See
     §4-risk-D — this is the single most likely thing to zero the run if missed.
3. **A tokens/s probe.** The 27B bench already emits `generated tokens/sec` in
   `summary.txt` (192 tok/s job-wallclock; the server log shows ~290–450 tok/s
   aggregate decode across 25 concurrent reqs during steady state). The v4 bench
   inherits this instrumentation for free — the same `summary.txt` line gives ρ.

### Games and passes that fit one build-time session
The bench runs **all games concurrently** (`concurrent_jobs: 32`, the server log shows
`Running: 25 reqs`), each game to a **FIXED ~7920 s (2h12m) wallclock** window — NOT a
fixed action budget (verified: every game's `final_wallclock_seconds ≈ 7920` in every
benchmark.json). This is the load-bearing fact for the null (§3).

- 27B reference: 25 games × 1 pass in **2h12m** wallclock, 192 tok/s.
- The 4 screen games (ft09/sb26/lp85/vc33) run **inside the same 2h12m window** as a
  4-game subset — the window length is fixed by the harness soft-deadline, not by
  game count, so a 4-game bench is still ~2h12m + model load. Running only 4 games
  (not 25) frees KV pressure and lets more concurrent replicas of those 4 games fit —
  but per A17 the screen is **full per-game budget**, so we keep 1 pass/game/window
  and get replication across seeds (separate pushes), matching the war_eval
  convention (seed N = push N).
- **Load time:** 27B took **231 s** to load weights (33.66 GiB) + ~200 s engine init
  (compile/warmup/KV) ≈ **~7 min** startup. 72B AWQ at 43 GB via Marlin (dequant on
  load) is heavier — budget **~10–14 min** startup. Startup is inside the kernel's
  own wallclock, not the 2h12m game window.

### GPU-h cost per bench push
One bench push ≈ model-load (~0.2 h) + game window (2.2 h) + diagnostics (~0.1 h) ≈
**~2.5 GPU-h/push**. (The 27B kernels wall in at ~2h12m game + overhead; call it 2.5 h
to be safe for the slower 72B load.)

---

## 3. THROUGHPUT-ADJUSTED NULL — the sealed formula

**Mechanism (why a null is needed at all):** the screen is a **fixed-wallclock** race,
not fixed-action. The 27B did N₂₇B actions per game in 7920 s at ~192 tok/s. The 72B
at ~1/ρ the decode speed does **N₇₂B = ⌊(1/ρ)·N₂₇B⌋** actions in the *same* 7920 s.
Comparing raw levels would credit the 72B for capability while hiding that it simply
got fewer moves. The null prices that in.

### Definitions (all sealable pre-screen)
- **ρ (rho)** = `tok/s(27B) / tok/s(72B)`, **measured** from the two `summary.txt`
  `generated tokens/sec` lines on the SAME SKU (§0). A17 assumes ρ ≈ 2.5–3.0; the
  screen uses the *measured* ρ, and both ρ=2.5 and ρ=3.0 tables below are pre-computed
  so no post-hoc choice is possible. If measured ρ lands outside [2.4, 3.1], recompute
  null_adj at the measured ρ using the frozen procedure below (the procedure, not the
  numbers, is what seals).
- **N₂₇B(game)** = total actions the **W0 27B baseline** took in that game =
  Σ `actions_per_level` (from `runs/kernel_pulls/w0_eval_s1/benchmark.json`; this is
  the (f)-only baseline the whole v3 campaign is scored against).
- **N₇₂B(game)** = ⌊(1/ρ)·N₂₇B(game)⌋.
- **null_adj(game)** = the number of levels the **W0 27B baseline had already
  completed by action N₇₂B** — i.e. walk the 27B's `actions_per_level` cumulatively
  and count how many level-blocks fully close within N₇₂B actions. (A level counts iff
  the 27B's cumulative action total reached the end of that level's block.)

### The frozen arithmetic (computed now, from W0 27B per-level actions)
W0 `actions_per_level` [levels_completed]: ft09 [27,10,2] lc2 · sb26 [16,209] lc1 ·
lp85 [8,139] lc1 · vc33 [7,19,43] lc2. N₂₇B = 39 / 225 / 147 / 69.

| game | N₂₇B | 27B full-budget lc | N₇₂B (ρ=2.5) | null_adj (ρ=2.5) | N₇₂B (ρ=3.0) | null_adj (ρ=3.0) |
|---|---:|---:|---:|---:|---:|---:|
| ft09 | 39 | 2 | 15 | **0** | 13 | **0** |
| sb26 | 225 | 1 | 90 | **1** | 75 | **1** |
| lp85 | 147 | 1 | 58 | **1** | 49 | **1** |
| vc33 | 69 | 2 | 27 | **2** | 23 | **1** |
| **Σ** | | **6** | | **Σ null_adj = 4** | | **Σ null_adj = 3** |

Note the honest sting: **ft09 null_adj = 0** even though the 27B cleared 2 levels —
the 27B needed 27 actions just to clear ft09-L1, but N₇₂B is only 13–15, so a 72B that
merely *matched* 27B skill would clear **zero** ft09 levels in the throttled budget.
That is the throughput penalty made concrete: Σnull_adj (3–4) sits *below* the 27B's
own full-budget Σlc (6). The 72B must clear this reduced bar with its extra capability.

**Sealed target: 72B (measured, same comparator statistic as §"comparator") must beat
Σ null_adj = 4 (if ρ≤2.5) / 3 (if 2.5<ρ≤3.0), computed at the measured ρ by the frozen
procedure above.** Because null_adj can only be recomputed by re-running the fixed walk
on the frozen W0 per-level data at the measured ρ, there is no post-hoc freedom.

---

## 1b/COMPARATOR — the single comparator statistic (R15 item 2)

A17's "≥2 levels beyond the 27B baseline" is undefined because per-game 27B lc spans
0–2 across the 3 certified seeds (ft09: 1/2/0; vc33: 2/1/2; sb26 1/1/1; lp85 1/1/1),
so mean-vs-max shifts the bar by up to ~2 levels = the entire GO margin.

**Decision (adopting rl-planning M5): the comparator is per-game MAX over the 3
certified seeds, on BOTH sides.** Justification: the screen tests **capability
existence** ("can a 72B reach levels the 27B tier cannot?"), and existence is a max-
property, not a mean-property. Using max on both sides is symmetric (no bias toward
either model) and is the same order-statistic the banking line already uses
(E[max-of-attempts], grinder_design §2). The 27B side max (from war_eval v1/v2/v3 +
W0) per game:

| game | 27B seed lc (v1/v2/v3/W0) | **27B MAX** | 72B GO bar (max + 2, capability prong) |
|---|---|---:|---:|
| ft09 | 1 / 2 / 0 / 2 | **2** | 4 |
| sb26 | 1 / 1 / 1 / 1 | **1** | 3 |
| lp85 | 1 / 1 / 1 / 1 | **1** | 3 |
| vc33 | 2 / 1 / 2 / 2 | **2** | 4 |
| **Σ** | | **Σ 27B MAX = 6** | **capability prong: Σ 72B MAX ≥ 8** |

- **Capability prong (A17's "≥2 levels beyond"):** Σ(72B per-game MAX) ≥ Σ(27B per-game
  MAX) + 2 = **≥ 8**. To keep the 72B seed count feasible the 72B side max is taken
  over however many 72B seeds the quota affords (≥1; see marginal rule).
- **Throughput prong / null:** Σ(72B per-game lc) > Σ null_adj (§3) — evaluated on the
  72B's *own* achieved actions in-window, per game, using the same cumulative-walk
  definition against the 72B's `actions_per_level`.

**Marginal-result rule (pre-stated, no coin flip):** if the capability prong lands at
**exactly +1** (Σ 72B MAX = 7), OR either prong is within one level of its threshold,
run **one additional 72B seed on the two decisive games** (the two with the largest
72B-vs-27B per-game gap) and re-evaluate MAX with the added seed. A single additional
seed, not a re-roll of the whole screen; if still +1 after the extra seed → **NO-GO**
(the capability is not robustly ≥2).

---

## GATE BOOLEAN — the sealed decision tree (R15 item 1)

A17's prose reads as a conjunction whose second conjunct (actions ≥90% of 27B)
**auto-fails** under any real 2.5–3× slowdown, making NO-GO deterministic and the
null_adj formula dead code. R15 requires this be an explicit disjunction:

```
GO  iff
    ( CAPABILITY:  Σ(72B per-game MAX lc)  ≥  Σ(27B per-game MAX lc) + 2      # ≥ 8
      AND
      ACTION-PARITY:  Σ N₇₂B  ≥  0.90 · Σ N₂₇B )                             # throughput not binding
  OR
    ( CAPABILITY  (same ≥8 bar)
      AND
      THROUGHPUT-ADJUSTED:  Σ(72B per-game lc)  ≥  Σ null_adj  +  MARGIN )   # throughput binding, but wins anyway
NO-GO otherwise.
```

- **First disjunct** = the (unlikely) world where the 72B is fast enough that
  throughput doesn't bind; then judge on raw capability at near-equal action counts.
- **Second disjunct** = the EXPECTED world (ρ≈2.5–3): throughput binds, so capability
  is judged against the throughput-adjusted null, NOT against raw 27B levels. This is
  the branch A17 left open; it is now closed.
- **Registered MARGIN (methodology N4 / systems #13):** MARGIN = **+1 level**. The 72B
  must beat Σnull_adj by at least one full level (Σ72B lc ≥ Σnull_adj + 1, i.e. ≥5 at
  ρ≤2.5 / ≥4 at ρ≤3.0). Rationale: null_adj is an integer step function of ρ; a +1
  margin protects against a ρ-measurement error nudging null_adj by one, and matches
  the "≥2 beyond baseline" spirit without demanding two on the throttled budget.
  **Test:** exact — integer level counts, no p-value; the margin IS the test. (n=1–2
  72B seeds do not support a sign-flip test; the screen is a capability-existence
  check, deliberately not a powered score gate. This is stated so the panel does not
  later demand α on a 1-seed screen.)
- **NO-GO consequence (A17-sealed):** "72B replicates the ~1-level grinder profile" →
  goes to the panel **immediately**; the campaign then has no registered wall-closer
  and the panel decides in July, not September.

---

## 4. RISKS

**A. Reset/resync fragility — MANDATORY A/B, not optional.** Community report: a
reset-cap change turned a 9-min agent into a 1-hour 0-score run. The harness pins
`ONLY_RESET_LEVELS=true` / `os.environ["ONLY_RESET_LEVELS"]="true"` (notebook cell 2)
and `max_runtime_minutes: 45` per game. **The 72B swap must change ONLY the model +
its 5 serve-config constants — no reset/timeout/budget logic.** The v4-eval builder
must assert (like the W0 builder asserts cell-12 is the warpack graft) that the reset
constants and the game-window deadline are byte-identical to the 27B baseline. The
72B bench is A/B'd against the frozen fork implicitly: the 27B W0 seeds ARE the
control arm; if the 72B kernel's game window is not ~7920 s (i.e. a reset/deadline
regression crept in), the null comparison is void and the run is discarded, not
scored.

**B. Memory / max-model-len.** 43 GB AWQ weights + VL vision-tower activations on one
96 GB card is **feasible but tight** — community guidance for VL-72B-AWQ leans multi-
GPU. Mitigation is forced in §0/§2: `--max-model-len 32768` (= analyzer context, zero
behavioral cost) to shrink KV pre-allocation; keep util 0.92. If it still OOMs on
load, drop util to 0.88 before touching max-model-len further. If single-GPU serve is
impossible, the screen is not runnable as scoped and that finding itself goes to the
panel (there is no 2nd GPU on the rail).

**C. AWQ kernel on Blackwell sm_120.** Verified via vLLM forums/issues: on sm_120
(cc 12.0) AWQ W4A16 runs through the **Marlin fallback** (`awq_marlin`), which **is
functional** — Qwen3-8B-AWQ confirmed serving on sm_120 RTX PRO 4000 Blackwell across
vLLM 0.15–0.18; vLLM 0.19.x confirmed running on RTX PRO 6000 Blackwell (May 2026).
Marlin-AWQ is in fact the *fast* path (reported ~741 tok/s / 10.9× on suitable models).
Known wrinkle: sm_120 is not always auto-recognized in the FP4 backend selector — this
affects **NVFP4/MXFP4**, NOT AWQ; still, set `--quantization awq_marlin` **explicitly**
rather than relying on auto-detection. Our wheelhouse pins **vLLM 0.19.0** (server log
+ `STAMP_TEXT` in setup_commands) — do NOT bump it; the whole rail is validated at
0.19.0. If awq_marlin errors on 0.19.0 for the VL arch specifically, that is a hard
blocker → panel.

**D. Tokenizer / prompt-format / parser mismatch (highest-probability silent-zero).**
The 27B is a Qwen3.5-family VL model served with `--tool-call-parser qwen3_coder
--reasoning-parser qwen3 --default-chat-template-kwargs '{"preserve_thinking":true}'`
and `enable_thinking:true`. **Qwen2.5-VL-72B has none of these**: its tool parser is
`hermes`, it has no qwen3 reasoning parser, and no thinking mode. Shipping the 27B
serve flags against the 72B will either crash the server (bad parser) or silently
break tool-call extraction (agent emits no valid actions → 0 score, the exact
`feedback_test_before_submit` failure class). **REQUIRED:** `--tool-call-parser
hermes`, drop `--reasoning-parser` and thinking kwargs, and set
`LOCAL_ANALYZER_ENABLE_THINKING=false` in the setup_env. Runtime-test (smoke test +
a ≥5-action canary that a tool call is parsed) BEFORE the scored screen push. The
bundle's own `run_vllm_api_smoke_test()` already fires a chat completion at boot — but
extend it to assert a *tool call* round-trips, since that's what the agent needs.

**E. Chat-template / multimodal image contract.** The harness sends `current_grid` as a
4× image. Qwen2.5-VL's image processor differs from Qwen3.5's; confirm the image
lands (server log should show `Qwen2_5_VLImageProcessor` and non-zero MM cache hits,
as the 27B log showed `MM cache hit rate: ~84%`). If MM cache stays 0%, the image path
is broken and the screen is testing a blind model — discard.

---

## 5. SCHEDULE — day-by-day to pre-Aug-1 (≤2 pushes/day, shared)

Push budget is **2 kernel pushes/day shared with other lines** (v3 windows W2/W3/W4,
A14/A15 confirmation replicates). This screen needs **1 successful bench push** + 1
reserved retry slot. Reset-caution and the config changes above mean the first push is
likely to need a fix, so a dry canary precedes the scored bench.

| date | action | pushes | GPU-h |
|---|---|---:|---:|
| **Jul 22** | Build `--v4` eval variant: patch setup_commands constants (model, awq_marlin, max-model-len 32768, hermes parser, thinking off), swap model source, extend smoke test to assert a tool-call round-trip. Attach `qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1`. **Runtime-test locally where possible** (metadata/JSON validity, builder asserts). NO push. | 0 | 0 |
| **Jul 24** | **Push #1 = boot/canary bench** (short window or 4 games): verify (a) VL-72B-AWQ serves on sm_120 at max-model-len 32768 without OOM, (b) smoke + tool-call round-trip passes, (c) MM image lands (non-zero MM cache), (d) `generated tokens/sec` line prints → **measure ρ**. If it errors → this is the reset/config-fragility slot; fix, use the day's 2nd push or Jul 25. | 1 (of 2) | ~2.5 |
| **Jul 26** | **Push #2 = scored screen bench**: 4 games (ft09/sb26/lp85/vc33), full 7920 s window, 1 pass. Emits per-game lc + actions_per_level + tok/s. This is seed 1 of the 72B arm. | 1 (of 2) | ~2.5 |
| **Jul 28** | (If marginal per the +1 rule, or a 2nd 72B seed wanted for MAX) **Push #3 = 72B seed 2** on the 4 games (or the 2 decisive games only). | 1 (of 2) | ~2.5 |
| **Jul 30** | **Screen look:** compute measured ρ → freeze null_adj at that ρ → evaluate the GATE BOOLEAN. GO or NO-GO(→panel). Buffer day for one more replicate if the marginal rule triggers. | 0–1 | 0–2.5 |

Pre-Aug-1 met with 2 days of slack. Weights need **no upload** (Kaggle Model source),
so no dataset-prep push and no download (respects scoping/no-spend).

### Quota ledger vs 30 GPU-h/week (R15 item 4)
This screen: 3 bench pushes × ~2.5 GPU-h ≈ **~7.5 GPU-h**, spread across the Jul 22–30
window (≤ two calendar weeks) → **~3.75–7.5 GPU-h in any single week** — comfortably
inside 30 GPU-h/wk **on its own**. **BUT** systems #12 flags that A14 (cumulative
confirmation look: final v3 stack vs W0, 3 certified seeds) and A15 (one full-budget
confirmation replicate) also draw on the same 30 GPU-h/wk envelope in this window. At
~2.5 GPU-h/push, 3 A14 seeds + 1 A15 replicate ≈ 10 GPU-h, plus this screen's 7.5, plus
any v3 W2/W3/W4 windows (~2.5 each) can approach the cap in a heavy week. **This doc
does not own that combined schedule** — it is a stated **dependency**: the war-room
weekly scheduler must slot A17's 3 pushes so that (screen + A14 + A15 + open v3
windows) ≤ 30 GPU-h in each of the Jul 20–27 and Jul 27–Aug-3 weeks. If contended,
A17's canary (Jul 24) and scored bench (Jul 26) are the protected pair; the marginal
2nd/3rd seed yields first.

---

## Summary (the four asks)

- **Top weights candidate:** `qwen-lm/qwen2.5-vl/transformers/72b-instruct-awq/1` —
  **Qwen2.5-VL-72B-Instruct-AWQ, 43.0 GB**, 4-bit AWQ, **vision-language** (mandatory:
  the harness feeds an upscaled grid image; a text-only 72B would break the modality
  contract). Attach as a Kaggle Model source; no upload/download needed.
- **Estimated GPU-h:** ~**2.5 GPU-h per bench push**; the full screen (canary + scored
  bench + optional marginal seed) ≈ **~7.5 GPU-h**, inside 30/wk on its own but
  contends with A14/A15 (dependency flagged, not owned).
- **Null formula:** fixed-wallclock race → **N₇₂B = ⌊(1/ρ)·N₂₇B⌋**, ρ = measured
  tok/s(27B)/tok/s(72B) on the verified RTX PRO 6000 SKU; **null_adj(game)** = levels
  the W0 27B baseline had completed by action N₇₂B (cumulative walk of its
  `actions_per_level`). Sealed target **Σ null_adj = 4 (ρ≤2.5) / 3 (ρ≤3.0)**; 72B must
  beat it by the registered **+1-level margin**, inside the disjunctive GO boolean
  (capability ≥2-over-27B-MAX AND [action-parity OR throughput-adjusted+margin]).
- **Biggest risk:** the **serve-config mismatch (risk D)** — Qwen2.5-VL needs
  `--tool-call-parser hermes` and no qwen3 reasoning/thinking flags; shipping the 27B's
  `qwen3_coder`/`qwen3`/`preserve_thinking` flags silently breaks tool-call parsing →
  0 score with no crash. Runtime-test a tool-call round-trip before the scored push
  (`feedback_test_before_submit`). Close seconds: single-GPU memory fit for the 43 GB
  VL model (forces max-model-len 32768) and the mandatory no-touch on reset/deadline
  logic (the 9-min→1-hour community failure).
```

Sources consulted for the sm_120/AWQ risk assessment:
- vLLM issue #31085 (SM120 NVFP4 backend recognition)
- vLLM forums: SM120 RTX PRO 6000 / RTX PRO 4000 AWQ + Marlin reports
- lastloop-ai/vllm-blackwell-guide (Qwen on RTX PRO 6000 Blackwell + vLLM 0.19)
- vLLM quantization docs (AWQ → Marlin fallback behavior)
