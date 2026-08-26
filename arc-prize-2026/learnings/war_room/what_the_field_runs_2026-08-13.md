# What the field actually runs — ARC-AGI-3 brain census, 2026-08-13

**Order.** "Look at what people are doing / what they are using on Kaggle and deal with it."
Enumerate every public kernel, read its attached weights, join to the leaderboard, and say what to swap.

**Method (read-only, zero pushes, zero submissions, zero spend).**
- `kaggle kernels list --competition arc-prize-2026-arc-agi-3` × `{hotness, voteCount, dateRun, scoreDescending}` × pages 1–7
  → **670 unique public kernels** (the full public population; pagination exhausted).
- `kaggle kernels pull -m` on **370 targets** (every kernel run since 2026-07-01 **or** with ≥5 votes)
  → **363 `kernel-metadata.json` retrieved** (7 unrecoverable after 3 retries; all zero-vote, pre-July).
- `kaggle competitions leaderboard -d` → **full 2,263-team CSV with `TeamMemberUserNames`**
  (snapshot `2026-08-13T01:14:00Z`), joined to kernel owner slugs → **266 of 363 kernels attributed to a scored team**.
- `kaggle datasets list -s <28 model queries> --sort-by updated` (2 pages each) → **658 unique datasets**;
  cross-referenced owner slug against the LB user map.
- `kaggle models list -s <15 queries>` + `kaggle models get` on the official org refs.
- Config-level reads of the two candidate brains: `config.json`, `README.md`, `hf_quant_config.json`,
  `chat_template.jinja`, `preprocessor_config.json` pulled file-by-file (a few hundred KB total — no weight downloads).
- Local: `duck_eval/taaf_bundle/src/ARC3-Inference/inference/framework/kaggle.py` (the real vLLM launch line).

Provenance: **[V]** verified by direct read · **[INF]** inference · **[UNK]** unknown.

---

## 0. HEADLINE — read this if you read nothing else

**The 1.4+ public band serves exactly one brain: `Qwen3.6-27B-FP8`. The one we already run.** [V]

| brain | kernels | distinct teams | best LB by any team that published it | that team |
|---|---:|---:|---:|---|
| **Qwen3.6-27B-FP8** (`driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`) | **99** | **52** | **1.62** | Tufa Labs #7 |
| Gemma-4-31B-IT (`google/gemma-4/…/gemma-4-31b-it/1`) | 14 | 12 | 1.51 | The AGI Boys #22 |
| Qwen3-dense-small (1.7B/8B/14B, side-cars) | 7 | 2 | 1.28 | Chew Kok Wah #134 |
| Qwen3-VL-30B-A3B-Instruct-FP8 (`qwen-lm/qwen-3-vl/…`) | 4 | 2 | 1.26 | TREX99 #155 |
| Qwen2.5-7B / Qwen2.5-VL-7B | 3 | 1 | 1.18 | 100rabh #270 |
| Qwen3.6-35B-A3B-FP8 (`nareshmeena07/qwen36-35b-a3b-fp8`) | 2 | 2 | **1.04** | Kh0a #440 |
| Gemma-4-26B-A4B-NVFP4 ("xCaliber") | 2 | 2 | **0.17** | xCalibrr #1513 |
| GPT-OSS-120B (`danielhanchen/gpt-oss-120b`) | 2 | 2 | **0.16** | Greg Kamradt #1636 |
| **Qwen3.5-122B-A10B-NVFP4** | **0 kernels** | — | — | *(weights uploaded by ippeiogawa, #14 @ **1.58**)* |

Three facts fall out of that table and they are the whole report:

1. **Bigger has been tried and publicly lost.** GPT-OSS-120B → 0.16. Gemma-4-26B-A4B-NVFP4 → 0.17.
   Qwen3.6-**35B**-A3B → 1.04. Every public attempt to leave the 27B has scored *worse*, in most cases
   catastrophically. The VLM-swap premise is not just unsupported by the public record — it is **contradicted** by it. [V]
2. **The top of the board is dark.** Of the **top 40 teams, only 6 have any public kernel at all**
   (Tufa Labs #7, FOYSAL #9, Helmut AGI #11, The AGI Boys #22, 暗黑AGI #32, Tara Labs #37).
   **cstl (#1, 2.52) has zero public kernels and zero public datasets** on either handle. KOJIMA (#2, 1.86): zero.
   Andy liu (#3): zero. Lord Han Solo (#4): zero. BambooCopter (#5): zero. GeniusYY (#6): zero ARC artefacts. [V]
   **There is no public artefact that explains anything above 1.62.**
3. **One leading indicator exists, and only one.** `ippeiogawa/qwen35-122b-a10b-nvfp4` — 71.3 GB,
   uploaded **2026-07-17 09:58Z**, 56 downloads — owned by **ippeiogawa, #14 @ 1.58 (on the gold line)**.
   It is the **only** non-default brain publicly uploaded by anyone in the 1.5+ band, and ippeiogawa's
   **only** other dataset in existence is `arc2024ev-sol` (an ARC-AGI-1 solution). He has **zero** public
   kernels in this competition. [V] The intent is unmistakable; the result is unverifiable.

**Verdict on the order's part (e), stated plainly: nobody public is running a better brain than we are.**
We already serve the exact model that produced every public score in the 1.4–1.62 band. Any move here is
a bet placed *ahead* of the field, not a catch-up.

---

## (a) THE KERNEL → MODEL TABLE

Full generated table (all 132 brain-bearing kernels, LB-joined) is reproduced in **Appendix A** below.
Environment is uniform and worth stating once: **every** GPU kernel in this competition reports
`machine_shape: NvidiaRtxPro6000`, `enable_internet: false`, `docker_image` pinned to the
`gcr.io/kaggle-private-byod/python@sha256:57e612…` BYOD image. There is no machine-shape edge to find. [V]

The 1.4+ slice:

| LB | rank | team | kernel | model served | quant |
|---:|---:|---|---|---|---|
| **1.62** | 7 | Tufa Labs | `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` | Qwen3.6-27B | FP8 W8A8 |
| **1.62** | 7 | Tufa Labs | `jeroencottaar/taaf-duck-harness-kaggle` | Qwen3.6-27B | FP8 W8A8 |
| **1.61** | 11 | Helmut AGI | `jakobbrggen/taaf-anim-arc-agi-3-solver` | Qwen3.6-27B | FP8 W8A8 |
| **1.51** | 22 | The AGI Boys | `romantamrazov/arc-real-agi-solution` | Gemma-4-31B-IT | bf16 (Kaggle Models) |
| **1.47** | 32 | 暗黑AGI | `boristown/agi-duck-harness-fast-eval` | Qwen3.6-27B | FP8 W8A8 |
| **1.46** | 37 | Tara Labs | `caoyupeng/arc3-duck-v12-1d7d88` | Qwen3.6-27B | FP8 W8A8 |
| **1.44** | 46 | I forgot the name.. | `zoli800/taaf-duck-harness-kaggle-share-resubmission-573a60` | Qwen3.6-27B | FP8 W8A8 |

**Caveat that must ride with every row:** a Kaggle team score is the **max over all its submissions**,
so a brain in a public kernel is *associated with*, never *attributable to*, that team's number.
Tufa Labs' 1.62 came from 104 submissions; the published notebook is one of them.

**What our own brain actually is** (verified from `driessmit1/…/config.json` + `README.md`): [V]
`vrfai/Qwen3.6-27B-FP8`, `Qwen3_5ForConditionalGeneration`, `pipeline_tag: image-text-to-text`,
**a VLM** — vision tower depth 27 / hidden 1152 / patch 16, `image_token_id 248056`, hybrid attention
(`full_attention_interval: 4`, 64 layers, 4 KV heads, `head_dim` 256, mamba/linear-attn interleave),
262 K max positions, 35.92 GB single-shard safetensors, `compressed-tensors` W8A8.

**And how it is served** (`kaggle.py:306-340`): [V]
```
vllm.entrypoints.openai.api_server --model <path> --served-model-name vrfai/Qwen3.6-27B-FP8
  --tensor-parallel-size 1 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --generation-config vllm --enable-prefix-caching
  --default-chat-template-kwargs {"preserve_thinking": true} --reasoning-parser qwen3
  --max-model-len 65536
```
with `MULTIMODAL_CONTEXT` defaulting to `current_grid` on the Kaggle path (`kaggle.py:119`) —
so the harness **is** multimodal by default, and a text-only brain would break the observation contract.

---

## (b) MODEL-VS-SCORE CORRELATION — what the numbers say

Plotted honestly, the correlation is **negative for size and null for family**:

- **Qwen3.6-27B-FP8**: n=52 teams. Score range **0.00 → 1.62**, i.e. the model explains *nothing* —
  the whole spread from bottom to top of the public board runs the same weights. **Harness and agent
  policy are the entire public variance.** This is the single most important quantitative result here.
- **Gemma-4-31B-IT**: n=12 teams, best 1.51, second best 1.23. A viable but not superior alternative;
  it also breaks the `qwen3_coder` tool-parser contract.
- **Everything larger than 31B in the public record scores ≤ 1.04.** GPT-OSS-120B (0.16) and
  Gemma-4-26B-A4B-NVFP4 (0.17) are near-floor. This is consistent with the throughput wall we
  already measured on the 72B: on this rail, **tokens/sec buys score and parameters do not**.
- **MoE has been tried at the small end and did not help**: Qwen3-VL-30B-**A3B** (3B active, VL, official
  Kaggle model, 4 kernels by TREX99+Deniz Mucur) → best 1.26; Qwen3.6-35B-A3B → best 1.04.
  So "MoE = fast = better" is **not** established. What is different about the 122B-A10B is argued in (c)/(d).
- **Fine-tuning the 27B is the live public frontier, not swapping it.** `auxentr` (#163, 1.25) shipped
  `iseesmth/arc3-nca-qwen36-sft` + `duck-harness-nca-qwen36-adapter-20260811` on 08-11/08-12 — a
  **LoRA adapter on the duck's own Qwen3.6-27B**. That is where the public effort is going.

**Nothing correlates with 1.4+ except the duck default.** [V]

---

## (c) RANKED SHORTLIST — concretely servable brains for our harness

Envelope: one **RTX PRO 6000 Blackwell, ~95.6 GiB usable** (`machine_shape: NvidiaRtxPro6000`), SM120,
no internet, single GPU, weights must arrive as an attached Kaggle dataset or Kaggle Model.

### 1. **Qwen3.5-122B-A10B-NVFP4** — `ippeiogawa/qwen35-122b-a10b-nvfp4` ⭐ **RECOMMENDED**

| property | value | source |
|---|---|---|
| HF origin | `nvidia/Qwen3.5-122B-A10B-NVFP4` (base `Qwen/Qwen3.5-122B-A10B`), Apache-2.0, released 06/01/2026 | README [V] |
| Kaggle path | `ippeiogawa/qwen35-122b-a10b-nvfp4` (public dataset, 07-17, 56 dl) | [V] |
| size at quant | 9 shards, **83.50 GB on disk (77.76 GiB)**; dataset shows 71.31 GB compressed | file listing [V] |
| fit in 96 GB @32k | **YES, ~17.8 GiB headroom.** `full_attention_interval 4` → only **12 of 48 layers** carry a KV cache; `num_key_value_heads 2`, `head_dim 256`, and `hf_quant_config` sets **`kv_cache_quant_algo: FP8`** → **KV ≈ 0.40 GB at 32 k** (0.81 GB if fp16). Linear-attn recurrent state is O(1) per sequence (~0.15 GB/seq). | config arithmetic [V]/[INF] |
| modality | **VL: text + image + video.** `vision_config` depth 27 / hidden 1152 / patch 16 / merge 2, `image_token_id 248056`, `processor_class: Qwen3VLProcessor` — **byte-identical preprocessing to our 27B** (same mean/std 0.5, same `longest_edge 16777216 / shortest_edge 65536`). | [V] |
| native tool-calling under vLLM | **YES, existing parser works unchanged.** `chat_template.jinja` is **7,756 B vs our 27B's 7,764 B and diffs in exactly 2 lines** (see below). Tool-call block syntax is identical → `--tool-call-parser qwen3_coder --reasoning-parser qwen3` carry over. | direct diff [V] |
| architecture | `qwen3_5_moe`, 256 experts / top-8, 48 layers, hidden 3072, **122B total / 10B active**, 262 K ctx, NVFP4 group-16 ModelOpt | [V] |
| **evidence someone scored well with it** | **Circumstantial but the strongest available.** Uploaded 2026-07-17 by **ippeiogawa (#14, LB 1.58, 54 subs)** — a gold-line team whose only other Kaggle dataset is an ARC-AGI-1 solution, who has never published an ARC-AGI-3 kernel. **No public kernel serves it. No score is attributable.** Our earliest local LB observation of ippeiogawa is 1.58 on 07-27 — *after* the upload — so we cannot even establish a before/after step. | [V] on facts, **[UNK]** on causation |

**The two-line chat-template diff (`ct27` → `ct122`), verbatim:**
```
100c100
<   {%- if (preserve_thinking is defined and preserve_thinking is true) or (loop.index0 > ns.last_query_index) %}
---
>   {%- if loop.index0 > ns.last_query_index %}
122c122
<   {%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}
---
>   {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
```
Line 100 matters: the stock 122B template has **no `preserve_thinking` branch**, so the harness's
`--default-chat-template-kwargs '{"preserve_thinking": true}'` would **silently no-op** and prior-turn
reasoning would be stripped — a behaviour change disguised as a config no-op. Fix is one line
(or `--chat-template` pointed at the 27B's file, which is otherwise identical). Line 122 is the
122B being *more* correct about tool-arg serialisation. **Both are free.**

**Why this one is different from the 72B that died.** The 72B-dense-AWQ failed on decode bandwidth:
~72 B params × 0.5 B/param = **~36 GB read per token**, vs our 27B-FP8's ~27 GB — *worse* than the
incumbent, hence 26–33 actions/window. The 122B-A10B-NVFP4 activates **10 B params at 4 bits ≈ 5 GB
per token — roughly 5× less traffic than the model we run today**, on a Blackwell card with native
FP4 tensor cores. **On the arithmetic this should be faster than the incumbent, not slower**, while
carrying 122B of stored knowledge. That is the only candidate in the census where the capability
axis and the throughput axis point the same way. [INF, from config + first-principles bandwidth]

**Risks, named:** (i) **NVFP4 on SM120** — vLLM issue #31085, backend recognition; mitigated below but
unverified on this rail; (ii) 83.5 GB dataset attach + mount time inside an 11 h budget; (iii) total
attached-input size limits on Kaggle kernels **[UNK]** — must be checked at gate 1; (iv) FlashInfer
SM120 JIT compile on cold start; (v) `gpu_memory_utilization` must be raised to ~0.93–0.95, which
removes the usual safety margin.

### 2. **Qwen3.6-27B-NVFP4** — `vrfai/Qwen3.6-27B-NVFP4` (HF; **not yet on Kaggle**)
Named in our own 27B's README as the Blackwell-only sibling of the exact weights we serve. [V]
**Same brain, ~18 GB instead of 36 GB, FP4 tensor cores.** Zero capability change; pure throughput —
which is precisely what an efficiency cap of 1.26–1.36 is made of. Requires us to mirror it into a
Kaggle dataset (proven route: `canivel/qwen25-vl-72b-awq`). **This is the low-risk fallback if #1's
NVFP4 path fails to come up** — it isolates "does NVFP4/SM120 work on this rail" from "does a bigger
brain help", using weights whose behaviour we have 100+ submissions of history on.
Corroboration that NVFP4 is a live route: **`jcole75/arc3-qwen36-runtime-wheels`** (see (d)) exists.

### 3. **Qwen3-VL-30B-A3B-Instruct-FP8** — `qwen-lm/qwen-3-vl/Transformers/30b-a3b-instruct-fp8/1`
Official Kaggle Model (no upload needed), VL, MoE 3B active, ~32 GB, trivially fits.
**Already tried publicly**: TREX99 (`arc3-duck-qwen3vl{2,3,4}`, 07-06) and Deniz Mucur (08-06).
Best associated team score **1.26**, and Deniz Mucur's team sits at 0.84. Prior generation to Qwen3.6.
**Cheapest possible screen** (one metadata line, no dataset upload) but the public evidence is *against* it.

### 4. **Gemma-4-31B-IT** — `google/gemma-4/Transformers/gemma-4-31b-it/1`
Best non-Qwen public brain (1.51, The AGI Boys #22; 14 kernels / 12 teams). Fits easily.
**But**: different chat template ⇒ `--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3` both
break; the whole tool contract must be re-derived. Ceiling observed is *below* the 27B's 1.62.
**Not worth 82 days of remaining schedule.**

### 5–7. **Rejected on public evidence**
- **Qwen3.6-35B-A3B-FP8** (`nareshmeena07/qwen36-35b-a3b-fp8`, `cmechevalier/face-of-agi-qwen36-35b-fp8-weights`,
  30.7 GB, 157 dl): best associated 1.04 / 0.70. Our 08-04 sweep records it as **text-only** — would
  force `MULTIMODAL_CONTEXT` off and change the harness contract. **Reject.**
- **GPT-OSS-120B** (`danielhanchen/gpt-oss-120b`): 0.16 / 0.15 across two teams. **Reject.**
- **Gemma-4-26B-A4B-NVFP4** (`pranshubahadur/gemma-4-26b-a4b-nvfp4`): 0.17 / 0.08. **Reject.**

### Explicitly checked and **NOT PRESENT** in a servable form
Per the order's named list, sweeping `kaggle datasets list` (658 unique hits) + `kaggle models list`:
- **GLM-4.6V** — nothing. (`glm-4.6v` and `glm4v` return zero rows. Only `jakiproton/glm52-colibri-part-{1..4}`,
  4×37 GB, 07-14/15, no ARC link and no LB owner.)
- **Qwen3.5/3.6-VL variants** — only the 122B-A10B-NVFP4 above, plus GGUF mirrors
  (`iyppmx/qwen3-vl-30b-a3b-instruct-gguf`, `micronic/qwen3-6-35b-a3b-gguf`, …). **GGUF is not vLLM-servable
  on this rail** — the harness is a hard vLLM OpenAI-server dependency.
- **Gemma-4** — present and used (31B-IT ×14 kernels), covered above. The 08-07/08-11 uploads
  (`spiritofvishwakarma/gemma4-{31b,26b-a4b,12b}-q4-with-mmprojq8-gguf`) are **GGUF/llama.cpp**, owner not on the LB.
- **gpt-oss-120b** — present, scores 0.15–0.16.
- **Mistral VL** — nothing beyond `mistral-ai/mistral/PyTorch/7b-instruct-v0.1-hf/1` (2019-era, one kernel).
- **Kimi VL** — only `sergiodefreitas/kimi-vl-a3b-thinking-2506` (10.3 GB, 07-26, 1 dl, owner not on LB)
  and `helium990/kimi-k3` (2.78 T MoE MXFP4 mirror — **wildly outside a single 96 GB card**).
- **DeepSeek VL** — nothing. (`ravi123a321at/deepseek-v4-flash-0731-iq2m-shard2`, 49.6 GB, 08-10, owner
  **Ravi's Agi #61 @ 1.39** — *is* an LB competitor, but it is an **IQ2_M GGUF shard 2 of N**, i.e. llama.cpp,
  and a 2-bit quant. Not servable here. Worth one line in the next sweep, no more.)
- **InternVL** — `khaledchenguel/opengvlabinternvl3-5-241b-a28b` (241B-A28B, no quant stated, no ARC link)
  and 8B/1B toys. Nothing in-envelope with evidence.
- **MiniCPM-V** — `jaccojurg/minicpm-v-46`, `kawchar85/minicpm-v4-5-awq` etc., all ≤ 8B. Too small.

---

## (d) THE ONE RECOMMENDED SWAP + EXACT SCREEN

> ### Swap `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` → `ippeiogawa/qwen35-122b-a10b-nvfp4`, served on `jcole75/arc3-qwen36-runtime-wheels`.

**The second half of that sentence is the real find of this sweep.** The duck's pinned wheelhouse
`driessmit1/arc3-vllm-h100-wheelhouse-v3` (README, [V]) is **`vllm==0.19.0, torch==2.10.0,
transformers==4.57.6, flashinfer 0.6.6, CUDA 12.8`** — an **H100** build, with **no ModelOpt**.
It cannot load an NVFP4/ModelOpt checkpoint. But a public, purpose-built replacement already exists:

**`jcole75/arc3-qwen36-runtime-wheels`** — 6.45 GB, 79 downloads, uploaded 2026-07-10 by **Jack Cole (#89, 1.33)**.
Its `requirements-runtime.txt`, quoted verbatim: [V]
```
# Built for Kaggle's ARC-AGI-3 Python 3.12 / RTX PRO 6000 Blackwell runtime.
vllm==0.24.0
transformers==5.13.0
nvidia-modelopt==0.45.0
# FlashInfer JITs SM120 kernels on Kaggle. Keep the packaged CUDA 13 compiler,
# headers, runtime, and cuRAND headers on one coherent 13.3 toolchain.
nvidia-cuda-nvcc==13.3.73 / nvidia-cuda-runtime==13.3.29 / nvidia-cuda-cccl==13.3.3.4.1
nvidia-cuda-nvrtc==13.3.33 / nvidia-curand==10.4.3.29 / nvidia-cublas==13.3.0.5
```
`nvidia-modelopt==0.45.0` is **exactly** the dependency an `hf_quant_config.json` with
`{"quant_algo":"NVFP4","quant_method":"modelopt"}` requires, and the comment names the SM120 JIT problem
by name. **Both halves of the swap are public, free, and already built by other people.**
(Honest note: Jack Cole is at 1.33 — the same score as us. His wheelhouse is *capability*, not *evidence of gain*.)

**Screen plan — free Kaggle BUILD rail only, no submission, no spend.**

**Step 0 — do not build from scratch.** Fork the frozen `arc3-baseline.ipynb` and run
`scripts/preflight.py`. (`feedback_arc_kernel_structural_drift`: 5 ERRORs, all from hand-built kernels.)
Use a **fresh kernel slug** (`feedback_fresh_kernel_slug`).

**Step 1 — `kernel-metadata.json`.** Keep `enable_gpu: true`, `enable_internet: false`,
`machine_shape: NvidiaRtxPro6000`, and the **same pinned `docker_image` sha** as the duck baseline
(`feedback_kaggle_env_match` — 5× confirmed). Set:
```json
"dataset_sources": [
  "jcole75/arc3-qwen36-runtime-wheels",
  "ippeiogawa/qwen35-122b-a10b-nvfp4",
  "<our taaf source-share dataset>"
]
```
**Remove** `driessmit1/arc3-vllm-h100-wheelhouse-v3` — mixing two wheelhouses is how you get an
unreproducible resolve. Push the source dataset separately and check the runtime banner
(`feedback_kaggle_dataset_code_sync`).

**Step 2 — `DuckKaggleConfig` (`inference/framework/kaggle.py:53-58`).**
`wheelhouse_dataset_source = "jcole75/arc3-qwen36-runtime-wheels"`,
`model_dataset_source = "ippeiogawa/qwen35-122b-a10b-nvfp4"`,
`served_model_name = "Qwen/Qwen3.5-122B-A10B"`, `max_model_len = 32768`, `tensor_parallel_size = 1`.
The wheelhouse install path differs (`wheels/` subdir + `requirements-runtime.txt`, not
`requirements.lock` at root) — patch `install_vllm_wheelhouse()` accordingly.

**Step 3 — chat template.** Ship the 27B's `chat_template.jinja` alongside and pass
`--chat-template <path>`, **or** patch line 100 of the 122B template to restore the
`preserve_thinking` branch. Do **not** skip this: without it the `preserve_thinking` kwarg is a silent no-op
and every prior-turn reasoning block is stripped — you would be A/B-ing two changes at once.

**Step 4 — vLLM args.** Keep `--enable-auto-tool-choice --tool-call-parser qwen3_coder
--reasoning-parser qwen3 --enable-prefix-caching --generation-config vllm`. Add:
`--quantization modelopt_fp4` **explicitly** (`a17_72b_screen_scope.md:301` — never let vLLM guess the
quant), `--kv-cache-dtype fp8` (the checkpoint declares it), `--gpu-memory-utilization 0.93`,
`--max-model-len 32768`, `--limit-mm-per-prompt '{"image":1}'`.
`MULTIMODAL_CONTEXT=current_grid`, `MULTIMODAL_UPSCALE=4` unchanged.

**Step 5 — GATE 1 (build only, ~15 min of the 30 GPU-h/wk allowance).** Does vLLM come up and pass
`run_vllm_api_smoke_test()`? Capture `vllm-openai-server.log` as the evidence artefact.
**KILL if:** NVFP4/SM120 backend unrecognised; OOM at 0.93; weights load > 45 min; total attached
input rejected by Kaggle. *On kill → fall back to shortlist #2 (`Qwen3.6-27B-NVFP4`), which isolates
the NVFP4/SM120 question on weights we already understand.*

**Step 6 — GATE 2 (the one that killed the 72B).** Measure **actions per 11 h window** on the 25 offline
environment files. **Hard bar: ≥ 100.** The 72B managed 26–33. Prediction from bandwidth arithmetic:
this should clear the incumbent, not merely the bar. **KILL if < 100.**

**Step 7 — GATE 3.** Local harness score vs the frozen fork, same seeds, `runs/null10` reference,
`scripts/phase1_gate.py`. Only then does it enter the submission queue.

**Cost: zero.** Both datasets are public; all three gates are Kaggle kernel *builds* on the free
30 GPU-h/wk RTX PRO 6000 allowance (`feedback_arc_zero_budget`).

---

## (e) THE HONEST STATEMENT

**Nobody public is running anything better than we are.**

We serve `Qwen3.6-27B-FP8`. So do 52 of the 2,263 teams who published a kernel, including **every
public team in the 1.4–1.62 band**. The same weights produce scores from 0.00 to 1.62, which means
**the public variance is entirely harness and agent policy, and none of it is the model.** Every public
attempt to serve something bigger has scored worse — 0.16 (GPT-OSS-120B), 0.17 (Gemma-4-26B-A4B-NVFP4),
1.04 (Qwen3.6-35B-A3B), 1.26 (Qwen3-VL-30B-A3B). The 72B result we produced ourselves is the fourth
data point in that pattern, not an anomaly.

**A model swap is therefore a speculative bet, not a catch-up move**, and it should be labelled as such
in the ledger. The census supports exactly one such bet — Qwen3.5-122B-A10B-NVFP4 — and it supports it
on **mechanism** (10 B active at 4 bits reads ~5× less memory per token than the 27B-FP8 we run; VL
preserved; chat template and tool parser identical; fits 96 GiB with 17.8 GiB spare) plus **one
circumstantial fact** (the only gold-line team to publicly upload a non-default brain uploaded this one).
It is **not** supported by any observed score. If gate 2 does not clear 100 actions/window, the bet is
dead in one build and costs nothing.

**And the census says something else that is worth more than the swap:** the 2.52 at #1 has **no public
explanation whatsoever**, and the public frontier of effort has moved to **fine-tuning the duck's own
27B** (`auxentr`'s `arc3-nca-qwen36-sft` LoRA adapter, shipped 08-11/08-12). Given four independent
negatives on the 27B's *instruction-following* and a public record where the model explains none of the
variance, **adapting the 27B is better-aligned with the evidence than replacing it.** Recommend the 122B
screen because it is cheap, gated, and falsifiable in one build — not because the field says it works.

---

## Appendix A — full brain-bearing kernel table (132 rows)

| LB | rank | team | kernel | last run | votes | model served | machine |
|---:|---:|---|---|---|---:|---|---|
| **1.62** | 7 | Tufa Labs | `jeroencottaar/taaf-duck-harness-kaggle` | 2026-06-30 | 238 | **Qwen3.6-27B-FP8** | None |
| **1.62** | 7 | Tufa Labs | `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` | 2026-07-01 | 268 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.61** | 11 | Helmut AGI | `jakobbrggen/taaf-anim-arc-agi-3-solver` | 2026-08-07 | 73 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.51** | 22 | The AGI Boys | `romantamrazov/arc-real-agi-solution` | 2026-06-30 | 22 | **Gemma-4-31B-IT** | NvidiaRtxPro6000 |
| **1.47** | 32 | 暗黑AGI | `boristown/agi-duck-harness-fast-eval` | 2026-07-22 | 262 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.47** | 32 | 暗黑AGI | `boristown/taaf-duck-harness-kaggle` | 2026-07-08 | 8 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.46** | 37 | Tara Labs | `caoyupeng/1-21-from-great-team-tufa-labs` | 2026-07-01 | 48 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.46** | 37 | Tara Labs | `caoyupeng/arc3-duck-v12-1d7d88` | 2026-07-23 | 64 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.44** | 46 | I forgot the name.. | `zoli800/taaf-duck-harness-kaggle-share-resubmission-573a60` | 2026-07-23 | 7 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.38** | 68 | Nabid Nur | `nabidnur/notebookd444f158e0` | 2026-07-20 | 19 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.38** | 67 | Rokaiya Somapti | `rokaiyasomapti/taaf-duck-harness-kaggle-share-resubmission` | 2026-07-05 | 149 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.33** | 87 | Samrish B | `samrishb/just-resubmission-rn-working-on-experiments` | 2026-07-15 | 59 | **Qwen3.6-27B-FP8** | None |
| **1.33** | 88 | Tanaka Ai24 | `tanakaai24/arc3-qwen3-6-duck-compact-trajectory-v1` | 2026-08-07 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.33** | 88 | Tanaka Ai24 | `tanakaai24/arc3-qwen3-6-duck-full-harness-v1` | 2026-08-08 | 2 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.33** | 88 | Tanaka Ai24 | `tanakaai24/arc3-qwen3-6-duck-lb117-safety-v1` | 2026-08-09 | 16 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.30** | 121 | Emre Cirak | `biohack44/1-07-great-team-tufa-labs` | 2026-07-06 | 4 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.29** | 124 | yw8837 | `yw8837/lb-1-17-arc-agi-3-qwen3-6-duck-full-code` | 2026-07-27 | 24 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.28** | 134 | Chew Kok Wah | `chewkokwahibrainai/tufa-labs-duck-harness-june-30-milestone-winner` | 2026-07-04 | 5 | **Qwen3-dense-small + Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.28** | 137 | FinalSunFlower | `finalsunflower/arc3-anim-lb161-exact-validation` | 2026-08-11 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.28** | 135 | Beyond Good and Eval | `thtennant/arc3-duck-v12` | 2026-08-09 | 40 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.28** | 135 | Beyond Good and Eval | `thtennant/arc3-duck-v18` | 2026-08-11 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.28** | 139 | Toprak Gundogdu | `toprakg/taaf-duck-prolong-memory` | 2026-08-04 | 2 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.27** | 142 | maxingkong733 | `maxingkong733/arc3-duck-dead-signature` | 2026-07-09 | 3 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.27** | 142 | maxingkong733 | `maxingkong733/arc3-duck-v7-submit` | 2026-07-08 | 3 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.26** | 155 | TREX99 | `trex99/arc3-duck-qwen3vl2` | 2026-07-06 | 3 | **Qwen3-VL-30B-A3B-FP8** | NvidiaRtxPro6000 |
| **1.26** | 155 | TREX99 | `trex99/arc3-duck-qwen3vl3` | 2026-07-06 | 1 | **Qwen3-VL-30B-A3B-FP8** | NvidiaRtxPro6000 |
| **1.26** | 155 | TREX99 | `trex99/arc3-duck-qwen3vl4` | 2026-07-06 | 0 | **Qwen3-VL-30B-A3B-FP8** | NvidiaRtxPro6000 |
| **1.26** | 155 | TREX99 | `trex99/arc3-duck-v2` | 2026-07-04 | 17 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.26** | 155 | TREX99 | `trex99/arc3-duck-v4` | 2026-07-07 | 1 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.26** | 155 | TREX99 | `trex99/arc3-duck-winner-prevframe` | 2026-07-03 | 7 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.25** | 163 | auxentr | `iseesmth/arc3-nca-sft-duck-public` | 2026-08-12 | 0 | **Qwen3.6-27B + NCA LoRA (SFT) + Qwen3.6-27B-FP8** | None |
| **1.25** | 163 | auxentr | `iseesmth/duck-harness-prolong-public-eval` | 2026-08-11 | 4 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.25** | 163 | auxentr | `iseesmth/prolong-eval` | 2026-08-11 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.23** | 187 | @🤞@ | `anglolodorf/fork-of-taaf-duck-harness-grammar` | 2026-07-09 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.23** | 187 | @🤞@ | `anglolodorf/taaf-duck-harness-grammar` | 2026-07-09 | 4 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.23** | 189 | Shivangoudaa | `ishivapatil/arc-agi-3-reki-optimized` | 2026-08-11 | 1 | **Gemma-4-31B-IT** | None |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/chimpanzee-1-1-anim` | 2026-08-12 | 1 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/chimpanzee-1-1-eval` | 2026-08-10 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/chimpanzee-1-1-eval-visible-updates` | 2026-08-12 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/gorilla-1-1-eval` | 2026-08-08 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/gorilla-1-1-scl-eval` | 2026-08-08 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/gorilla-1-1-sp` | 2026-08-06 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/gorilla-eval-new` | 2026-08-05 | 1 | **Qwen3.6-27B-FP8** | None |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/sandwich` | 2026-08-05 | 16 | **Qwen3.6-27B-FP8** | None |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/tiger` | 2026-08-08 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/tufa-duck-visible-updates` | 2026-08-12 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.22** | 197 | Jason Feng | `iamjasonfeng/wles-wltd-mrps` | 2026-08-05 | 0 | **Qwen3.6-27B-FP8** | Gpu |
| **1.20** | 236 | KAIWALYA RAUT | `kaiwalyaatulraut/arc-agi-3-solution` | 2026-08-06 | 20 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.20** | 236 | KAIWALYA RAUT | `kaiwalyaatulraut/notebook6fc82669d5` | 2026-07-25 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.19** | 255 | Jason Jin | `cascadematrix/arc-agi-3-capacity-recovery-v1` | 2026-08-12 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.19** | 255 | Jason Jin | `cascadematrix/arc-agi-3-causal-animation-v1` | 2026-08-11 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.18** | 270 | 100rabh | `saurabhkumar234/tough-guard-v2` | 2026-08-03 | 9 | **Qwen2.5-7B/VL + Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.17** | 275 | Julian Camilo Villa | `juliancamilovilla/arc-agi3-duck` | 2026-08-11 | 6 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.17** | 275 | Julian Camilo Villa | `juliancamilovilla/arc-agi3-llm` | 2026-08-11 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.17** | 279 | Kunal Desale | `kunaldesale2408/duck-harness-fast-eval` | 2026-08-11 | 6 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.16** | 285 | haodou092 | `haodou092/feedback-guided-arc-agent` | 2026-07-10 | 4 | **Gemma-4-31B-IT** | NvidiaRtxPro6000 |
| **1.16** | 296 | Sushanth Tiruvaipati | `sushanthtiruvaipati/arc3-duck-lb117-fork` | 2026-08-09 | 0 | **Gemma-4-31B-IT** | Gpu |
| **1.16** | 296 | Sushanth Tiruvaipati | `sushanthtiruvaipati/arc3-taaf-qwen3-v2-no-patches` | 2026-08-13 | 1 | **Qwen3.6-27B-FP8** | None |
| **1.15** | 305 | Md Boktiar Mahbub Murad | `mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone` | 2026-06-30 | 97 | **Gemma-4-31B-IT** | NvidiaRtxPro6000 |
| **1.14** | 313 | Felipe Galdino Folk | `felipegaldinofolk/arcagi3-020-intrinsic-duck` | 2026-07-05 | 0 | **Qwen3.6-27B-FP8** | Gpu |
| **1.14** | 313 | Felipe Galdino Folk | `felipegaldinofolk/capivara-1` | 2026-07-05 | 7 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.14** | 310 | Celestia | `gladoscc/celestia-depthmap` | 2026-07-22 | 2 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.14** | 310 | Celestia | `gladoscc/celestia-duck-literal` | 2026-07-21 | 2 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.10** | 349 | Manan Gupta #2 | `obirdy/arc3-duck-known-score-baseline` | 2026-07-19 | 3 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.10** | 349 | Manan Gupta #2 | `obirdy/arc3-duck-state-ledger-gpu-jul26-a2` | 2026-07-26 | 1 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.10** | 349 | Manan Gupta #2 | `obirdy/arc3-duck-verified-world-model-cpu-commit` | 2026-07-20 | 2 | **Qwen3.6-27B-FP8** | None |
| **1.10** | 349 | Manan Gupta #2 | `obirdy/arc3-duck-verified-world-model-gpu-v1` | 2026-07-25 | 5 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.10** | 342 | ocean240812 | `ocean240812/arc3-v12-tufa-milestone-winner-fork` | 2026-08-10 | 2 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.10** | 354 | zhangzhuang li | `zhangzhuangli/arc3-exp001-qwen3-6-duck-safety` | 2026-08-10 | 1 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.05** | 414 | Liang Wan Yiu David  | `liangwanyiudavid/cognitive-duck-harness` | 2026-07-28 | 1 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.05** | 420 | Reki | `ruichardliu/milestone1-2nd-solution` | 2026-06-30 | 54 | **Gemma-4-31B-IT** | None |
| **1.04** | 430 | Göktürk Akman | `gktrkakman/taaf-duck-sub-20260805-share` | 2026-08-05 | 6 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.04** | 430 | Göktürk Akman | `gktrkakman/taaf-subv1x-share` | 2026-08-07 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.04** | 440 | Kh0a | `llkh0a/tufa-labs-duck-harness-but-gemma4` | 2026-07-05 | 1 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.04** | 440 | Kh0a | `llkh0a/tufa-labs-duck-harness-but-qwen-35b` | 2026-07-02 | 2 | **Qwen3.6-27B-FP8 + Qwen3.6-35B-A3B-FP8** | NvidiaRtxPro6000 |
| **1.03** | 450 | Nikita #2 | `nikitagajbhiye30/arc-agi-3-00` | 2026-08-10 | 5 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.01** | 472 | Keven Li | `kevin250304/arc3-duck-minimal-action7-reproducible` | 2026-07-18 | 3 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.01** | 472 | Keven Li | `kevin250304/arc3-duck-v9b-recovery-banking` | 2026-07-12 | 5 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.00** | 488 | Krizsó Gergely | `lucifer19/arc-agi-3-blackcat-worldmodel` | 2026-08-08 | 3 | **Qwen3-4B-4bit** | Gpu |
| **1.00** | 488 | Krizsó Gergely | `lucifer19/arc-agi-3-duck-v13-causal-scientist` | 2026-08-10 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **1.00** | 490 | Melisakman | `melisakman/taaf-subv1x-share` | 2026-08-07 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.99** | 493 | J&F AI | `junjin2/tufa-duck-original-fastsubmit` | 2026-07-10 | 6 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.97** | 517 | BBK | `bang1850/arc-agi-3-deep-reasoning-agent-179-183-levels` | 2026-07-24 | 2 | **Qwen3.6-27B-FP8** | Gpu |
| **0.97** | 517 | BBK | `bang1850/arc-agi-3-duck-balanced-ternary-10d-int4sp-v6` | 2026-07-21 | 5 | **Qwen3.6-27B-FP8** | Gpu |
| **0.97** | 517 | BBK | `bang1850/arc-agi-3-duck-reliability-v2-lean-token` | 2026-07-12 | 0 | **Qwen3.6-27B-FP8** | Gpu |
| **0.97** | 517 | BBK | `bang1850/arc-agi-3-ultimate-agent-phase-b` | 2026-08-09 | 1 | **Qwen3.6-27B-FP8** | Gpu |
| **0.97** | 517 | BBK | `bang1850/arc-agi-3-v20-multimodal` | 2026-08-03 | 0 | **Qwen3.6-27B-FP8** | Gpu |
| **0.97** | 517 | BBK | `bang1850/arc-agi-3-v21-duck-harness` | 2026-08-09 | 2 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.97** | 517 | BBK | `bang1850/arc3-hybrid-v19-ultra-low-vram-causal-fusion` | 2026-07-18 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.97** | 517 | BBK | `bang1850/arc3-hybrid-v20-verified-causal-fusion` | 2026-07-18 | 0 | **Qwen3.6-27B-FP8** | Gpu |
| **0.97** | 517 | BBK | `bang1850/arc3-hybrid-v21-pure-algorithmic` | 2026-07-18 | 0 | **Qwen3.6-27B-FP8** | Gpu |
| **0.97** | 517 | BBK | `bang1850/arc3-hybrid-v22-pure-algo-fixed` | 2026-07-19 | 0 | **Qwen3.6-27B-FP8** | Gpu |
| **0.97** | 517 | BBK | `bang1850/arc3-hybrid-v23-dict-fix` | 2026-07-19 | 0 | **Qwen3.6-27B-FP8** | Gpu |
| **0.96** | 532 | taopy2 | `taopy2/ag3-009-tufa-milestone-repro` | 2026-07-15 | 3 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.94** | 544 | Aman Atar | `amanatar/agi-duck-harness-fast-eval` | 2026-08-02 | 5 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.92** | 567 | Quanyi Li | `ko0kip/arc-agi-3-gemma-4-31b-reflection-agent` | 2026-07-06 | 81 | **Gemma-4-31B-IT** | NvidiaRtxPro6000 |
| **0.89** | 592 | Dalton Omondi | `daltongabrielomondi/arc-agi-3-competition-agent` | 2026-07-04 | 3 | **Gemma-4-12B-IT + Gemma-4-31B-IT** | None |
| **0.87** | 612 | Ozan AKMAN | `ozanakman/taaf-subv1x-share` | 2026-08-07 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.84** | 633 | Deniz Mucur | `denizmucur/arc-agi-3-baba` | 2026-08-06 | 2 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.84** | 633 | Deniz Mucur | `denizmucur/arc-agi-3-group-1` | 2026-08-06 | 4 | **Qwen3-VL-30B-A3B-FP8** | NvidiaRtxPro6000 |
| **0.81** | 647 | Hyunsik Park | `hyunsikpark/arc3-verified-offline-duck` | 2026-08-11 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.79** | 661 | Shreyas Mahimkar | `shreyas4/arc3-duck-v12` | 2026-07-13 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.72** | 694 | Chris Paul Walker | `chrispaulwalker/g1b-gemma-4-31b-murad-smoke-m025` | 2026-07-11 | 6 | **Gemma-4-31B-IT** | NvidiaRtxPro6000 |
| **0.72** | 694 | Chris Paul Walker | `chrispaulwalker/taaf-duck-share-probe` | 2026-07-06 | 4 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.70** | 701 | face-of-agi | `cmechevalier/face-of-agi-arc-agi-3-rtx6000` | 2026-06-30 | 5 | **Qwen3.6-35B-A3B-FP8** | NvidiaRtxPro6000 |
| **0.67** | 711 | prvsiyan | `prvsiyan/arc-agi-3-duck-decision-ledger-action7` | 2026-07-26 | 1 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.67** | 711 | prvsiyan | `prvsiyan/arc-agi-3-stock-taaf-action7-shadow` | 2026-08-09 | 4 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.50** | 753 | Emrullah Söyler | `emrullahsyler/arc-agi-ben-m` | 2026-08-04 | 0 | **Gemma-4-31B-IT** | NvidiaRtxPro6000 |
| **0.48** | 759 | Deniz Eryılmaz | `denizeryilmaz/arc3-gemma31b-murad-reproduction-v1` | 2026-08-10 | 0 | **Gemma-4-31B-IT** | NvidiaRtxPro6000 |
| **0.48** | 759 | Deniz Eryılmaz | `denizeryilmaz/arc3-reki-hybrid-explorer-v2` | 2026-08-10 | 0 | **Gemma-4-31B-IT** | NvidiaRtxPro6000 |
| **0.44** | 778 | Sonnia AI | `drozza/sonnia-concurrency-bench-v2` | 2026-07-07 | 0 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.34** | 868 | Matthew Blake Ward | `matthewblakeward/glyphmatics-sigilagi-highscore-solver` | 2026-07-17 | 0 | **Gemma-4-31B-IT** | Gpu |
| **0.34** | 859 | Nine1Eight | `wethepeople918/agi-duck-harness-fast-eval-b25-score-optimized` | 2026-07-26 | 1 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.34** | 859 | Nine1Eight | `wethepeople918/arc3-crow` | 2026-07-13 | 1 | **Qwen3.6-27B-FP8** | NvidiaTeslaT4 |
| **0.23** | 1226 | Yousef Turk | `yousefturk/fluidmind-arc-agi-3-agent` | 2026-07-09 | 0 | **Qwen3.6-27B-FP8** | Gpu |
| **0.22** | 1252 | Isaiah Nwukor | `thezetaproject/theorycoder-rlvr-arc-agi-3` | 2026-04-30 | 10 | **Qwen3-dense-small** | NvidiaTeslaT4 |
| **0.17** | 1513 | xCalibrr | `pranshubahadur/xcaliber-aa3-nvidia-gemma-4-26b-nvfp4` | 2026-08-04 | 8 | **Gemma-4-26B-A4B-NVFP4** | NvidiaRtxPro6000 |
| **0.17** | 1510 | Vladimir Yakunin | `vladimiryakunin/arc-prize-2026-lcld-qwen` | 2026-07-15 | 0 | **Qwen3.6-27B-FP8** | None |
| **0.17** | 1510 | Vladimir Yakunin | `vladimiryakunin/arc-prize-2026-lcld-qwen-v9` | 2026-08-12 | 1 | **Qwen3.6-27B-FP8** | NvidiaRtxPro6000 |
| **0.16** | 1567 | Juan Lavieri Goiri | `juanlavierigoiri/evidloom-arc-agi-3-llm` | 2026-07-01 | 1 | **GPT-OSS-120B** | NvidiaRtxPro6000 |
| **0.15** | 1636 | Greg Kamradt | `gregkamradt/arc-agi-3-gpt-oss-120b` | 2026-06-17 | 36 | **GPT-OSS-120B** | None |
| **0.08** | 1923 | orangger | `orangger/arc-agi-3-xcaliber-gemma-4-baseline` | 2026-07-30 | 0 | **Gemma-4-26B-A4B-NVFP4** | NvidiaRtxPro6000 |
| — | 2240 | NGABO FREDDY | `ngabofreddy/arc-prize-2026-arc-agi-3-starter` | 2026-07-20 | 2 | **Gemma-4-31B-IT** | None |
| — | — | (not on LB) | `ahmedmohamedfergany/barbados-2` | 2026-08-12 | 2 | **Qwen2.5-7B/VL** | NvidiaRtxPro6000 |
| — | — | (not on LB) | `arkadymaximov/tufa-labs-arc-agi-kam1k` | 2026-07-06 | 1 | **Qwen3.6-27B-FP8** | Gpu |
| — | — | (not on LB) | `artnaweb/notebook09c5fdf91b` | 2026-07-24 | 0 | **Qwen3-4B-4bit** | NvidiaTeslaT4 |
| — | — | (not on LB) | `hussensehs/trace` | 2026-08-11 | 5 | **Qwen3-dense-small** | NvidiaRtxPro6000 |
| — | — | (not on LB) | `namthanh189/fine-tuning-qwen-3-8b` | 2026-07-22 | 0 | **Qwen3-dense-small** | NvidiaRtxPro6000 |
| — | — | (not on LB) | `nguyenlamphuquy/ai-race-experiment-notebook` | 2026-07-31 | 5 | **Qwen2.5-32B** | NvidiaRtxPro6000 |
| — | — | (not on LB) | `trungnguyen2710t/aifinancial` | 2026-08-09 | 0 | **Qwen3-dense-small** | NvidiaRtxPro6000 |
| — | — | (not on LB) | `trungnguyen2710t/rtx6000` | 2026-07-02 | 1 | **Qwen3-dense-small** | None |
| — | — | (not on LB) | `trungnguyen2710t/sdft-training` | 2026-07-22 | 0 | **Qwen3-dense-small** | NvidiaRtxPro6000 |

---

## Appendix B — artefact index

| artefact | why it matters |
|---|---|
| `ippeiogawa/qwen35-122b-a10b-nvfp4` | 83.5 GB VL MoE, the recommended brain. Owner **#14 @ 1.58**. |
| `jcole75/arc3-qwen36-runtime-wheels` | vLLM 0.24.0 + **modelopt 0.45.0** + CUDA 13.3, "built for RTX PRO 6000 Blackwell". The only public NVFP4-capable runtime. Owner **#89 @ 1.33**. |
| `driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot` | The incumbent. 99 kernels, 52 teams, 1,178 downloads. |
| `driessmit1/arc3-vllm-h100-wheelhouse-v3` | vLLM **0.19.0**, H100 build, **no ModelOpt** → cannot serve NVFP4. |
| `iseesmth/arc3-nca-qwen36-sft` + `duck-harness-nca-qwen36-adapter-20260811` | LoRA on the duck's own 27B, 08-11/08-12. The actual public frontier. Owner **#163 @ 1.25**. |
| `jakobbrggen/taaf-kaggle-source-anim-20260807-anim` | Helmut AGI (#11 @ 1.61) animation-awareness bundle — already ADOPTed 08-11. |
| `ravi123a321at/deepseek-v4-flash-0731-iq2m-shard2` | 49.6 GB IQ2_M GGUF, 08-10, owner **Ravi's Agi #61 @ 1.39**. Not vLLM-servable. Watch. |
| `cstl` / `gatamaz` / `tehnar` | **Zero public kernels, zero public datasets.** The 2.52 leaves no trace. |
