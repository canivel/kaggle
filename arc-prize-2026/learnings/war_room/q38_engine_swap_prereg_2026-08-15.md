# Q38 ENGINE-SWAP EVAL — PREREG, sealed 2026-08-15 BEFORE the push

**Arm:** `canivel/arc3-q38-engine-eval` (fresh slug) · **one variable: the weights.**
**Question:** does `Qwen3.8-27B-FP8`, dropped into the frozen duck fork with nothing else
changed, move our local-25 number — and specifically, does it reproduce the *"consistent 2×
score on the local 25 dataset"* that a competitor reported in Kaggle discussion 735243?

This is the **named primary falsifier** of `learnings/war_room/research_restart_2026-08-15.md`
§8.1: *"We screen Qwen3.8 and it does not beat the frozen fork. Then the engine is not the
driver, the five jumps need another explanation, and §2 collapses. This resolves in one free
build."*

Provenance tags: **[V]** verified by direct read/execution this session · **[V-ext]** external
primary source · **[INF]** inference.

---

## 1. THE DROP-IN CLAIM — RE-VERIFIED INDEPENDENTLY, NOT INHERITED

The order was explicit that this campaign has twice let a hypothesis harden. Everything below
was re-derived this session from files pulled fresh, and from the **pinned vLLM 0.19.0 wheel
itself** (432 MB, downloaded from `driessmit1/arc3-vllm-h100-wheelhouse-v3` and read), not from
the research report.

### 1.1 Mirror selection — **`saltb0x/qwen3-8-27b-fp8`** [V]

All three public mirrors were listed and their config/tokenizer/template files downloaded and
hashed. **All seven small files are byte-identical across all three mirrors** (sha256-16):

| file | saltb0x | mustangliu | johnlussier |
|---|---|---|---|
| `config.json` | `74227dd615bf1ea9` | same | same |
| `chat_template.jinja` | `c3cf9e34abf4f9e3` | same | same |
| `tokenizer_config.json` | `b11349aafa7cdc6a` | same | same |
| `generation_config.json` | `e70c136c1b78ddc1` | same | same |
| `preprocessor_config.json` | `27225450ac9c6529` | same | same |
| `video_preprocessor_config.json` | `7768af27c1fafa9c` | same | same |
| `model.safetensors.index.json` | `f0838c766951bdfe` | same | same |

**No mirror is missing a file** — the 122B failure mode (a snapshot missing
`tokenizer_config.json`/`processor_config.json`, discovered only *after* a 78 GB load) does
**not** recur here. All three carry 64 `layers-N.safetensors` + `mtp.safetensors` +
`outside.safetensors` + the index + a complete tokenizer set (`tokenizer.json`, `vocab.json`,
`merges.txt`).

**Chosen: `saltb0x/qwen3-8-27b-fp8`.** Reason, and it is a weak reason honestly stated: the
three mirrors are *content-identical*, so the choice cannot be made on content. It is made on
**operational risk only** — saltb0x has the most downloads (30 vs 25 vs 6), is Apache-2.0
licensed on Kaggle (mustangliu's is `unknown`, which is a needless licence question on a
competition artefact), and its owner is an LB competitor at 1.39 who presumably has it
attaching cleanly. If it fails to attach, `mustangliu/qwen38-27b-fp8-hf-snapshot` is a
drop-in replacement of the same bytes.

### 1.2 Config diff vs the served incumbent — field by field [V]

`text_config` is **identical in all 33 fields**: vocab 248320, 64 layers, hidden 5120, 24
heads, 4 KV heads, head_dim 256, `full_attention_interval` 4, intermediate 17408, max_pos
262144, `attn_output_gate` true, `mtp_num_hidden_layers` 1, `partial_rotary_factor` 0.25,
identical `rope_parameters` (mrope_section [11,11,10], theta 1e7). `image_token_id` 248056,
`video_token_id`, `vision_start/end` all identical. `architectures` identical.

**Every difference, exhaustively:**

| field | Qwen3.6 (served) | Qwen3.8 (new) | assessed |
|---|---|---|---|
| `quantization_config` | compressed-tensors W8A8 per-tensor | **fp8 blockwise e4m3, `weight_block_size [128,128]`, dynamic activations, 882 `modules_to_not_convert`** | **the real difference** |
| root `dtype` | `bfloat16` | **absent** | benign — see 1.3 |
| `vision_config.dtype` | `bfloat16` | absent | benign — see 1.3 |
| `vision_config.model_type` | `qwen3_5_vision` | **`qwen3_5`** | benign, and *more* correct — see 1.3 |
| `transformers_version` | 5.6.2 | 5.8.0.dev0 | benign (vLLM ships its own config class) |
| weight layout | one 35.9 GB `model.safetensors` | 64 shards + index, **25.3 GB** | benign |
| image processor | `processor_config.json`, `Qwen2VLImageProcessor` | `preprocessor_config.json`, `Qwen2VLImageProcessorFast` | benign — see 1.3 |

### 1.3 The four differences that could have bitten, each resolved against the pinned wheel [V]

Reading the actual `vllm-0.19.0` and `transformers-4.57.6` wheels, not the docs:

1. **`vision_config.model_type` changed.** `vllm/transformers_utils/configs/qwen3_5.py` defines
   `class Qwen3_5VisionConfig(PretrainedConfig): model_type = "qwen3_5"` and
   `Qwen3_5Config.__init__` builds it as `self.sub_configs["vision_config"](**vision_config)` —
   the class is fixed by `sub_configs`, the dict's `model_type` is never used as a lookup key.
   The new value **matches vLLM's own class attribute**, the incumbent's did not. Risk retired,
   and the new file is the more aligned of the two.
2. **Root `dtype` absent** could have silently selected fp16 instead of bf16.
   `ModelArchConfigConvertorBase.get_torch_dtype` falls back to
   `hf_config.get_text_config().dtype` **before** the `torch.float32 → float16` default, and
   `text_config.dtype = "bfloat16"` is present. bf16 is preserved. Risk retired.
3. **Blockwise FP8 on SM120.** `Fp8Config.from_config` accepts `quant_method: "fp8"`,
   `activation_scheme: "dynamic"`, `weight_block_size: [128,128]` and maps
   `modules_to_not_convert → ignored_layers`; `get_min_capability()` is 75. The blockscale
   dispatcher `_dispatch_w8a8_blockscale_op` falls through CUTLASS → aiter → **Triton**, an
   architecture-agnostic path, so an unsupported kernel *degrades*, it does not raise. **This
   is structurally unlike the 122B death**, which was `flashinfer/jit/fused_moe.py` refusing
   nvcc flags for arch 12 — a **MoE** path. Qwen3.8-27B is **dense** and never enters it.
4. **`Qwen2VLImageProcessorFast` + `TRANSFORMERS_NO_TORCHVISION=1`.** transformers 4.57.6's
   `AutoImageProcessor` has the explicit fallback `if use_fast and not is_torchvision_available():
   → get_image_processor_class_from_name(image_processor_type[:-4])`. vLLM already requests
   `use_fast=True` for the incumbent (`qwen3_vl.py:698`), so this path is already exercised
   today. Risk retired.

Also verified: `mtp.*` weights (22 tensors, 477 MB) are skipped by
`Qwen3_5ForConditionalGeneration.load_weights(..., skip_prefixes=["mtp."])`; the special-token
ids in `tokenizer_config.json` (`added_tokens_decoder`, 33 entries, 248044–248076) match the
config's `image_token_id`/`video_token_id`/`vision_*`; the new `tokenizer_config.json` is the
**transformers-v4 layout**, i.e. *closer* to the pinned 4.57.6 than the incumbent's v5 layout.

**Verdict on the drop-in claim: IT SURVIVES.** The claim is confirmed and, on two points
(vision `model_type`, tokenizer-config layout), the new snapshot is a *better* fit for the
pinned runtime than the incumbent is.

**Residual named risk [INF]:** the blockwise-FP8 GEMM on SM120 may land on the Triton fallback
rather than a tuned CUTLASS kernel. That is a **throughput** risk, not a correctness risk, and
it is exactly why tok/s is a required secondary read (§4).

---

## 2. THE `reasoning_effort` PIN — the isolated knob, and how it was chosen

Qwen3.8's template adds `reasoning_effort`, **defaulting to `xhigh`**, injected as a
system-prompt paragraph. On a rail our own evidence says is wallclock-bound, shipping that
default alongside the weights would make the arm two-variable.

**Reading the template (lines 45–56 of the attached `chat_template.jinja`):** `xhigh` and `low`
each set `reasoning_instructions` to a sentence; **`medium` is validated-but-silent** — it
passes the allowed-value check and leaves `reasoning_instructions` empty. That is the neutral
value, and it was confirmed by rendering, not by reading.

**Measured** (`duck_eval/q38/q38_smoke.py` §5, both real templates, harness-shaped payload:
system + tools + multimodal user + assistant-with-tool_call + tool + user, `preserve_thinking=true`,
`enable_thinking=true`): [V]

| render | chars | identical to the Qwen3.6 template? |
|---|---:|---|
| Qwen3.6 (incumbent) | 1495 | — |
| **Qwen3.8, `reasoning_effort="medium"`** | **1495** | **YES — zero-line diff** |
| Qwen3.8, default (= `xhigh`) | 1704 | no (+209 chars of instruction) |
| Qwen3.8, `"low"` | 1633 | no |
| Qwen3.8, `"ultra"` | — | raises (fails loud, as designed) |

**PINNED: `--default-chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "medium"}'`.**

**Why the pin reaches the template** [V, read in the wheel]: the harness sends per-request
`chat_template_kwargs={"enable_thinking": bool}` (`openai_compat.py:78`); vLLM merges
`default_chat_template_kwargs | request_chat_template_kwargs`
(`vllm/entrypoints/openai/engine/serving.py:807`) — server defaults survive for keys the
request does not send. `enable_thinking` keeps coming from the request exactly as today.

**Two other template deltas, both established to be no-ops for OUR harness** [V]:
- `preserve_thinking`'s default flipped to true. We pass it **explicitly**, so no change.
- Qwen3.6's `else` branch that split an inline `</think>` out of assistant `content` was
  removed. Our harness sets `assistant_message["reasoning"]` (`tool_agent.py:1969`), **never**
  `reasoning_content`, and strips `<think>` tags from content before replay — so
  `reasoning_content` is empty under *both* templates. Identical behaviour. (Separately: this
  is the mechanism behind our own "the agent FORGOT" finding, and the swap neither fixes nor
  worsens it.)

**`xhigh` / `low` are a SEPARATE, LATER ARM.** Not touched here.

---

## 3. THE ARTIFACT

Built by `duck_eval/q38/build_q38_eval.py` from
`notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb` (the frozen fork —
`feedback_arc_kernel_structural_drift`, never hand-built). Fresh slug
(`feedback_fresh_kernel_slug`). **A plain BUILD of this fork *is* the offline 25-game eval** —
`TRUE_SUBMISSION` is unset outside a rerun, so cell 14 plays the bundled environments via
`_offline_games()`; no eval-mode graft is added.

**Exactly three code cells differ from the frozen fork: 2, 6, 8.** Cells 12 (customization
hook) and 14 (the run cell — the entire measurement surface) are **byte-identical**. [V]

- **cell 2** — greppable identity banner. No behavioural change.
- **cell 6** — `DATASET_SOURCES`: the engine entry only.
- **cell 8** — anchored, counted, vetoed, invariant-checked, **FAIL-LOUD** rewrite of the
  bundle's single setup command. Policy is inverted vs the graft builds (which fall back to
  vanilla duck): here a fallback would **silently serve Qwen3.6** and hand us a number we would
  read as an engine result.

The **entire footprint** on the pristine setup command is 4 removed lines: [V]

```
-MODEL_OWNER = 'driessmit1'
-MODEL_SLUG = 'vrfai-qwen3-6-27b-fp8-hf-snapshot'
-SERVED_MODEL_NAME = 'vrfai/Qwen3.6-27B-FP8'
-        '{"preserve_thinking": true}',
```

**18 invariants are asserted to survive** the rewrite, at build time and again at runtime — the
one-variable proof, mechanised rather than asserted in prose: wheelhouse owner/slug,
`--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`, `--enable-prefix-caching`,
`--generation-config vllm`, `preserve_thinking`, `VLLM_MAX_MODEL_LEN = 65536`,
`ANALYZER_CONTEXT_WINDOW = 32768`, `VLLM_TENSOR_PARALLEL_SIZE = 1`, temperature 0.6, top_p 0.95,
top_k 20, `ENABLE_THINKING=true`, `MULTIMODAL_CONTEXT=current_grid`, `MULTIMODAL_UPSCALE=4`,
`YIELD_SECONDS=60`, the vLLM 0.19.0 stamp.

### 3.1 In-kernel gates (all FAIL-LOUD)

**Pre-serve, before the 25 GB load** — the 122B lesson, applied: required-file check, then
`architectures`, `quant_method == fp8` + `weight_block_size == [128,128]` (**the one field that
cannot be true of Qwen3.6**), `transformers_version` 5.8.x, vocab/layers/hidden/KV/head_dim/
interval, `image_token_id`, 64 shards, and the template's `reasoning_effort` knob.

**The effort pin is certified by TWO independent instruments, each with its own positive
control**, and it is **FATAL if neither certifies** — an uninterpretable 2 h run costs the slot,
dying at boot costs minutes:
1. *local render* of the attached template inside the setup process (no transport, cannot fail
   for infra reasons);
2. *server probe* — `POST /tokenize` with `return_token_strs`, which applies the same
   `default_chat_template_kwargs` as `/chat/completions`, so it observes the **real served
   prompt**.
   Each asserts `reasoning_instruction ABSENT` **and** that the `xhigh` control renders it
   PRESENT. A probe that cannot see the string is declared BLIND rather than passing.

**Post-serve:** `/v1/models` must list exactly `Qwen/Qwen3.8-27B-FP8`; forced tool-call
round-trip (gating); auto tool-call round-trip through `qwen3_coder` (reported, not fatal); one
real 64×64 PNG through the vision tower (gating).

### 3.2 Build gates — RUN, not merely written

`python duck_eval/q38/q38_smoke.py` → **78 passed, 0 failed** [V]. It does not inspect the
notebook, it **executes** the rewrite against the real bundled `setup_commands.json`, walks the
rewritten command's AST for unresolved names (**the exact class that killed LoRA v1 on 08-14 —
`compile()` catches syntax, not scope**), and actually runs the injected pre-serve asserts.

**Every gate has a paired negative control**, because a gate that only ever passes is not a
gate:
- the asserts **REJECT** a staged real Qwen3.6 snapshot (`quant_method` mismatch) — a
  silent-incumbent run cannot pass;
- they **REJECT** a snapshot with `tokenizer_config.json` removed, **before** the load;
- they **REJECT** a 63-shard set;
- the name gate **CATCHES** an injected out-of-scope call;
- the neutrality probe **DETECTS** the instruction under `xhigh`.

`python duck_eval/q38/q38_score.py --selftest` → **22 passed, 0 failed** [V].

---

## 4. THE READ — SEALED BEFORE THE PUSH

**Decision statistic: mean Δ levels-completed per game** (SCREEN_PROTOCOL §0). Everything else
is descriptive and non-inferential.

### 4.1 Baseline (P1/P2/P3 as required by SCREEN_PROTOCOL §1)

**Family `duck-harness-kaggle`, m = 3** — the plain frozen-fork offline bench: [V]

| run | lc | mean score | tok/s (job wallclock) | actions |
|---|---:|---:|---:|---:|
| `runs/kernel_pulls/gate_eval_v1` | 18 | 1.427 | 212.52 | 4757 |
| `runs/kernel_pulls/gate_eval_v2` | 19 | 1.939 | 203.66 | 4033 |
| `runs/tmp_pullback_duckgate_v1post` | 21 | 3.420 | 197.47 | 4632 |

**Per-game mean lc = 58 / (3 × 25) = 0.773333.** m = 3 ⇒ **SCREENABLE** (P2 satisfied).
σ̂ = **0.141740**, df 6 — the SCREEN_PROTOCOL §P3 standing pooled build-rail estimate (P3
satisfied). SE(Δ) at k=1, m=3 = σ̂·√(1+1/3) = **0.163667** lc/game = 4.09 levels over 25.

**P1 CAVEAT, STATED NOT BURIED.** All three baseline runs are the frozen fork **plus the
boristown readiness-gate cell** (a standalone `wait_vllm_ready()` poll inserted before
`bm.run`); our arm has no such cell. This is a genuine composition difference and P1 asks for
banner evidence, so here it is: the gate touches **no solver surface**, adds no package or
dataset, and is **functionally redundant** — cell 8's own setup command already blocks in
`wait_for_vllm_server()` before the notebook proceeds. Both compositions stamp the identical
`bm.label = duck-harness-kaggle`. I judge this a legal control and record the caveat so a
future reader can overturn the judgement without re-deriving it. It is the **largest m** and
the **only vanilla** family available; the alternative (m=2 `continuation-v1`) is not screenable
at all.

**Secondary comparators, descriptive only:** `duck-null-seedNNN` m=10 (lc 16,11,16,15,16,15,14,
18,18,13 → 0.608/game), `duck-harness-kaggle-warpack-v1` m=3 (22,15,13 → 0.667/game — **the
warpack band is an ILLEGAL control** per `r17_thresholds.json`, diagnostic only),
`animation_v1` (17 → 0.680/game).

### 4.2 The lines (sealed in `duck_eval/q38/q38_score.py`)

Evaluated in this order:

| verdict | condition | arm lc over 25 |
|---|---|---|
| **INFRA DEATH (not decisive)** | no `benchmark.json` / ≠25 games / boot asserts never PASSED / >2 windows drifted >5% | — |
| **HARM (decisive)** | Δlc ≤ **−0.286320** (= −C(3)·σ̂, C(3)=2.02) | ≤ 12 |
| **REFUTE-2× (decisive)** | Δlc ≤ **+0.250000** | ≤ 25 |
| **CONFIRM-2× (decisive)** | Δlc ≥ **+0.500000** | ≥ 32 |
| **INDETERMINATE** | between | 26–31 |

### 4.3 WHAT WOULD FALSIFY THE COMPETITOR'S CLAIM — stated before the data

The claim is *"a consistent 2× score on the local 25"*. Taken on the decision statistic, a
doubling means Δlc = **+0.7733/game = +19.3 levels**, i.e. an arm total near **39 levels**.

- **REFUTE-2× (≤ 25 levels) is the falsifier.** Under a true doubling the probability of
  landing there is Φ((0.25 − 0.7733)/0.163667) = Φ(−3.20) = **0.07 %**. So a REFUTE reading
  rejects the 2× claim on our harness at better than 3σ.
- **CONFIRM-2× (≥ 32 levels)** has **95.3 %** power against a true doubling and a **0.11 %**
  false-positive rate under the null. This screen is well powered for the question asked — the
  claimed effect is enormous relative to σ̂.
- A **HARM** reading additionally kills `research_restart` §2 outright.

**POWER HONESTY (SCREEN_PROTOCOL §4.6).** The lc read is well powered *against a doubling* and
badly powered against small effects: the 80 %-power floor is
`(2.02 + 0.8416)·0.14174·√(1+1/3)` = **0.468 lc/game = 11.7 levels**. An INDETERMINATE result
therefore means exactly what it says — one seed cannot separate a +0.25…+0.50 lift from noise —
and **may not be reported as either a confirmation or a refutation.**

**The score-based reading carries NO verdict and this is decided now, not after seeing it.**
The baseline's own mean_score is 1.427 / 1.939 / 3.420 — sd 1.033 on n = 3, a 2.4× spread
*within the null*. A score-based 2× test has ≈60 % power, below the SCREEN_PROTOCOL bar. It is
reported as description only.

### 4.4 Required secondary reads (descriptive, no verdict)

- **tok/s (job wallclock)** — baseline 212.52 / 203.66 / 197.47, mean 204.55. A 25.3 GB model
  moves 29 % less weight per token than a 35.9 GB one, so **[INF] it should be FASTER**. If
  tok/s falls below **197.47** (under the entire baseline range) that is a real throughput
  regression and is the visible signature of the Triton-fallback risk in §1.3.4. **Record it
  either way — a smaller model that is not faster is itself a finding.**
- **actions/window** — baseline 4757 / 4033 / 4632 total (mean 178.96 per 7920 s window).
- **windows** — all 25 should end at 7920 ± 5 % (the rail is wallclock-bound).

---

## 5. WHAT THIS ARM DOES NOT ANSWER

- **Nothing about the leaderboard.** This is the local-25 offline bench. No submission is made;
  the daemon owns the submission window.
- **Nothing about `reasoning_effort`.** Pinned to the neutral value by construction.
- **Nothing about cstl.** cstl's 2.70 predates the release by 43 h and is chronologically immune
  to this explanation either way.
- **n = 1.** One arm seed. A CONFIRM licenses a follow-up (a second seed, and only then a
  submission arm); it does not license a submission on its own.

---

## 6. COST AND CONSTRAINTS

One free Kaggle BUILD, ~2 h 12 m of the 30 GPU-h/week allowance
(`feedback_arc_zero_budget`). **No submission. No queue edit. No cloud spend.**
Slot: 08-15's single remaining slot, re-confirmed from `ITERATION_LOG.md` immediately before the
push (§11.4) by `duck_eval/q38/q38_push.sh`, which also carries the `--confirm-push` intent
interlock and the idempotence guard (the two guards the 08-14 duplicate-push incident produced).

---

## 7. HOW TO READ THE RESULT

```bash
kaggle kernels status canivel/arc3-q38-engine-eval
kaggle kernels output canivel/arc3-q38-engine-eval -p runs/kernel_pulls/q38_v1
python duck_eval/q38/q38_score.py runs/kernel_pulls/q38_v1
```

Greps, in order of decisiveness:

```bash
# 1. did we actually serve Qwen3.8, and was the pin real?
grep -E "Q38-EVAL (engine-config|served|effort-pin|effort-pin-certified-by)" runs/kernel_pulls/q38_v1/*.log
# 2. did every boot gate pass?
grep -E "Q38-EVAL (setup-commands rewrite OK|tool-call-roundtrip|mm-image-roundtrip|BOOT-ASSERTS PASSED)" runs/kernel_pulls/q38_v1/*.log
# 3. anything fatal or degraded?
grep -E "Q38-EVAL (FATAL|WARN)" runs/kernel_pulls/q38_v1/*.log
# 4. THE NUMBER (levels are the decision statistic; score is descriptive)
grep -E "^(benchmark|games|mean score|total actions|generated tokens/sec)" runs/kernel_pulls/q38_v1/summary.txt
grep -c "levels=" runs/kernel_pulls/q38_v1/summary.txt
grep -o "levels=[0-9.]*" runs/kernel_pulls/q38_v1/summary.txt | cut -d= -f2 | paste -sd+ | bc
# 5. the engine actually loaded the way we predicted
grep -E "Loading weights took|Model loading took|quantization|blockscale|Triton|cutlass" runs/kernel_pulls/q38_v1/vllm-openai-server.log
```

**Baseline to hold it against, from this file, not from memory:** lc 18 / 19 / 21 (mean 19.33),
mean score 1.427 / 1.939 / 3.420, tok/s 212.52 / 203.66 / 197.47.
**CONFIRM-2× at ≥ 32 levels. REFUTE-2× at ≤ 25. HARM at ≤ 12.**

---

## 8. ADDENDUM — PUSH RECORD, 2026-08-15 (append-only; nothing above was edited)

**PUSHED: `canivel/arc3-q38-engine-eval` version 1, 2026-08-15, against 08-15 slot 2 (the day's
last).** Status at hand-off: **RUNNING** (~2 h 15 m expected).

**Ledger re-confirm (§11.4) executed and it mattered.** The mechanised excerpt surfaced a clause
my initial read had missed: *"ALL CURRENT LANES STOOD DOWN pending the research restart… No slot
spend until the research lands"*, and separately *"ZERO slot spend today; 08-15 slot 2 FREE AND
UNSPENT."* Resolution: the stand-down is scoped **"until the research lands"**, the research has
landed (`research_restart_2026-08-15.md`, logged), and this arm is the principal's explicit
post-research order naming this slot — the re-authorization the log's own OWED item (3)
contemplates. Corroborated independently: `kaggle kernels list --user canivel --sort-by dateRun`
shows the newest run on any of our kernels at **2026-08-14 13:40**, i.e. nothing pushed today.

**Pre-push gates:** build idempotent · smoke **78/0** · sealed-scorer selftest **22/0** ·
idempotence guard PASS (not a duplicate) · `--confirm-push` intent interlock.

**Pull-back verify:** 3/3 `dataset_sources` survived **including the 25 GB engine**
(`saltb0x/qwen3-8-27b-fp8`) — the entry most likely to be silently dropped; incumbent engine
absent; `enable_gpu` true, `enable_internet` false, `machine_shape NvidiaRtxPro6000`,
`competition_sources` and docker sha `…4cb13c` all byte-identical to the frozen fork; 17 cells;
all engine + pin tokens present in the remote source.

**Preflight (post-push):** `--family duck-harness --baseline <frozen fork>
--expect-diff-cells 2,6,8` → **ALLOW, 0 fails, 0 warns, D4 = [2, 6, 8] EXACT**, D2 byte-identical
(both absent), D3 cell shape matches.

### 8.1 One real defect in v1, measured and NOT waved away

The pull-back code sha did **not** match (`2114be21…` local vs `9633bc07…` remote). Diagnosed
rather than assumed:

- Five cells differ: **7, 11, 13, 16** — all *pristine frozen-fork cells we never edited* — and
  **8**. Every difference is a single non-ASCII character re-encoded UTF-8 → cp1252 mojibake
  (`—` U+2014 → `â€"`). **Remote is ASCII-identical to local in all five.**
- **Which side mangles was settled by experiment, not inference:** pulling the *upstream public*
  notebook `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner` returns **clean
  U+2014**. The pull path is faithful ⇒ **the PUSH path (Kaggle CLI 2.0.1) is the mangler**, and
  the artifact stored on Kaggle genuinely carries mojibake. `preflight.py` already knows this
  class and treats cell 16 as equal.
- **The only load-bearing casualty is `.replace('\u0120', ' ')` in the server-side effort probe**
  (the BPE space marker in the `/tokenize` normaliser). Consequence, traced end to end: the
  joined prompt keeps its `\u0120` separators → the `xhigh` positive control does not match →
  the probe correctly declares itself **BLIND** and prints a WARN → **it does not raise**. The
  pin remains certified by the **local render**, which is pure ASCII and unaffected, and the
  `FATAL if neither instrument certifies` rule still stands.
- **Nothing that gates the measurement is affected:** the setup rewrite, vetoes, 18 invariants,
  pre-serve config asserts, served-model-id check, tool-call and MM probes are all ASCII.

**RULING: v1 stands; NO v2.** The defect costs one *corroborating* instrument, not the
measurement and not a gate. A re-push would spend a slot we do not have on a cosmetic fix — the
exact reflex the 08-14 duplicate-push incident exists to stop.

**Banked so it cannot recur:** the builder now writes the two BPE markers as `\uXXXX` **escapes**
(7-bit source), and `q38_smoke.py` gained a hard gate — *cells 2, 6, 8 must be pure ASCII*
(**smoke now 81/0**). `q38_push.sh` gained a **one-shot guard** (exit 5 if the kernel already
exists, overridable only by `Q38_ALLOW_V2=1`) so the now-divergent local artifact cannot be
pushed as a v2 without a fresh slot and a fresh authorization.

**New durable lesson for the campaign:** *the Kaggle CLI push path is not byte-transparent — it
re-encodes non-ASCII. Any character we author into a kernel cell must be 7-bit, or written as an
escape. The frozen fork's own em-dashes have been quietly mangled on every push for months; it
only became load-bearing the moment we injected a non-ASCII literal into executable code.*

---

## 9. POST-MORTEM — v1 ERRORed at t=425 s. **THE ENGINE WORKED. MY PROBE KILLED IT.** (append-only)

**Sealed scorer verdict: `INFRA DEATH (not decisive)` — "no benchmark.json; infra signature 'Q38-EVAL FATAL'".** The third state was the right one to have built; there is no legal way to read this run as evidence about Qwen3.8's capability, in either direction.

### 9.1 The log-retrieval problem is SOLVED, and the LoRA lane's conclusion was wrong

The LoRA canary was diagnosed blind because `kaggle kernels output` downloads `/kaggle/working` first and this kernel's working dir holds the multi-GB `vllm-site-packages` tree; that lane recorded *"the log never arrived usefully on either CLI."* **That is false, and it cost that lane its diagnosis.** Kaggle CLI **2.2.3** (we have it at `/f/kaggle/march-madness-2026/.venv/Scripts/kaggle`, distinct from the 2.0.1 at `…/Python313/Scripts/kaggle.exe` used for pushes) has **`kernels logs`**, which streams the full stdout JSON **without touching the working dir**: 238 KB, 1,501 entries, in seconds. Every future post-mortem gets the log. `kernels files` returns empty for *all* errored kernels (checked against b122 and the LoRA canary) — it is not evidence of anything.

### 9.2 Where it died — measured, to the second

| t (s) | event |
|---:|---|
| 8.13 | `CUDA GPU check passed … ['NVIDIA RTX PRO 6000 Blackwell Server Edition']` |
| 8.06 | `Q38-EVAL setup-commands rewrite OK (6 anchors replaced, 18 invariants held)` |
| 8.22 | `Q38-EVAL effort-pin=medium local-render reasoning_instruction=ABSENT control(default=xhigh)=PRESENT` |
| 8.24 → 99.8 | wheelhouse install (91.5 s) |
| 99.8 → **394.8** | vLLM boot + **25.3 GB weight load = 295 s (4 m 55 s)**; `vLLM server ready: id='Qwen/Qwen3.8-27B-FP8', root='/kaggle/input/datasets/saltb0x/qwen3-8-27b-fp8', max_model_len 65536` |
| 417.3 | stock smoke: **`Generated: 2 + 2 equals 4.`** |
| 417.3 | `Q38-EVAL served=Qwen/Qwen3.8-27B-FP8` |
| ~420 | `Q38-EVAL tool-call-roundtrip=OK mode=forced … args={"action":"ACTION6","x":3,"y":7}` |
| ~423 | `Q38-EVAL tool-call-roundtrip=OK mode=auto parser=qwen3_coder name=submit_action` |
| **425.5** | `RuntimeError: Q38-EVAL FATAL: MM boot probe returned empty content` → papermill → **kernel ERROR** |

**Total cost: 7 min 6 s of GPU, not a GPU-hour.** The fail-loud design at least failed fast.

### 9.3 Was it the engine or was it us? **US. Unambiguously.**

**None of the four cleared differences bit. Every one is now confirmed GOOD in production:**

1. **Blockwise FP8 on SM120 — WORKS.** The server came up and generated. The Triton-fallback risk did not manifest as a failure. (Whether it cost throughput is still unmeasured.)
2. **Absent root `dtype` — no issue.** Model served; the `text_config.dtype` fallback held.
3. **`vision_config.model_type` — no issue.** `Q38-EVAL engine-config` passed all field asserts and vLLM resolved the architecture.
4. **Image-processor fast/slow path — no issue.** The multimodal request returned **HTTP 200**, not a 400 — the image was accepted and processed. (A broken processor raises inside `request_json` as `HTTPError`; the traceback is a `RuntimeError` from *my* assert.)

**Plus three things this run positively established:** the engine **loads and serves in ~5 minutes**; **tool-calling works in BOTH modes including `auto` through `qwen3_coder`** — the exact path the harness uses, with arguments parsed correctly; and **the `reasoning_effort` pin bound in production** (`reasoning_instruction=ABSENT`, xhigh control PRESENT).

**The defect is mine, and it is a payload bug.** My MM probe sent `max_tokens: 32` and **no `chat_template_kwargs`**, so `enable_thinking` defaulted ON. Under `--reasoning-parser qwen3` the first 32 tokens are routed to `reasoning_content` and `content` is `''`. My assert then declared *"the vision path is broken"* and raised.

The proof is in the same log, seconds apart, from the same server:
- the **stock smoke** passes `chat_template_kwargs: {'enable_thinking': False}`, `max_tokens: 96` → **non-empty content**;
- my **auto tool-call probe** had thinking ON but `max_tokens: 256` → **succeeded**;
- my **MM probe** had thinking ON and `max_tokens: 32` → **empty content**.

The only variables separating the last two are the token budget and the image, and the image returned 200. **This is the Jason Feng finding turned on us**: 66.8 % of Qwen tool-call responses in this harness return hidden reasoning with zero visible content. Empty `content` is the *modal* behaviour of this stack, and I asserted on it.

**Honest limit: I cannot fully close it from this log, and that is a second, worse defect.** The assert printed a **conclusion** ("the vision path is broken") and none of the **observation** — no `reasoning_content`, no `finish_reason`, no response body. A probe that judges without dumping its evidence cannot be audited afterwards.

### 9.4 Partial signal against the sealed lines: **NONE**

Zero games ran; `bm.run()` was never reached. No levels, no bench tokens/s, no actions. `CONFIRM ≥ 32 / REFUTE ≤ 25 / HARM ≤ 12` are **untouched and remain sealed**. The only performance number obtained is infrastructural: **server ready 295 s after launch**. Per-phase load timings live in `vllm-openai-server.log` inside `/kaggle/working` and were not downloaded.

---

## 10. GATE RECLASSIFICATION — written BEFORE the rebuild, per coordinator instruction (2026-08-15)

Doctrine, adopted verbatim:

> **Fail the kernel iff the failure would make the number MEAN something other than what we would read it as. Report-only iff the failure would simply BE the number.**

**Corollary, stated explicitly rather than bent into the rule** — the rule above discriminates only among failures that still permit a number. Some failures permit no number at all (an incomplete snapshot cannot load). For those the choice is not fatal-vs-report-only but *fail now vs fail in five minutes*, and the honest justification is **cost, not poisoning**. Those gates stay fatal and are labelled `NO-NUMBER` below so nobody later mistakes a cost argument for a meaning argument.

**Ambiguity policy (coordinator's instruction):** where the classification is genuinely unclear, the gate stays **FATAL** and the reason is written down. Nothing was downgraded because we are impatient for a number.

### 10.1 The classification, gate by gate

| # | gate | before | **after** | class | one-line justification |
|---|---|---|---|---|---|
| A | required snapshot files present | fatal | **FATAL** | NO-NUMBER | An incomplete snapshot cannot serve; failing pre-load turns a 5-minute wasted load into a 2-second exit. Cost argument, not poisoning. |
| B | `quant_method == fp8` and `weight_block_size == [128,128]` | fatal | **FATAL** | POISONING | The one field that cannot be true of Qwen3.6. If it passes, the weights are not the incumbent — this is *the* gate against reading a Qwen3.6 number as a Qwen3.8 number. |
| C | `architectures == [Qwen3_5ForConditionalGeneration]` | fatal | **FATAL** | POISONING | A different architecture is a different experiment wearing this arm's label. |
| D | `transformers_version` starts `5.8` | fatal | **FATAL** | POISONING (weak proxy — **ambiguous, kept fatal**) | Strictly weaker than B, which already pins weight identity. Kept fatal because it is a zero-cost pre-load string compare and a version drift means the mirror is no longer the artifact we diffed in §1.2. Flagged as the one gate whose fatality rests on caution rather than necessity. |
| E | `text_config` vocab/layers/hidden/KV/head_dim/interval | fatal | **FATAL** | POISONING | The screen's entire premise is "structural drop-in". If these differ, the thing we measured is not the thing the prereg describes. |
| F | `image_token_id == 248056` | fatal | **FATAL** | POISONING | A drifted image token silently corrupts every multimodal prompt; the resulting score would be a number about a broken prompt pipeline, not about the engine. |
| G | exactly 64 layer shards | fatal | **FATAL** | NO-NUMBER | A short shard set fails the load. Fail fast. |
| H | attached template contains a `reasoning_effort` knob | fatal | **FATAL** | POISONING | Without the knob our pin is a silent no-op and the arm runs at an unknown effort — two variables read as one. |
| I1 | pinned render still injects an instruction | fatal | **FATAL** | POISONING | Same as H: the prompt is not neutral, so the arm is not one-variable. |
| I2 | xhigh control injects nothing (probe BLIND) | warn | **WARN** | INSTRUMENT | A blind probe proves nothing in either direction; it is an instrument fault, not an arm fault. Already correct — unchanged. |
| I3 | system sentinel did not render | fatal | **FATAL** | POISONING (**ambiguous, kept fatal**) | If the template drops the system message, every harness prompt is malformed and the score measures that, not the engine. Could be argued as "that IS the number"; kept fatal because a silently truncated system prompt is indistinguishable from a capability result. |
| J | cell-8 rewrite: 1 command, anchors ×1, veto absent, 18 invariants survive | fatal | **FATAL** | POISONING | A failed rewrite serves the incumbent or changes more than one variable. The canonical case for the inversion. |
| K | `/v1/models == [Qwen/Qwen3.8-27B-FP8]` | fatal | **FATAL** | POISONING | The canonical gate. A silent incumbent read as the new engine is the one failure this screen cannot survive. |
| L | server `/tokenize` effort probe | warn | **WARN** | INSTRUMENT | A corroborator for I1. Its absence never changes what the number means. Unchanged — **but see §10.2, its URL was wrong.** |
| M | pin uncertified by BOTH instruments | fatal | **FATAL** | POISONING | An unknown-effort run read as a neutral-effort run. Kept fatal; it costs minutes at boot versus an uninterpretable 2-hour run. |
| N | forced tool-call round trip | fatal | **REPORT-ONLY** | **RECLASSIFIED** | A broken tool-call path produces a genuine near-zero score for Qwen3.8-in-this-harness. That is a true measurement, not a poisoned one — it **IS** the number. Additionally now evidenced working in production (v1, both modes). |
| O | auto tool-call round trip via `qwen3_coder` | warn | **WARN** | unchanged | Same argument as N; was already correct. |
| P | MM image round trip | fatal | **REPORT-ONLY** | **RECLASSIFIED — this is the gate that killed v1** | A broken vision path yields a low score for Qwen3.8, which is a real result we would read correctly. It is the number, not a corruption of it. |

**Net effect: exactly two gates change (N and P), both fatal → report-only, and both by the same argument.** Applied to the v1 log, the kernel would have printed two WARNs at t≈425 s and proceeded into the 25-game bench.

### 10.2 Three defects in the probes themselves, fixed alongside the reclassification

1. **The payload bug that actually killed v1.** The MM probe sent `max_tokens: 32` with **no `chat_template_kwargs`**, so thinking stayed on and all 32 tokens went to `reasoning_content`. **Fix:** every probe payload is now *harness-shaped* — `chat_template_kwargs` always present, `max_tokens >= 256` always. Enforced by a static lint in the smoke (§10.3), so it cannot regress.
2. **The `/tokenize` 404 was my URL, not a missing endpoint.** I verified the route exists in the wheel (`vllm/entrypoints/serve/tokenize/api_router.py → post /tokenize`) but not the path it is mounted at. It is **root-level**, while `/v1/models` is under `/v1`; I built the URL by appending to `VLLM_BASE_URL` (which ends in `/v1`) and got `/v1/tokenize`. **Same class of error as the payload bug: I verified existence, not call shape.** Fixed — the probe now derives the root URL, and the second pin instrument comes back.
3. **Fail-loud messages printed a conclusion and not the observation.** `"the vision path is broken"` was an inference; the response body, `finish_reason` and `len(reasoning_content)` were never printed, which is why the vision question still cannot be closed from the v1 log. **Fix:** every probe now emits a `Q38-EVAL OBSERVE …` line carrying `finish_reason`, `content_chars`, `reasoning_chars` and a truncated body **before** any verdict, pass or fail.

### 10.3 Coverage boundary — the smoke now prints what it does NOT cover

Adopted from the coordinator's note on *"81 passed, 0 failed is a true statement about a surface that excludes the failure region, printed with no indication of that exclusion."* The suite now ends with an explicit boundary statement naming the unvalidated surface (served-model semantics: token budgets, thinking routing, endpoint mount paths, kernel selection, throughput) and the payload lint that partially closes it. A pass count without a coverage boundary is a half-truth.

---

## 11. FIXES LANDED — 2026-08-15, zero slots, no push (append-only)

All four authorized fixes are in. **v2 artifact: `code_sha256=8babf6de9934c3e5`, cells [2,6,8], smoke 109/0, scorer 22/0.** Nothing was pushed.

### 11.1 What changed in the boot path

Per §10, **exactly two gates moved fatal → report-only** (N tool-call, P MM image). Three probe defects were fixed alongside:

- **Payload bug (the killer):** every generation payload is now harness-shaped — `chat_template_kwargs` always sent, `max_tokens` 512 (was 32 with no kwargs). The MM probe additionally runs with `enable_thinking: False`, so an empty `content` now means the vision path really is dead rather than that the model is still thinking.
- **`/tokenize` 404 was our URL, not a missing endpoint** — it is mounted at the **root**, while `/v1/models` is under `/v1`, and we appended to `VLLM_BASE_URL`. Fixed by deriving the root. **The second pin instrument is restored**, and the healthy-server replay now records `effort-pin-certified-by=local-render,server-tokenize`.
- **Observation before verdict:** a new `_q38_observe()` emits `Q38-EVAL OBSERVE <tag> finish_reason=… content_chars=… reasoning_chars=… tool_calls=… completion_tokens=… content_head=… reasoning_head=…` for **every** probe, pass or fail, *before* any judgement.

### 11.2 The fix is proven by replay, not by inspection

`q38_smoke.py` §6d stubs the server **exactly as v1 observed it** — `/v1/models` correct, `/tokenize` 404, tool calls fine, MM answering **200 with empty `content` and the output in `reasoning_content`** — and runs the real `_q38_boot_asserts` against it:

- **the exact v1 scenario now COMPLETES** and reaches `BOOT-ASSERTS PASSED - handing off to the 25-game offline bench`;
- it emits `WARN mm-image-roundtrip=EMPTY-CONTENT reasoning_chars=42` — **the number that would have closed the vision question in v1 and was never printed**;
- a broken tool-call path is reported, not fatal;
- a healthy server certifies the pin by **both** instruments;
- **negative control: a silently-served `vrfai/Qwen3.6-27B-FP8` is STILL FATAL.** The poisoning gate is intact; only the "would BE the number" gates were relaxed.

### 11.3 The static lint that would have caught v1 with no GPU

`q38_smoke.py` §6b parses the injected defs' AST and enforces, on every generation payload: **`chat_template_kwargs` present** (the harness always sends it — `openai_compat.py:78`) and **`max_tokens >= 256`**. `/tokenize` payloads are exempted from the budget rule **and say so out loud** rather than being skipped silently. Negative control: the lint rejects the reconstructed v1 payload.

This is the honest closure of the gap. The unreachable surface is *served-model semantics*; the reachable proxy is *payload shape*, and payload shape is where the bug actually was.

### 11.4 Coverage boundary is now printed with the pass count

Adopted verbatim from the coordinator's note. The suite ends with `COVERAGE BOUNDARY`, naming what is validated (structure, metadata, the rewrite executed against the real bundle, AST name resolution, real template renders, pre-serve asserts executed against a staged snapshot with negative controls, payload shape) and — more importantly — **what is not**: endpoint mount paths, response-field routing, token budgets vs thinking, kernel selection, throughput, and the number itself. *A pass count without its coverage boundary is a half-truth; v1's "81 passed, 0 failed" was true of every check it ran.*

### 11.5 `kernels logs` banked where the next post-mortem will find it

`duck_eval/README.md` step 3 now carries the procedure, the two-CLI trap (2.0.1 pushes and has no `logs`; **2.2.3** has it), the note that `kernels files` is empty for every errored kernel, and an explicit correction of the LoRA lane's standing "the log never arrived on either CLI". `q38_push.sh`'s post-run hint was rewritten to point at `kernels logs` first. A dated correction was appended to `learnings/war_room/lora_lane_2026-08-13.md` itself — the original text is left readable, as change control requires.

### 11.6 What tomorrow's operator must do — three deliberate steps, none skippable

The one-shot guard now refuses with instructions rather than a bare stop. A v2 push requires:

1. **bump `PUSH_DATE`** in `q38_push.sh` to the actual push day — *the date edit is the moment you re-read the ledger, not a formality*;
2. **`Q38_ALLOW_V2=1`** in the environment;
3. **a free slot re-confirmed from `ITERATION_LOG.md` for that date (§11.4 discipline)**, which the script prints and refuses without.

Then `bash duck_eval/q38/q38_push.sh --dry-run` and, only after reading the ledger excerpt, `--confirm-push`.

**The read is unchanged and still sealed: CONFIRM-2× ≥ 32 levels · REFUTE-2× ≤ 25 · HARM ≤ 12 · INFRA DEATH.** No constant in `q38_score.py` was touched while diagnosing v1 — fixing a gate after seeing data is the antipattern the seal exists to prevent, and v1 produced no data to fit to.
