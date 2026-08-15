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
