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

---

## 12. PUSH RECORD — v2, 2026-08-16 (append-only; nothing above was edited)

**`canivel/arc3-q38-engine-eval` version 2 PUSHED and RUNNING.** 08-16 **slot 1 of 2**; slot 2 remains free. All three §11.6 steps were performed deliberately and in order: `PUSH_DATE` bumped 08-15 → 08-16 (and the stale 08-14/08-15 accounting prose in gate 0b rewritten to today's, so the script no longer prints a false statement about the slot budget), `Q38_ALLOW_V2=1`, and the free slot re-confirmed from the printed `### 2026-08-16` ledger excerpt.

**Pre-push gates, run twice (dry-run then live), identical both times:** artifact `code_sha256=8babf6de9934c3e5`, cells 17, differing cells `[2,6,8]`, **smoke 109 passed / 0 failed**, **scorer selftest 22 passed / 0 failed**, idempotence guard cleared. This is a byte-exact match to the v2 fingerprint sealed in §11 — *the thing pushed is the thing that was authorized.*

### 12.1 Step 3 aborted on `CODE MISMATCH`. **The artifact was right and the VERIFIER was wrong — twice.**

The pull-back verify threw `AssertionError: CODE MISMATCH` on a sha compare. It was **not** a race: a second independent pull reproduced it exactly, and the difference localised to **one code cell, 527 chars local vs 529 remote**.

**Defect 1 — non-ASCII round-trip mangling of a BASELINE byte.** Codepoint census: local carries a single `U+2014` EM DASH at offset **471**; remote carries `U+00E2 U+20AC U+201D` at the same offset — the textbook signature of UTF-8 bytes re-read as cp1252. Kaggle's push path did it, as §0- always said it does. **The decisive fact: that em-dash is the FROZEN FORK's own byte.** `notebooks/duckfork/tufa-labs-duck-harness-june-30-milestone-winner.ipynb` cell 16 is 527 chars with `U+2014` at offset 471 — identical. It is in a `print()` in the trailing diagnostics cell, and cell 16 is **not** one of our arm cells (`--expect-diff-cells 2,6,8`). **ASCII-hardening it would have manufactured a fourth differing cell and broken precisely the baseline byte-identity D2/D3/D4 exist to protect.** So the artifact must not change. Corroboration that this reading is not self-serving: **`preflight.py` D4 independently reached it** — `1 cell(s) at [16] differ ONLY by non-ASCII pull round-trip mangling (treated as equal)`. Preflight had been hardened for this class; step 3 had not.

**Defect 2 — an assert that predated the gate it was breaking.** After Fix 1 the run reached `assert "vrfai/Qwen3.6-27B-FP8" not in remote_src.replace(...)` and failed. Cause: the incumbent name now occurs **twice**, and the second site is **`Q38_VETO = ('vrfai-qwen3-6-27b-fp8-hf-snapshot', 'vrfai/Qwen3.6-27B-FP8')`** — v2's **poisoning gate**, the §11.2 negative control whose entire job is to make a silently-served incumbent fatal. The assert was written against v1, before that constant existed. **Local and remote both have exactly 2 occurrences at exactly these 2 sites** — i.e. the check was demanding the deletion of the gate that protects the measurement.

**Both fixes are to `q38_push.sh` step 3, not to the artifact, and neither relaxes anything load-bearing.** Fix 1 prefers an exact match and falls back to ASCII-normalised equality — *any ASCII-visible drift is still fatal*. Fix 2 **enumerates the two licensed sites, asserts each is PRESENT, and forbids a third** — strictly stronger than the original, which would have passed a notebook with the veto tuple deleted. The repaired step 3 was then **extracted verbatim from the script and replayed against the live remote**: `code MATCH after non-ASCII round-trip normalisation` · `pull-back OK … poisoning gate (Q38_VETO) intact`, exit 0.

### 12.2 The load-bearing verification — which the abort had skipped, and which is the real lesson

The abort fired *before* the dataset/env block, so the push briefly stood with **only** a sha comparison done. Re-run manually and then via the repaired script, all of it passes: **`dataset_sources` = 3/3 with the 25 GB engine `saltb0x/qwen3-8-27b-fp8` present** (the silent-drop failure mode of `feedback_kaggle_model_attach`, and the one thing most likely to void the arm), incumbent snapshot **absent**, `enable_gpu=True`, `enable_internet=False`, `machine_shape=NvidiaRtxPro6000`, `competition_sources=[arc-prize-2026-arc-agi-3]`, docker sha `…4cb13c`, and all six engine/pin tokens present including `"reasoning_effort": "medium"`. **Step 4 preflight: `ALLOW`, 0 fails / 0 warns / 5 n/a, D3 `17 cells, 8 code`, D4 `[2,6,8] MATCHES`.**

**The generalisable defect: a verifier that aborts on its cheapest, least load-bearing check never runs its most load-bearing ones.** A cosmetic em-dash in someone else's cell suppressed the 25 GB-engine attachment check. Ordering matters in a gate suite, and `assert` gives none — this is the same family as `feedback_audit_the_instrument`: *the instrument was the defect, three times in three days.*

**Status at time of writing: `KernelWorkerStatus.RUNNING`.** The read is unchanged and still sealed — **CONFIRM-2× ≥ 32 levels · REFUTE-2× ≤ 25 · HARM ≤ 12 · INFRA DEATH.** No constant in `q38_score.py` was touched. If it ERRORs, the post-mortem starts with `kernels logs` on **CLI 2.2.3**, never `kernels output`.

---

## 13. DISPOSITION ANNEX — sealed 2026-08-16, **BEFORE the kernel completed and before any output was pulled**

Panel round 26 directive #1 (**5/5 reviewers, the only unanimous item**). **No constant in `q38_score.py` was touched, no
threshold moved, nothing unsealed.** This annex adds only *operating characteristics* — what the sealed design can and
cannot detect — which is a **pre-data power audit, not a post-hoc revision**. `prog-synthesis` put the distinction
correctly: *"sealing protects constants from post-hoc motion, not from a pre-data power audit."* Kernel status at the
moment of writing: **`KernelWorkerStatus.RUNNING`**.

### 13.1 The disposition table

Thresholds as sealed (REFUTE-2× `Δlc ≤ +0.250`, CONFIRM-2× `Δlc ≥ +0.500`), `SE(Δ) = σ̂·√(1+1/3) = 0.163667` lc/game from
§4.3's pre-registered `σ̂ = 0.141740`. **Independently recomputed from the sealed constants, not copied:** the `δ = 0.773`
row reproduces §4.3's own **95.3% / 0.07%**, which is the check that this arithmetic and the seal are the same object.

| true Δlc (lc/game) | ≈ levels over 25 | P(REFUTE-2×) | P(INDETERMINATE) | P(CONFIRM-2×) |
|---|---|---|---|---|
| 0.000 (null) | 19.3 | **93.67%** | 6.22% | 0.11% |
| 0.100 | 21.8 | **82.03%** | 17.24% | 0.73% |
| 0.250 | 25.6 | 50.00% | 43.67% | 6.33% |
| 0.468 (80%-power MDE) | 31.0 | 9.14% | 48.61% | 42.25% |
| 0.773 (a true 2×) | 38.6 | 0.07% | 4.70% | **95.23%** |

**80%-power MDE floor = 0.4684 lc/game = 11.7 levels over 25.** §4.3 already said this out loud before the push — *"well
powered against a doubling, badly powered against small effects."* **The panel's shared "nobody did a power analysis"
framing is therefore ~80% false and must not enter the log:** the MDE was sealed on 08-15; what was missing was only the
δ-grid, which is this table.

### 13.2 THE PRE-COMMITTED READING — binding, written before the data exists

> **A REFUTE-2× reading kills the "2× on the local 25" claim on our harness at better than 3σ, and says NOTHING about the
> engine's marginal value at a +0.17-class effect, which sits below this design's 11.7-level MDE and is therefore
> UNMEASURED by this arm.**

Unless the true effect is **≥ ~+0.25 lc/game, REFUTE-2× is the modal outcome BY PRE-REGISTERED DESIGN.** That is neither a
surprise nor a power failure: the arm was sealed as a test of a **doubling**, and the verdict is literally named
`REFUTE-2×`. It refutes **the claim**, not **the engine** — and it does not impeach our power either, because the power
was declared honestly before the push. Per `systems`' prescription, if the realised read is REFUTE-2× it is additionally
logged **`UNDERPOWERED-AT-PRIOR`**, because the day's re-anchored prior (**+0.17-class**, from the only two dateable
community points: Ya Xu 1.30→1.47, FOYSAL 1.61→1.61) sits **below the MDE**.

### 13.3 The LB conversion is NOT DERIVABLE — say so rather than inventing a ratio

There is no defensible Δscore → Δlc conversion. The baseline's **within-null** `mean_score` spread is **1.427 / 1.939 /
3.420** — sd **1.033 on n=3, a 2.4× spread *inside the null*.** This is why §4.4 already ruled the score-based reading
carries no verdict. Any "+0.17 LB ⇒ X levels" claim is fabrication.

### 13.4 Separately, for the LB arm (not this kernel)

Recomputed from `runs/ledger.json` (n=33, mean 0.9424, s 0.1563), bar **1.0826**, mean-of-4 σ/√4 = **0.0781**: a true
**+0.17** shift moves the draw mean to **1.1124** and clears the bar only **64.9%** of the time; under the null it clears
**3.6%**. `systems`' figure (*"~64%"*) is **verified correct**. So even a *real* +0.17-class engine gain is closer to a
coin-flip than a promotion on a single mean-of-4 window.

### 13.5 Secondary read added per OQ-4 (methodology) — strictly descriptive

A **per-game** secondary read is added to guard Le Grand's public/private-split risk. It is **explicitly non-inferential**:
it carries **no verdict**, must **never** contaminate the primary read, and any per-game claim requires multiplicity
correction across the ~25 games. Recorded here so it cannot later be presented as a finding.

---

## 12. THE VERDICT — v2 COMPLETE, 2026-08-16. **REFUTE-2× (decisive).** (append-only)

```
Q38 ENGINE-SWAP EVAL - SEALED VERDICT: REFUTE-2x (decisive)
decisive: True
reason:   mean dlc +0.0667 <= +0.2500: the 'consistent 2x on the local 25' claim
          is NOT reproduced on our harness
  levels_completed  21 over 25 games (0.8400/game)
  baseline (m=3)    19.33 levels (0.7733/game)  [duck-harness-kaggle 18/19/21]
  mean dlc          +0.0667 /game  (+1.67 levels over 25)   z = +0.41
  lines             HARM <= -0.2863 | REFUTE-2x <= +0.2500 | CONFIRM-2x >= +0.5000
  windows_drifted   0/25
```

Run clean: 25/25 games, `state=gave_up` on all (normal), zero window drift, no runtime fatals, `BOOT-ASSERTS PASSED`. **The instrument fixes worked** — the reclassified probes let the run proceed, and `_q38_observe` gave us the per-turn evidence that made §12.2 diagnosable.

**The engine-generation hypothesis failed its own named primary falsifier.** §8.1 said: *"We screen Qwen3.8 and it does not beat the frozen fork. Then the engine is not the driver."* It did not beat the frozen fork. **+1.67 levels on a line that needed +12.7.**

### 12.1 The score-based secondary carries NO verdict — holding the line declared in §4.3

`mean_score 2.795` vs baseline `1.427 / 1.939 / 3.420` (sd 1.033, n=3). **This is not a win and may not be reported as one.** It sits *below* the best single baseline run. The pre-registered power of a score-based 2× test was ~60 %, below the SCREEN_PROTOCOL bar, and that was decided before the data existed precisely so this number could not be spun afterwards. The median tells the same story: **0.40 arm vs 0.47/0.00/0.00 baseline.** Descriptive only.

### 12.2 ft09 did NOT fail to run. It ran for the full window and never took a single action.

The coordinator's read of the summary line (`actions=0, tokens=0`) understates it. The benchmark record says `final_wallclock_seconds: 7920.3`, `solver_note: tokens=95317`, `history: []`. The summary's token column sums `history[].generated_tokens`, and the history is empty — **so `tokens=0` is a reporting artifact; ft09 actually burned 95,317 generated tokens.**

Pulled the transcript selectively (`kernels output --file-pattern ft09` — 5.3 MB, no site-packages) and it is unambiguous:

- **64 turns, every one a clean `python` tool call.** `tool_call_count: 1`, `finish_reason: tool_calls`, `tool_call_markup_in_text: no` on all of them. **The harness and the tool-call parser worked perfectly.**
- **Zero action calls in the entire 1.4 MB transcript** — `ACTION1`…`ACTION6` appear literally 0 times.
- **`content_chars: 0` on all 64 turns**; `reasoning_chars` mean **2,760**, max 4,938.
- The final turn is still writing exploratory analysis (`# Let me understand the target patterns more carefully`).

**This is analysis paralysis, not a harness failure, not a timeout, not an ordering artifact.** The agent spent 2 h 12 m inspecting the grid and never decided to act. **No baseline run has a single zero-action game across all 75 game-runs.** It is new behaviour under this engine.

**Sensitivity — the verdict is robust and ft09 does not rescue it.** Baseline ft09 mean lc is 2.333 (joint-highest game, lc 2/2/3). Crediting the arm with:
- ft09 at baseline mean → 23.33 levels, Δ = +0.160/game, z = +0.98 → **still REFUTE-2×**
- ft09 at its best-ever baseline (3) → 24 levels, Δ = +0.187/game → **still REFUTE-2×**

CONFIRM needs **32**. **21 is a floor, and even the most generous floor-lift lands 8 levels short.**

### 12.3 The real finding: the engine is better PER ACTION and takes 36 % fewer actions

| | arm (Qwen3.8) | baseline m=3 | ratio |
|---|---:|---:|---:|
| levels completed | 21 | 19.33 | 1.09 |
| **total actions** | **2,857** | **4,474** | **0.64** |
| total generated tokens (solver-note) | 2,217,229 | 1,654,728 | 1.34 |
| **tokens per action** | **776** | **370** | **2.10** |
| **levels per action** | **0.00735** | **0.00432** | **1.70** |
| tok/s (job wallclock) | 237.83 | 204.55 | 1.16 |

**The engine converts each action into ~1.7× more progress and spends ~2.1× more tokens getting there. On a wallclock-bound rail those cancel almost exactly, and the level count barely moves.** That is a mechanism, and it answers the coordinator's question 2 directly: **this is "better but throttled", not "no better."**

The per-game pattern is consistent with it. Where the arm kept acting it gained: `sb26 +2.00` (153 actions, 4 levels), `re86 +1.33`, `sc25 +1.33`, `su15 +1.00`, `cd82 +1.00`, `ls20 +1.00`. Where it thought itself to a standstill it lost: `ft09 -2.33` (**0** actions), `tu93 -1.67`, `vc33 -1.33` (49 actions vs 139), `sp80 -1.00`, and `lp85` burned **5,096 tokens per action** across just 19 actions. **14 of 25 games sat at 93k–98k tokens** against a baseline that clustered at 64k–68k.

**Caveat, stated rather than buried:** levels-per-action is a post-hoc ratio on one seed with a small integer numerator. It is descriptive and it is NOT in the sealed read. It is a hypothesis with a named test — see §12.5.

### 12.4 Throughput: real-looking, not decisive, and CONFOUNDED — we still have no clean tok/s

237.83 vs 204.55 (baseline sd 7.62, n=3): t = 3.78 on df=2, **two-sided p ≈ 0.063, one-sided ≈ 0.032**. Suggestive, not decisive at m=3 — and it is **not in the sealed read**.

**More importantly it is not a decode-speed measurement.** `generated tokens/sec (job wallclock)` is *total generated tokens ÷ job duration*, and both runs occupied the same ~7,960 s job. So the metric is arithmetically just "how many tokens were generated", and the arm generated more **because it thinks longer per turn**, which §12.3 establishes independently. The +16 % is therefore consistent with a 29 % smaller weight footprint *and* equally consistent with pure verbosity, and this run cannot separate them.

**Say it plainly: this is the third consecutive model-level lane to end without a clean tokens/s number** (b122 obtained zero; A17 anchored only the 27B). A standing instrumentation gap, now named.

### 12.5 THE RECKONING — what actually explains the leaderboard, given this result

**The strong form is dead.** *"The 2.5+ regime IS the engine; drop it in and get 2×"* is refuted at our own pre-registered 3.2σ line, on our own harness, with the engine verified serving (`Q38-EVAL served=Qwen/Qwen3.8-27B-FP8`, config asserts passed, pin certified). No hedging: **§2 of `research_restart_2026-08-15.md` does not survive as written.**

**But the diffusion prediction in §8.2 is being CONFIRMED, and that is the tension worth sitting with.** The dated falsifier was: *"if by 2026-08-20 the 1.5–1.65 band is still flat and the top is still exactly these seven teams, the engine explanation is wrong."* Today's pull (`runs/lb_daily/lb_full_2026-08-16.csv`, n=2345) against 08-15 (n=2331):

| band | 08-15 | **08-16** | Δ in 24 h |
|---|---:|---:|---:|
| ≥ 1.90 | 5 | **7** | +2 |
| ≥ 1.75 | 7 | **10** | +3 |
| ≥ 1.62 | 15 | **20** | +5 |
| ≥ 1.44 | 61 | **73** | **+12** |
| p99 | 1.58 | **1.61** | +0.03 |
| median | 0.25 | 0.25 | 0 |

**The wall is migrating upward exactly as predicted, and two new teams crossed 1.90** (`Fufront-RyanX-AGI-Team` 2.25 — straight into #3 — and `aRc (binary relation)` 1.91, plus `Ryan #3` at 1.88). The median is still 0.25: it remains a top-of-board phenomenon. **We are #130 of 2345 at an unchanged 1.33; we fell 11 more places overnight without doing anything wrong.**

So the board says a real capability is diffusing, and our controlled test says the engine alone does not deliver it *in our harness*. Three readings survive, ranked:

1. **ENGINE × HARNESS, and our harness is the limiting factor — best supported, and supported by our own numbers.** The engine really is better per action (1.70×). Our rail is wallclock-bound, so a 2.10× token cost per action eats the entire gain. A team whose harness spends fewer tokens per action — shorter prompts, tighter turn structure, a smaller context, or simply not re-sending 32k of history every turn — converts the same engine into real levels. **This is not a rescue narrative; it is the measured mechanism, and it makes the engine a necessary-but-not-sufficient component rather than the cause.**

2. **The knob we deliberately deferred is now the leading candidate for the difference.** We pinned `reasoning_effort: medium` to isolate the weights — correctly, and I would do it again. But the field is running the **`xhigh` default or tuning it**, and the one value we did not test, **`low`** (*"keep your thinking brief and focused, moving directly to the conclusion"*), attacks precisely the term that cancelled our gain. **The isolation that made our test clean is also what makes it silent about the configuration everyone else is running.** That is an honest limitation of a well-designed experiment, not a flaw in it.

3. **Our harness throws away 100 % of the model's thinking, and the better the thinker the worse that hurts.** ft09 showed `content_chars: 0` on all 64 turns with 2,760 chars of reasoning each. The harness stores it as `assistant_message["reasoning"]`, but the Qwen template reads **`reasoning_content`** — so on replay the `<think>` block renders **empty** (§2, established pre-push). We pay 2.1× the tokens for reasoning and then discard every one of them before the next turn. **This is our own long-standing "the agent FORGOT" root cause, now quantified and made more expensive by a better model.** A better thinker whose thoughts are thrown away is not a better agent.

**What this does NOT explain, and I am not going to pretend otherwise: cstl.** 2.70, banked 43 h *before* the release, flat for five days, still #1, zero public artefacts. Nothing in this run touches it. It was chronologically immune to the engine story and it is immune to this refutation too.

**Standing correction to the record:** the census conclusion the research restart "partially obsoleted" — *"within a fixed engine generation the model explains none of the variance; harness and agent policy are the entire public variance"* — **survives this test intact, and now spans a generation boundary.** We swapped one engine generation for the next, cleanly, and the harness ate the difference.

### 12.6 Registered before any follow-up is proposed

- **21 levels is a floor** (ft09 zero-action), and the verdict is robust to every generous correction of it.
- **Levels-per-action (1.70×) and tokens-per-action (2.10×) are descriptive, one seed, post hoc.** They are the strongest thing this run produced and they are **not** sealed results.
- **No clean decode-rate measurement was obtained.**
- **A CONFIRM would have licensed a follow-up; a REFUTE licenses nothing** except the cheap arms §12.5 names. Sunday slot 2 stays free pending the coordinator's decision.

---

# PART II — THE TOKEN-COST ARM (`arc3-q38-low-eval`), prereg sealed 2026-08-16 BEFORE building

## 13. ★ CORRECTION FIRST: AUTHORIZED CHANGE #1 IS A NO-OP. I WAS WRONG.

The coordinator authorized two changes on a premise **I supplied**, and verifying it before building refuted it. Recording this before anything else, because half the authorized arm has just evaporated.

**The claim I made** (prereg §2, §12.5 reading 3, and the ITERATION_LOG entry for 08-16):
> *"Our harness stores `assistant_message["reasoning"]`; the Qwen template reads `reasoning_content` — so the replayed `<think>` block renders EMPTY. We pay 2.1× for thinking and discard 100 % of it."*

**That is false.** vLLM 0.19.0 bridges the two keys itself. `vllm/entrypoints/chat_utils.py :: _parse_chat_message_content`:

```python
reasoning = message.get("reasoning")
...
if role == "assistant":
    if reasoning is not None:
        result_msg["reasoning"] = cast(str, reasoning)
        result_msg["reasoning_content"] = cast(str, reasoning)   # keep compatibility
```

Confirmed end-to-end, not just read: rendering the **real Qwen3.8 template** with the **exact message the harness appends** (`{"role":"assistant","reasoning":…,"tool_calls":[…]}`, tool_agent.py:1969 + 2010) gives an **empty** `<think>` block **before** normalisation and the **full reasoning text inside `<think>`** after it. The served prompt gets the reasoning.

**Same error class as the last two.** I verified the harness end (`reasoning` is set, `reasoning_content` never is) and the template end (`reasoning_content` is what it reads) and never checked the path between them. That is the third time: `/tokenize` existed but not at the path I called; the MM probe's field existed but the tokens went elsewhere; here both keys existed but the server already reconciles them. **"Verified existence, not call shape" is now a named recurring defect of mine, and the countermeasure is the same each time — test the whole path, not the endpoints.**

**Consequences, all of them:**
- **Change #1 is struck.** It would have been a null edit shipped as a fix, and if the arm had improved we would have credited it.
- **§12.5 reading 3 is WITHDRAWN.** "The harness throws away 100 % of the model's thinking" is not true.
- **The arm is now SINGLE-VARIABLE by construction**, and the coordinator's confound dilemma dissolves. No 2×2 is needed and none is being justified.

### 13.1 What the real mechanism is — measured, and it is better than the one I withdrew

Re-reading the ft09 transcript for the *actual* per-turn accounting (`[ANALYZER STATUS]`):

| | ft09 (0 levels, **0 actions**) | sb26 (4 levels, 153 actions) |
|---|---|---|
| analyzer turns | 61 | 87 |
| `step_executed: True` | **0** | 48 |
| yields on `turn_time_budget` | **61 / 61 (100 %)** | 22 |
| mean `reasoning_chars` | 2,760 | 2,784 |

**Every single ft09 turn ran out of its 60-second analyzer budget before executing a step** (`LOCAL_ANALYZER_YIELD_SECONDS=60`, `context_budget_tokens: 31744`). The two games think identically hard per turn; they differ in whether the turn *finishes* inside 60 s. `history_messages` reached 24–27, so the agent was carrying context, not starving for it.

**The binding constraint is a 60-second per-turn wallclock yield, and thinking length decides how much of it is spent before acting.** That is a sharper, better-evidenced mechanism than the one I withdrew, and it makes `reasoning_effort=low` the exactly-correct lever rather than a plausible one. There is no context-overflow anywhere in the run (`context_overflow_recovered`: 0).

**A second lever is now visible and is deliberately NOT being shipped:** `LOCAL_ANALYZER_YIELD_SECONDS` itself. It is a harness config change, it would be a second variable, and it belongs in its own arm. Registered, not taken.

---

## 14. THE ARM

**`canivel/arc3-q38-low-eval`** (fresh slug), built from the same frozen duck fork.
**The single change vs the run we just completed: `reasoning_effort` `"medium"` → `"low"`.**

Everything else is byte-identical to the Q38 arm: same engine (`saltb0x/qwen3-8-27b-fp8`), same wheelhouse, same 18 invariants, same parsers, same 65536/32768 windows, same sampling, same `MULTIMODAL_UPSCALE=4`, cells 12 and 14 untouched.

`low` injects: *"Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."* Measured render delta vs `medium`: **+138 chars of system prompt** and nothing else.

**One added instrument, report-only:** the decode-rate probe (§16). It runs in cell 8 *before* the bench, so it consumes kernel time (~2 min) and **zero measurement window**.

### 14.1 One arm or two — the decision, and why the question changed

The coordinator leaned "ship together, we lack the slots for a clean 2×2", and explicitly invited disagreement. **I do not need to disagree: the correction in §13 makes this a single-variable arm automatically.** Change #1 is not being deferred as a judgement call — it is being struck as a null edit. The arm carries one variable because there is only one real change left, which is the strongest position available and costs nothing.

---

## 15. THE READ — sealed before building

**Primary is ACTIONS and LEVELS. Tokens/s is explicitly NOT primary** (§12.4: job-wallclock tok/s cannot separate speed from verbosity, demonstrated three lanes running).

### 15.1 Two comparators, and they answer different questions

| | levels | actions | tokens/action |
|---|---:|---:|---:|
| **B1** `duck-harness-kaggle` m=3 (Qwen3.6) | 19.33 (0.7733/game) | 4,474 | 370 |
| **B2** Q38 arm, `effort=medium` (n=1, 2026-08-16) | 21 (0.8400/game) | 2,857 | 776 |

- **vs B1** answers *"does Qwen3.8 + low beat the incumbent?"* — the campaign question. Uses the sealed K3″ machinery (σ̂ 0.141740, df 6, C(3)=2.02).
- **vs B2** answers *"did the knob recover the actions the engine cost?"* — the mechanism question. **n=1 vs n=1: this comparison has NO error model and is DESCRIPTIVE ONLY.** Stated now so it cannot be promoted later.

### 15.2 Sealed lines (in `q38low_score.py`, selftested before the push)

**PRIMARY-A — levels vs B1** (identical thresholds to the Q38 arm; unchanged so the two arms are directly comparable):

| verdict | condition | arm levels |
|---|---|---|
| **HARM** | Δlc ≤ −0.286320 | ≤ 12 |
| **NO-LIFT** | Δlc ≤ +0.250000 | ≤ 25 |
| **LIFT** | Δlc ≥ +0.500000 | ≥ 32 |
| INDETERMINATE | between | 26–31 |

**PRIMARY-B — actions/window vs B2** (the mechanism claim, pre-registered because the whole arm rests on it):

- **ACTION-RECOVERY** iff total actions **≥ 3,665** (= B2's 2,857 + 50 % of the 1,617-action gap to B1's 4,474). Justification: the arm exists to recover the action budget the engine cost; recovering *half* of a 36 % deficit is the smallest effect that would make the mechanism story real rather than rhetorical.
- **NO-RECOVERY** iff actions ≤ 3,100 (< 15 % of the gap).
- **PARTIAL** between.
- Reported alongside: **tokens/action** (B2 776, B1 370) — the term the knob is supposed to move.

**The two primaries are reported jointly and neither overrides the other.** The interesting quadrant is pre-named: **ACTION-RECOVERY + NO-LIFT** would mean the knob buys actions but the shorter thinking gives back the per-action quality — i.e. the 1.70× levels/action was *bought* by the 2.10× tokens/action, and the trade is roughly neutral in both directions. That would be a real finding and it must not be read as failure.

**INFRA DEATH (not decisive)** third state carried forward unchanged.

**Power honesty:** unchanged from §4.3 — the levels read is well powered against large effects (80 % floor = 11.7 levels) and badly powered against small ones. The actions read has **no error model at all** (n=1 vs n=1) and is a threshold on a single draw; it is registered as a **decision rule, not a significance test**, and is labelled as such in the scorer output.

### 15.3 Score-based secondary: still non-inferential, same reasons

B1 mean_score 1.427/1.939/3.420 (sd 1.033, n=3); B2 2.795. ~60 % power, below the SCREEN_PROTOCOL bar. **Reported, never a verdict.** The Q38 arm's 2.795 was not a win and a higher number here will not be one either.

---

## 16. THE DECODE-RATE GAP — closed, cheaply, at last

Named as a standing gap in §12.4 after three consecutive blind lanes. A real fix exists and costs ~2 minutes of kernel time outside the measurement window.

`vllm/entrypoints/openai/chat_completion/protocol.py` supports **`ignore_eos`** and **`min_tokens`**. With `ignore_eos: True, min_tokens: 256, max_tokens: 256` every request emits **exactly 256 tokens**, so tokens/sec becomes arithmetic instead of inference. The probe runs at **fixed concurrency 1 and 8**, thinking off, fixed prompt:

```
Q38LOW-DECODE concurrency=1 tokens=256  elapsed=..s  tok/s=..
Q38LOW-DECODE concurrency=8 tokens=2048 elapsed=..s  tok/s=..
```

**This is a true decode rate**, independent of how verbose the model chooses to be — the exact confound that made 237.83 uninterpretable. **REPORT-ONLY** under the poisoning rule: a slow engine *is* the number. Wrapped so it can never fail the run.

*Caveat registered in advance:* this measures the engine at a synthetic concurrency, not the harness's live 25-game concurrency, and it is not comparable to the historical job-wallclock figures. It is a new series starting at n=1, and the Qwen3.6 point does not exist — obtaining it would cost a slot and is **not** proposed.

---

## 17. ft09 — does either change plausibly move it?

**Yes, and `low` is aimed straight at it.** ft09's pathology is 61/61 turns yielding on the 60-second budget with `step_executed: False`; `low` shortens the thinking that consumes that budget, so it is the one change that could convert timed-out turns into executed steps. **It is worth ~2.3 levels on its own** (baseline mean 2.333; the arm scored 0), which is 14 % of the entire baseline total from a single game. The struck change #1 is irrelevant to it — the agent already had its reasoning and 24–27 messages of history and still never acted, which is precisely why the withdrawn "it forgot" story does not fit.

**Pre-registered, so it cannot be claimed post hoc:** *if the arm posts ANY non-zero action count on ft09, that is direct mechanism evidence for the yield-budget explanation, independent of the primaries.* If ft09 is again 0/0, the yield explanation survives but `low` is too weak a lever for it and the next arm should target `LOCAL_ANALYZER_YIELD_SECONDS` directly. **A zero-action-game count across all 25 games is read from `benchmark.json` offline — no cell-14 change, the measurement surface stays byte-identical.**

---

## 18. OPEN — carried forward, NOT inherited as premises

The coordinator's instruction is explicit and it is the right one: **this arm must not quietly inherit "the engine is why they jumped."** It does not.

1. **The engine-generation hypothesis is REFUTED** (§12) and is not a premise here. This arm tests a knob on an engine we have measured, nothing more. **A LIFT here would say the knob helps us; it would say nothing about why anyone else jumped.**
2. **cstl is unexplained by anything we have tested.** 2.70, banked 43 h before the release, flat six days, still #1, zero public artefacts. Untouched by this arm.
3. **The field is still diffusing past us.** 08-15 → 08-16: ≥1.44 went **61 → 73** in 24 h, two new teams crossed 1.90, p99 1.58 → 1.61, median unchanged 0.25. **We are #130 of 2345 at a static 1.33.** Whatever is spreading, we have not identified it, and this arm is not claimed to.
4. **The 08-13 census stands and now spans a generation boundary:** within a fixed engine generation the model explains none of the variance; harness and agent policy are the entire public variance. The engine swap was clean and the harness ate it. **This arm is a harness-policy change, which is where the census says the variance actually lives** — that, and not the engine story, is its justification.

---

## 19. BUILT AND SEALED — 2026-08-16, no push (append-only)

**`canivel/arc3-q38-low-eval` is ready for 08-17 slot 1.** `code_sha256=29af2aef6b3399d6`, cells [2,6,8] vs the frozen fork, pure ASCII.

| gate | result |
|---|---|
| `q38_smoke.py` (low arm) | **112 / 0** |
| `q38_smoke.py` (medium arm, regression) | **112 / 0** |
| `q38low_score.py --selftest` | **23 / 0** |
| `q38_arm_diff.py` — the one-variable proof | **16 / 0** |
| `q38low_push.sh --dry-run` today | **refuses** (date guard: 08-17, not 08-16) |

**The one-variable proof is mechanised, not asserted.** `q38_arm_diff.py` builds both arms and checks that **cell 8 is byte-identical once the effort literal is normalised — 0 residual character differences across 28,096 chars.** The serve config, all 18 invariants, both probes and the decode-rate instrument are the same bytes; only cells 2 (arm banner) and 8 (the effort word) differ; metadata differs only in `id`/`title`/`code_file`; both arms attach identical `dataset_sources`. Prose cannot enforce "one variable" against a comparator arm; this does.

**Three gates caught real defects while building, which is the point of having them:**
1. the payload lint rejected the decode probe's `max_tokens: ntok` (a name it could not resolve) — fixed by using literals, with a runtime assert keeping `ntok` in sync;
2. the smoke's effort assertions were hardcoded to `medium` and failed the low build — now parameterised, plus a new check that **neither arm contains the other's effort value**;
3. the arm-diff caught nothing, because the first two had already been fixed — recorded so its 16/0 is not mistaken for a gate that never fires.

### 19.1 Two provenance notes, both hazards if left unwritten

- **`q38_push.sh` changed underneath me while I worked.** A concurrent session pushed the engine arm's v2 with it and added a pull-back verifier fix (accept differences confined to non-ASCII codepoints, since the mangled em-dash lives in the **frozen fork's own cell 16** and ASCII-hardening it would manufacture a fourth differing cell and break D2/D3/D4). That fix is correct and is inherited by `q38low_push.sh`. **But the same session had bumped `PUSH_DATE` to 2026-08-16, and my derived script silently inherited today's date for an arm authorized for 08-17.** Caught by reading the derived file rather than trusting the copy. **This is the 08-14 "a push script is SHARED MUTABLE STATE" lesson recurring in a new form: not a duplicate push this time, but a derived artifact inheriting a stale constant.** `q38low_push.sh` is pinned to **2026-08-17** and carries its own one-shot guard (`Q38LOW_ALLOW_V2=1` to override).
- **The local medium-arm notebook has drifted from what actually ran.** `q38_arm_diff.py` rebuilds both arms, so `notebooks/q38-eval/` now contains a medium artifact that *includes the decode-rate probe* and is therefore **not** the v2 that produced the sealed 21-level result. **The pushed v2 is the record; the local file is a comparator.** `q38_push.sh`'s one-shot guard already refuses to push the engine arm again.

### 19.2 Reading it

```bash
kaggle kernels output canivel/arc3-q38-low-eval -p runs/kernel_pulls/q38low_v1 \
  --file-pattern '^(benchmark\.json|summary\.txt)$'
python duck_eval/q38/q38low_score.py runs/kernel_pulls/q38low_v1
kaggle kernels logs canivel/arc3-q38-low-eval > runs/kernel_pulls/q38low_v1/q38low.log
grep -E "Q38-EVAL (DECODE|effort-pin-certified-by|served|BOOT-ASSERTS PASSED|WARN)" runs/kernel_pulls/q38low_v1/*.log
```

**The scorer was validated against the engine arm's real data before sealing**: fed `runs/kernel_pulls/q38_v2` it returns `A=NO-LIFT / B=NO-RECOVERY` and reproduces B2's own figures exactly (776 tokens/action, 0.00735 levels/action, ft09 = 0 actions). An instrument that reads the null correctly is the minimum bar before it is allowed to read the arm.

---

## 20. PUSH RECORD, 2026-08-17 — the low arm is RUNNING, after a misfire that is mine (append-only)

**`canivel/arc3-q38-low-eval` v1 PUSHED and RUNNING, 08-17.** Sequence per authorization: ledger re-confirmed (§11.4 — today's morning-check section showed 0 pushes, slot 1 the coordinator's call, and the coordinator called it), `--dry-run` green (smoke 112/0 low + 112/0 medium regression, `q38low_score` selftest 23/0, arm-diff 16/0), then `--confirm-push`. Pull-back: **effort=low pinned, medium literal ABSENT, decode probe present, 3/3 `dataset_sources` incl. the 25.3 GB engine, env byte-identical, code ASCII-identical, 17 cells.** Preflight: **ALLOW, 0 fails, 0 warns, D4 = [2,6,8] EXACT.**

### 20.1 INCIDENT — the first `--confirm-push` went to the WRONG KERNEL, and the accounting takes the hit

The first confirm-push printed *"Kernel version 3 successfully pushed … arc3-q38-engine-eval"*. **My `NB_DIR_WIN` substitution in the derived script had silently failed to match, no assert checked it, and my post-derivation grep verified `KERNEL=` and `PUSH_DATE=` but not the one variable that actually decides where a push goes.** `kaggle kernels push -p <dir>` pushes whatever the *directory's* `kernel-metadata.json` says; the script's `KERNEL` variable is otherwise just a comment. So the engine arm's slug received a v3 carrying the drifted local medium artifact (medium + decode probe) — exactly the §19.1 hazard, one day later, through the door I had documented but not locked.

- **Slot accounting: 08-17 = 2 of 2 spent.** The unintended engine-v3 push counts, per the 08-14 precedent ("an unexplained push is still a push"). No further push today under any circumstances.
- **engine-eval v3 was left to run, deliberately.** The CLI has no cancel; `kernels delete` would destroy the sealed v2 record and is not an option. What v3 actually is: **a second seed of the medium configuration plus the decode probe** — i.e. an unplanned second draw of PRIMARY-B's n=1 comparator. Whether its result may be *used* (B2 → n=2) is **the coordinator's call, not mine**; it was not pre-registered and I am not folding it into the sealed read unilaterally. Recorded here so the decision is available, not made.
- **Fix shipped, structural not cosmetic: step 1c "push-target integrity"** — the push script now asserts, immediately before pushing, that the target directory's `kernel-metadata.json` `id` equals the script's `KERNEL`. This closes the class (a derived script whose directory and slug disagree), not the instance. It fail-closed twice on its own bugs before passing, which is the correct order to discover them in.
- **Two stale-derivation defects also caught by the run itself:** the step-3 verifier still asserted the *medium* literal (fixed: asserts `low` present AND `medium` absent), and §19's recorded sha `29af2aef` was stale — it predated the decode-probe literal fix; the true artifact sha is **`3c9854ffa3f2e922`** (remote ASCII-identical at `9c86789d…`, the known push-path mangling of the fork's own em-dashes).
- **The pattern, named for the third time in three days: deriving a script by string-replacement inherits every constant the replacement misses, silently.** The 08-16 entry called a push script "shared mutable state"; the general form is that **a derived artifact is a copy of every mistake you did not explicitly overwrite**. The integrity gate is the countermeasure that does not depend on me replacing strings correctly.

### 20.2 Reading it when it lands

Unchanged from §19.2: selective pull, `q38low_score.py`, then `kernels logs` for the `Q38-EVAL DECODE` lines. The sealed lines are §15.2; the ft09 mechanism read is §17; the score-based reading remains non-inferential. Morning-check context worth carrying into the read: **the board moved massively overnight (gold line 1.65 → 2.00, ≥1.90 went 7 → 19, we fell to #175 of 2365 on unchanged bytes), and the field's steps are arriving as single draws on tiny submission histories — whatever is being adopted, it is cheap and transferable, and it is still UNKNOWN.** A LIFT here would be evidence about *our* rail only; it would not identify what the field found.
