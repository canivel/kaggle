# Math-Focused LLMs for RTX 3080 (10GB VRAM) - Local vLLM Inference

Generated: 2026-03-29

## Platform Note

vLLM does not run natively on Windows. On Windows 11, you need **WSL2** or **Docker** to run vLLM. All vLLM commands below assume a Linux environment (WSL2 recommended).

---

## VRAM Estimation Rules

| Precision | GB per 1B params | 7-8B model total | Context headroom |
|-----------|-----------------|-----------------|-----------------|
| BF16/FP16 | ~2.0 GB | ~14-16 GB | None on 3080 |
| INT8 | ~1.0 GB | ~7-8 GB | Marginal (~2 GB for KV) |
| 4-bit GPTQ/AWQ | ~0.5-0.6 GB | ~4-5 GB | Comfortable |
| 3.8B at FP16 | ~7.5 GB | ~7.5 GB | Tight but workable |
| 1.5B at BF16 | ~3 GB | ~3 GB | Excellent |

**Bottom line for 10GB VRAM**: 7-8B models REQUIRE 4-bit quantization. 3.8B fits at FP16 (tight). 1.5B fits comfortably at FP16. INT8 on 7B leaves ~2 GB for KV cache — too tight for long reasoning chains (R1-style models need 32k+ context).

---

## Summary Ranking Table

Evaluation protocol note: AIME 2024 pass@1 = single greedy/temperature sample. RM@256 = best-of-256 with reward model reranking. These are not comparable — RM@256 can be 2-3x higher than pass@1 on the same model.

| Rank | Model | Params | Best Quant for 3080 | VRAM Est | MATH-500 | AIME 2024 | AIME Protocol | TIR/Code | vLLM |
|------|-------|--------|---------------------|----------|----------|-----------|---------------|----------|------|
| 1 | DeepSeek-R1-0528-Qwen3-8B | 8B | AWQ 4-bit | ~5 GB | not reported | 86.0% | pass@1 | Yes | Yes |
| 2 | Phi-4-mini-reasoning | 3.8B | FP16 (fits native) | ~7.5 GB | 94.6% | 57.5% | pass@1 | Python | Yes |
| 3 | DeepSeek-R1-Distill-Qwen-7B | 7B | GPTQ/AWQ 4-bit | ~5 GB | 92.8% | 55.5% | pass@1 | Yes | Yes |
| 4 | Qwen3-8B (thinking mode) | 8B | AWQ 4-bit (official) | ~5 GB | not reported | 80.4% | pass@1 | Yes | Yes |
| 5 | Qwen2.5-Math-7B-Instruct | 7B | BNB 4-bit / self-AWQ | ~5 GB | 85.3% | ~5/30 greedy | pass@1 | Yes (TIR) | Experimental |
| 6 | Qwen2.5-Math-1.5B-Instruct | 1.5B | FP16 (no quant needed) | ~3 GB | 79.7% | not reported | — | Yes (TIR) | Yes |
| 7 | NuminaMath-7B-TIR | 7B | AWQ 4-bit | ~5 GB | 68.1% | 5/30 | pass@1 (0-shot) | Yes (TIR) | Yes |
| 8 | Mathstral-7B-v0.1 | 7B | GGUF only | ~5 GB | 56.6% | 2/30 | maj@16 | No | Limited |
| 9 | DeepSeek-Math-7B-RL | 7B | community GGUF | ~5 GB | 51.7% | not reported | — | CoT only | Yes |
| 10 | GLM-4-9B-Chat | 9B | GPTQ-Int4 | ~6 GB | not reported | not reported | — | Yes | Limited |
| 11 | InternLM2-Math-7B | 7B | FP16 only (8 GB) | ~8 GB | 34.6% | not reported | — | Lean3/Code | Limited |

---

## Detailed Model Profiles

### 1. DeepSeek-R1-0528-Qwen3-8B [BEST MATH ABILITY]

**HuggingFace ID**: `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`

**What it is**: DeepSeek's distillation of R1-0528 onto Qwen3-8B base. Released 2025. Architecture is identical to Qwen3-8B but trained with R1-0528's extended chain-of-thought reasoning traces.

**Benchmark scores**:
- AIME 2024: **86.0% (Pass@1)** — highest among open sub-10B models as of March 2026
- AIME 2025: **76.3% (Pass@1)**
- MATH-500: not separately reported (expected very high given AIME performance)
- Code/LiveCodeBench: strong (inherits Qwen3-8B base)

**TIR/Code support**: Yes. Inherits Qwen3's code generation and tool-calling.

**VRAM at 4-bit**: ~5 GB, comfortable on 3080.

**Quantized variants for vLLM**:
- `hxac/DeepSeek-R1-0528-Qwen3-8B-AWQ-4bit` — **AWQ 4-bit, recommended for vLLM**
- `Intel/DeepSeek-R1-0528-Qwen3-8B-int4-AutoRound-gptq-inc` — AutoRound GPTQ INT4 from Intel
- `QuantTrio/DeepSeek-R1-0528-Qwen3-8B-GPTQ-Int4-Int8Mix` — mixed INT4/INT8 GPTQ (sensitive layers kept at INT8); vLLM support for mixed-precision GPTQ requires vLLM 0.9.0+ with a patch
- GGUF only (not for vLLM): `unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF`

**vLLM command**:
```bash
vllm serve hxac/DeepSeek-R1-0528-Qwen3-8B-AWQ-4bit \
  --max-model-len 32768 \
  --enforce-eager
```

**License**: MIT (commercial + distillation allowed)

**Important**: Temperature 0.6, no system prompt. Prepend `<think>\n` for best reasoning. Budget for long outputs — R1-style models generate 2k-8k thinking tokens before answering.

---

### 2. Phi-4-mini-reasoning [SURPRISE CONTENDER — BEST MATH AT 3.8B]

**HuggingFace ID**: `microsoft/Phi-4-mini-reasoning`

**What it is**: Microsoft's 3.8B reasoning model trained on 1M+ synthetic math problems from DeepSeek-R1. Dense transformer, 128K context. Despite 3.8B params it achieves better MATH-500 than much larger models.

**Benchmark scores**:
- MATH-500: **94.6%** — exceptional for 3.8B, competitive with 70B+ models
- AIME 2024: **57.5% (Pass@1)** — beats DeepSeek-R1-Distill-Qwen-7B (55.5%)
- GPQA Diamond: 52.0%
- AIME 2025: not reported on this variant

**TIR/Code support**: Yes, Python-focused code generation. Reasoning chain uses `<think>...</think>` blocks.

**VRAM at FP16**: ~7.5 GB — fits on 3080 natively, no quantization needed.

**Quantized variants**:
- 34 community quantized models via llama.cpp/LM Studio/Ollama (mostly GGUF)
- No official AWQ or GPTQ from Microsoft; community GPTQ variants may exist
- FP16 is the recommended path for 3080 (7.5 GB leaves ~2.5 GB for KV cache)

**vLLM support**: Yes. Standard transformers-compatible architecture (dense decoder-only).

**License**: MIT

**Why it matters**: At 3.8B you get AIME 57.5% and can run FP16 without quantization. This means no quantization degradation, faster iteration, and ability to run the 1.5B Qwen2.5-Math alongside it in the same 10 GB for ensemble voting.

**Caveats**: vLLM benchmark wasn't explicitly confirmed on the model card. The flash variant (`Phi-4-mini-flash-reasoning`) uses Mamba-hybrid architecture and requires `mamba-ssm` which may complicate vLLM deployment — stick with the standard `Phi-4-mini-reasoning`.

---

### 3. DeepSeek-R1-Distill-Qwen-7B [BEST TESTED OPTION FOR vLLM]

**HuggingFace ID**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

**What it is**: The original DeepSeek-R1 distillation onto Qwen2.5-Math-7B base. Built on the math-specialized backbone with R1 reasoning traces. Well-tested in production vLLM deployments since early 2025.

**Benchmark scores**:
- MATH-500: **92.8% (Pass@1)**
- AIME 2024: **55.5% (Pass@1)**, 83.3% (cons@64 — majority vote over 64 samples)
- LiveCodeBench: 37.6%
- CodeForces Rating: ~1189

**TIR/Code support**: Yes. Long-form reasoning with Python generation.

**VRAM at 4-bit**: ~5 GB.

**Quantized variants (vLLM-confirmed)**:
- `RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16` — GPTQ w4a16, **most widely tested with vLLM**, RedHat-supported
- `kaitchup/DeepSeek-R1-Distill-Qwen-7B-AutoRound-GPTQ-4bit` — AutoRound GPTQ 4-bit
- `ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2` — GPTQModel optimized
- `jakiAJK/DeepSeek-R1-Distill-Qwen-7B_AWQ` — AWQ 4-bit (community)
- `AngelSlim/Deepseek_r1_distill_qwen-7b_int4_gptq` — INT4 GPTQ

**vLLM command** (using best-tested variant):
```bash
vllm serve RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16 \
  --max-model-len 32768 \
  --enforce-eager
```

**License**: MIT

**Why prefer this over R1-0528-Qwen3-8B**: More production-tested quantized variants. RedHatAI's w4a16 is the most battle-hardened vLLM-compatible quantization of any 7B math model. Use this if you need reliable deployment today; use R1-0528-Qwen3-8B when you need maximum math ability.

---

### 4. Qwen3-8B (thinking mode) [OFFICIAL AWQ — EASIEST vLLM SETUP]

**HuggingFace ID**: `Qwen/Qwen3-8B`
**Official AWQ**: `Qwen/Qwen3-8B-AWQ`

**What it is**: Qwen3's 8B dense model with switchable thinking/non-thinking modes. General-purpose but math rivals dedicated math models. Qwen3-8B base is equivalent in capability to Qwen2.5-14B.

**Benchmark scores**:
- AIME 2024: **80.4% (thinking mode, pass@1)** per Qwen3 Technical Report (arXiv 2505.09388)
- MATH-500: not separately reported in available sources
- Code generation: strong (agent capabilities, tool-calling)

**TIR/Code support**: Yes. Full tool-calling, code generation, agent mode. Thinking mode adds `<think>...</think>` blocks.

**VRAM at 4-bit AWQ**: ~5 GB.

**Quantized variants (vLLM-compatible)**:
- `Qwen/Qwen3-8B-AWQ` — **official AWQ 4-bit from Qwen team**, cleanest vLLM path
- `pytorch/Qwen3-8B-AWQ-INT4` — TorchAO variant

**vLLM command**:
```bash
vllm serve Qwen/Qwen3-8B-AWQ \
  --max-model-len 32768 \
  --enable-reasoning \
  --reasoning-parser qwen3
```

**License**: Apache 2.0

**Why it stands out**: Official Qwen-team AWQ quantization makes this the most reliable and easiest vLLM deployment path among all 8B options. The `--reasoning-parser qwen3` flag properly handles thinking mode output parsing. 80.4% AIME in thinking mode is remarkable for a general-purpose model.

**Caveats**: Requires vLLM version that supports `--reasoning-parser qwen3` (vLLM 0.6+). Thinking mode produces significantly more tokens, which affects throughput.

---

### 5. Qwen2.5-Math-7B-Instruct [DEDICATED MATH/TIR, NO CLEAN QUANT]

**HuggingFace ID**: `Qwen/Qwen2.5-Math-7B-Instruct`

**What it is**: Purpose-built math model with first-class TIR support. The base architecture for DeepSeek-R1-Distill-Qwen-7B. Strong on MATH benchmark with TIR, but substantially weaker than R1-distilled variants on AIME.

**Benchmark scores**:
- MATH (CoT): **83.6%**
- MATH (TIR): **85.3%**
- AIME 2024 greedy (pass@1): **~5/30** (approximately 16-17%)
- AIME 2024 with RM@256: 21/30 (note: not comparable to pass@1 scores above)
- AMC 2023: 29/40 (RM@256)

**TIR/Code support**: Yes. First-class TIR with Python/sympy. Uses `\`\`\`output` stop string pattern.

**VRAM at 4-bit**: ~5 GB.

**Quantized variants**:
- NO official AWQ or GPTQ from Qwen (they released GGUF only for this Math variant)
- `unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit` — BitsAndBytes 4-bit (vLLM BNB support is experimental)
- `bartowski/Qwen2.5-Math-7B-Instruct-GGUF` — GGUF (llama.cpp ONLY, not vLLM)
- For proper vLLM: self-quantize with AutoAWQ (takes ~1 hour on GPU)

**vLLM support**: Experimental via BNB. For production: run AutoAWQ locally first.

**License**: Apache 2.0

**Verdict**: The TIR implementation here is the reference implementation and works very well for structured math problems with Python tools. However the weak pass@1 AIME score (~16%) vs. R1-Distill's 55.5% shows the R1 distillation adds enormous value. Use this only if TIR-specific behavior matters more than raw math ability.

---

### 6. Qwen2.5-Math-1.5B-Instruct [BEST FOR FAST MAJORITY VOTING]

**HuggingFace ID**: `Qwen/Qwen2.5-Math-1.5B-Instruct`

**What it is**: 1.5B math specialist. Fits in ~3 GB at FP16 — no quantization needed on 3080. Leaves 7 GB free for KV cache, enabling very long context or parallel batch inference.

**Benchmark scores**:
- MATH (TIR): **79.7%**
- MATH (CoT): ~75%
- AIME 2024: not separately reported (1.5B significantly weaker than 7B on hard problems)
- GSM8K: strong

**TIR/Code support**: Yes. Same TIR interface as 7B variant.

**VRAM at FP16**: ~3 GB — runs alongside other models or with large KV cache.

**Quantized variants** (smaller is already fast):
- `unsloth/Qwen2.5-Math-1.5B-Instruct-bnb-4bit` — ~1 GB VRAM
- `bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF`

**vLLM command**:
```bash
vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --max-model-len 4096
```

**License**: Apache 2.0

**Use case**: Run 64-256 parallel samples for majority voting or self-consistency. At 3 GB VRAM you can batch aggressively. Combine with a process reward model (Qwen2.5-Math-PRM-7B) to pick the best solution.

---

### 7. NuminaMath-7B-TIR [COMPETITION-PROVEN, OLDER BASELINE]

**HuggingFace ID**: `AI-MO/NuminaMath-7B-TIR`

**What it is**: Competition-focused model trained for AIMO (AI Math Olympiad) using ToRA-style TIR. Won the AIMO Progress Prize 1 (29/50). Based on DeepSeek-Math-7B base. The GPTQ variant was explicitly created for Kaggle submissions.

**Benchmark scores**:
- MATH: **68.1%** (0-shot)
- GSM8K: **84.6%** (0-shot)
- AIME 2024: **5/30** (0-shot, pass@1), 10/30 with majority@64
- AMC 2023: 20/40 with majority@64

**TIR/Code support**: Yes, purpose-built. Two-stage training (CoT then Python+feedback). Stop string: `\`\`\`output`.

**VRAM requirements**:
- FP16: ~14 GB (does not fit on 3080)
- `AI-MO/NuminaMath-7B-TIR-GPTQ` — GPTQ-Int8 (~8 GB — marginal, tight for long context)
- `PrunaAI/AI-MO-NuminaMath-7B-TIR-AWQ-4bit-smashed` — AWQ 4-bit (~5 GB — fits well)

**vLLM support**: AWQ variant is vLLM-compatible. GPTQ-Int8 should work. GGUF variants (bartowski, QuantFactory) are llama.cpp only.

**License**: Apache 2.0

**Verdict**: Lower MATH score than all top-3 options. But it was purpose-built for exactly the competition format you're running, and the Kaggle GPTQ variant has a proven deployment path. Worth keeping as a backup or ensemble member.

---

### 8. Mathstral-7B-v0.1 [SKIP]

**HuggingFace ID**: `mistralai/Mathstral-7B-v0.1`

**Benchmark scores**:
- MATH: **56.6%** (greedy), 74.59% with RM@64
- AIME 2024: **2/30** (maj@16)
- GSM8K: **77.1%**

**TIR/Code support**: No. CoT only.

**VRAM at 4-bit**: ~5 GB. No official AWQ/GPTQ; 28 community GGUF variants (llama.cpp only).

**License**: Apache 2.0

**Verdict**: Significantly weaker than Qwen2.5-Math-7B on all benchmarks, no TIR support, no official quantizations for vLLM. Skip entirely.

---

### 9. DeepSeek-Math-7B-RL [HISTORICAL BASELINE]

**HuggingFace ID**: `deepseek-ai/deepseek-math-7b-rl`

**Benchmark scores**:
- MATH: **51.7%** (CoT, greedy)
- GSM8K: **88.2%**
- AIME 2024: not reported

**TIR/Code support**: CoT only. No tool integration.

**vLLM support**: Yes. Community GGUF and base model.

**License**: DeepSeek (commercial allowed)

**Verdict**: Outclassed by every model above it. Useful only as a historical baseline or ablation reference.

---

### 10. GLM-4-9B-Chat [NOT RECOMMENDED]

**HuggingFace ID**: `THUDM/glm-4-9b-chat`
**GPTQ variant**: `model-scope/glm-4-9b-chat-GPTQ-Int4` (~7 GB), `ModelCloud/glm-4-9b-gptq-4bit`

**Benchmark scores**: No math-specific benchmarks publicly reported. AIME: unknown.

**VRAM at 4-bit**: ~6 GB.

**vLLM support**: Available but GLM uses non-standard architecture (bidirectional attention prefix). Less tested than Qwen/LLaMA. The Int8 variant is recommended over Int4 for this model.

**Verdict**: Not math-specialized, no math benchmark data, architecture risk with vLLM. Skip.

---

### 11. InternLM2-Math-7B [SKIP UNLESS YOU NEED LEAN 3]

**HuggingFace ID**: `internlm/internlm2-math-7b`

**Benchmark scores**:
- MATH (greedy CoT): **34.6%** — 2.5x worse than Qwen2.5-Math-7B
- GSM8K: **78.1%**
- AIME 2024: not reported

**TIR/Code support**: Lean 3 formal proof generation + Python code interpreter via lagent. The Lean 3 support is unique.

**VRAM**: FP16 ~16 GB; load with `torch_dtype=float16` for ~8 GB (marginal on 3080). Only 3 community GGUF variants; no AWQ or GPTQ.

**Verdict**: 34.6% MATH is far too weak for competition use. Only use if formal proof verification (Lean 3) is specifically required.

---

## Recommended Configuration for RTX 3080

### Option A: Maximum math ability (competition use)
```bash
# Best AIME performance at 4-bit, WSL2
vllm serve hxac/DeepSeek-R1-0528-Qwen3-8B-AWQ-4bit \
  --max-model-len 32768 \
  --enforce-eager
# ~5 GB VRAM, AIME 2024 pass@1 = 86%
```

### Option B: Best-tested production path
```bash
# RedHat's w4a16 is most battle-hardened for vLLM
vllm serve RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16 \
  --max-model-len 32768 \
  --enforce-eager
# ~5 GB VRAM, AIME 2024 pass@1 = 55.5%
```

### Option C: Official AWQ, easiest setup
```bash
# Official Qwen AWQ, guaranteed vLLM compatibility
vllm serve Qwen/Qwen3-8B-AWQ \
  --max-model-len 32768 \
  --enable-reasoning --reasoning-parser qwen3
# ~5 GB VRAM, AIME 2024 pass@1 = 80.4%
```

### Option D: No quantization, strong math (3.8B)
```bash
# Phi-4-mini-reasoning at FP16 — 7.5 GB, no quant degradation
vllm serve microsoft/Phi-4-mini-reasoning \
  --max-model-len 16384
# ~7.5 GB VRAM, AIME 2024 = 57.5%, MATH-500 = 94.6%
```

### Option E: Fast majority voting (cheap ensemble)
```bash
# 1.5B at FP16, 3 GB — batch many samples
vllm serve Qwen/Qwen2.5-Math-1.5B-Instruct --max-model-len 4096
# ~3 GB VRAM, run with n=64 for self-consistency
```

---

## Key Findings

**1. R1-distillation models dominate 7-8B math**: DeepSeek-R1-0528-Qwen3-8B (86% AIME pass@1) is in a different league from Qwen2.5-Math-7B-Instruct (~16% AIME pass@1). The distillation from full R1 thinking traces adds ~40+ points on AIME at this size.

**2. Phi-4-mini-reasoning is the surprise**: At only 3.8B params, it achieves 94.6% MATH-500 and 57.5% AIME 2024 pass@1 — better than DeepSeek-R1-Distill-Qwen-7B on MATH-500 and competitive on AIME. Fits FP16 in 7.5 GB. Best choice if you want maximum math without quantization tradeoffs.

**3. AIME evaluation protocol matters critically**: Never compare RM@256 scores against pass@1 scores. Qwen2.5-Math-7B's "21/30 AIME" is RM@256 (best of 256 scored attempts); its greedy pass@1 is ~5/30. The R1-distilled models report pass@1 = 55-86%.

**4. Qwen2.5-Math-7B has no official GPTQ/AWQ**: The Qwen team only published GGUF for this variant. For vLLM you must use experimental BNB or self-quantize. This is a real deployment friction point.

**5. Qwen3-8B-AWQ is the safest official path**: Qwen publishes official AWQ for Qwen3-8B. 80.4% AIME in thinking mode. Easiest one-command vLLM deployment.

**6. GGUF is not for vLLM**: All bartowski/QuantFactory/lmstudio-community/unsloth-GGUF variants are llama.cpp format only. For vLLM you need AWQ, GPTQ (w4a16 or standard), or BNB.

**7. Qwen3-Math does not exist as a separate family**: There is no dedicated `Qwen/Qwen3-Math` model. Qwen3's general models absorbed the math specialization; Qwen3-8B in thinking mode is already stronger than Qwen2.5-Math-7B.

**8. NuminaMath-TIR is competition-proven but aging**: Its 68.1% MATH is now well below state-of-the-art, but it has a Kaggle-specific GPTQ deployment story and won AIMO Progress Prize 1. Useful as ensemble member, not primary solver.

---

## Sources

- [Qwen2.5-Math HF Model Card](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
- [Qwen2.5-Math Blog](https://qwenlm.github.io/blog/qwen2.5-math/)
- [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3 Technical Report arXiv 2505.09388](https://arxiv.org/pdf/2505.09388)
- [Qwen3-8B HF](https://huggingface.co/Qwen/Qwen3-8B)
- [DeepSeek-R1-Distill-Qwen-7B HF](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [DeepSeek-R1-0528-Qwen3-8B HF](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)
- [Phi-4-mini-reasoning HF](https://huggingface.co/microsoft/Phi-4-mini-reasoning)
- [Phi-4-mini-flash-reasoning HF](https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning)
- [NuminaMath-7B-TIR HF](https://huggingface.co/AI-MO/NuminaMath-7B-TIR)
- [NuminaMath-7B-TIR-GPTQ HF](https://huggingface.co/AI-MO/NuminaMath-7B-TIR-GPTQ)
- [Mathstral-7B HF](https://huggingface.co/mistralai/Mathstral-7B-v0.1)
- [DeepSeekMath arXiv 2402.03300](https://arxiv.org/html/2402.03300v3)
- [AIME 2024 Leaderboard](https://llm-stats.com/benchmarks/aime-2024)
- [RedHatAI DeepSeek-R1-Distill w4a16 HF](https://huggingface.co/RedHatAI/DeepSeek-R1-Distill-Qwen-7B-quantized.w4a16)
