# Research: huikang/gpt-oss-120b-aimo3 Fine-Tuned Model

**Date**: 2026-04-01
**Researcher**: Claude (Sonnet 4.6)

---

## Summary

The `huikang/gpt-oss-120b-aimo3` model by Tong Hui Kang is a **publicly available** fine-tuned
variant of `openai/gpt-oss-120b` hosted on Kaggle. It uses the variation label `160a` across
all 20 versions (v1-v20). It is actively used by top-scoring AIMO3 notebooks but is NOT the
dominant model among the absolute top-scoring notebooks. Most 43-44/50 notebooks use the
unmodified base model.

---

## Model Registry Facts

### Primary Model: huikang/gpt-oss-120b-aimo3

- **Kaggle model ID**: 574178
- **Kaggle ref**: `huikang/gpt-oss-120b-aimo3`
- **Author**: Tong Hui Kang (`huikang`)
- **Framework**: transformers
- **Instance slug**: `default` (contains 20 versions, all labeled variation `160a`)
- **All versions public**: Yes (`private: False` for all v1-v20)
- **Size**: ~65.3 GB (version 20: 65,277,136,492 bytes)
- **Latest version**: 20 (as of 2026-03-15)
- **Kaggle path**: `/kaggle/input/models/huikang/gpt-oss-120b-aimo3/transformers/160a/{version}`

### Versions observed in notebooks

| Version | Used by notebook |
|---------|-----------------|
| 20 | `huikang/finetuned-model-on-fork` (huikang's own latest notebook) |
| 16 | `jonathanchan/aimo3-gpt-oss-120b-with-bayesian` (Jonathan Chan Bayesian) |
| 15 | Jonathan Chan (in model_sources alongside v14, v16) |
| 14 | Jonathan Chan (primary), mentioned in competition question |
| 9 | `kienngx/aimo3-oss-120b-inference-only` (139 votes) |

### What "160a" means

The `160a` label is the **variation/instance slug name** within the Kaggle model registry.
This is huikang's own internal naming convention; the meaning is unknown. All 20 versions
share this same variation name, so "160a" refers to the fine-tuning recipe or configuration,
not a single checkpoint version number.

### Other notable model: huikang/huikang-use-only

- **Kaggle ref**: `huikang/huikang-use-only`
- **Versions**: 13 (all public, variation `default`)
- This appears to be a separate fine-tuned model for personal use only. Huikang made it
  public but it is not referenced by other notebooks — likely a private experiment.

---

## Is the Model Public and Usable?

**Yes, it is public.** The `private: False` flag is confirmed for all 20 versions. You can
add it as a model source in any AIMO3 notebook using:

```json
"model_sources": [
    "huikang/gpt-oss-120b-aimo3/Transformers/160a/20"
]
```

And access it in code at:
```python
model_path = '/kaggle/input/models/huikang/gpt-oss-120b-aimo3/transformers/160a/20'
```

---

## What Does the Fine-Tuning Change?

From analyzing the notebooks that use it (especially `kienngx` and `jonathanchan`), the
fine-tuned model is used as a **drop-in replacement** for the base model path. The code
is identical — same Harmony protocol, same vLLM flags, same sampling parameters. The only
change is `model_path`.

This means the fine-tuning presumably improves math reasoning accuracy on IMO-style problems
while maintaining full compatibility with the `openai_harmony` protocol and MXFP4 quantization.

**Training method inference**: The model is ~65.3 GB (same as base), suggesting it is a
full fine-tune (not LoRA). The `160a` label combined with huikang's notebook history suggests
GRPO or SFT on curated math competition problems, potentially including AIMO3 reference data.

---

## Which Notebooks Use huikang's Fine-Tuned Model?

| Notebook | LB Score | Model Used | Notes |
|----------|----------|------------|-------|
| `huikang/finetuned-model-on-fork` | unknown | `160a/20` | Huikang's own latest experiment |
| `jonathanchan/aimo3-gpt-oss-120b-with-bayesian` | ~185 votes | `160a/14,15,16` | Bayesian posterior voting |
| `kienngx/aimo3-oss-120b-inference-only` | 139 votes | `160a/9` | Earlier version |

**Critical finding**: The highest-scoring PUBLIC notebooks (44/50 nihilisticneuralnet and
40/50 ZaynYu) do NOT use huikang's fine-tuned model — they use the base `danielhanchen/gpt-oss-120b`.

---

## Leaderboard Context: Top Scores and Models

Current AIMO3 leaderboard top 5 (as of 2026-04-01):

| Rank | Team | Score | Model (inferred) |
|------|------|-------|-----------------|
| 1 | ippeiogawa | 46 | Private (notebooks not public) |
| 2-4 | just public 44, Batman's Butler, Riku Suzuki, Seungjun Lee | 45 | Unknown |
| 5-20 | Various | 44 | Mix of approaches |

The **#1 scorer (ippeiogawa, 46/50)** has no public AIMO3 notebooks. Their older notebooks
are for ARC and other competitions. The approach is fully private.

**Notable public notebooks by score**:

| Notebook | Public LB | Model |
|----------|-----------|-------|
| `nihilisticneuralnet/44-50-let-me-over-cook` | 44/50 | `danielhanchen/gpt-oss-120b` base |
| `nihilisticneuralnet/43-50-aimo-3-gpt-oss-120b-weighted-entropy` | 43/50 | `danielhanchen/gpt-oss-120b` base |
| `zaynyu/40-50-gpt-oss-120b-tir-dynamictime-kernelpool` | 40/50 | `danielhanchen/gpt-oss-120b` + Qwen3-30B |
| `jonathanchan/aimo3-gpt-oss-120b-with-bayesian` | unknown | `huikang/gpt-oss-120b-aimo3` (160a) |
| `kienngx/aimo3-oss-120b-inference-only` | unknown | `huikang/gpt-oss-120b-aimo3` (160a/9) |

---

## Other Models Beyond GPT-OSS-120B

### GPT-OSS-20B

- **Path**: `danielhanchen/gpt-oss-20b/Transformers/default/1`
- Used as a **secondary model** in some notebooks (Jonathan Chan includes it in model_sources
  but primary inference is on 120B)
- Fits on H100 much faster (much smaller)
- Not known to score competitively alone on 50 IMO-level problems

### Qwen3-30B-A3B (MoE, fp8)

- **Path**: `qwen-lm/qwen-3/Transformers/30b-a3b-thinking-2507-fp8/1`
- Used by ZaynYu in the 40/50 notebook alongside GPT-OSS-120B
- Available on Kaggle with multiple variants including the 235B-A22B version
- Active parameter count: ~3B (very efficient on H100)
- The thinking variant (with chain-of-thought) is available

### Qwen3-235B-A22B (MoE)

- **Path**: `qwen-lm/qwen-3/Transformers/235b-a22b-thinking-2507-fp8/1`
- Very large MoE, 22B active params. May not fit alongside reasoning chains on single H100
- Not observed in top-scoring notebooks

### Fine-tuned GPT-OSS-120B variants (other authors)

| Model | Ref | Notes |
|-------|-----|-------|
| ZaynYu SFT | `zaynyu/gpt-oss-120b-mxfp4-lora-sft/transformers/gpt-oss-120b-mxfp4-sft-nemotron-math-sft-v5/1` | NemoTron Math SFT v5, MXFP4 |
| Muhammad Ibrahim LoRA | `muhammadibrahim3093/gpt-oss-120b-aimo-lora-v1/transformers/merged/6` | Merged LoRA, v6 latest |
| IceBeam777 merged LoRA | `icebeam777/gpt-oss-120b-merged-lora/transformers/gpt-oss-120b-merged-lora-epoch-1/2` | Epoch 1 |
| GPT-OSS 20B NuminaMath | `tensorhydra/gpt-oss-20b-numinamath/...` | 20B SFT on NuminaMath-TIR 70k |
| GRPO Math | `nurikw3/gptoss/transformers/default/1` | GRPO on GMSK dataset |

---

## Jonathan Chan Bayesian Notebook — Key Details

This notebook (`jonathanchan/aimo3-gpt-oss-120b-with-bayesian`) uses huikang's model and adds
a sophisticated Bayesian posterior voting system on top of the standard Harmony TIR approach.

**Model**: `huikang/gpt-oss-120b-aimo3/Transformers/160a/14` (primary)

**Bayesian posterior mechanism**:
```python
def _compute_bayesian_posterior(self, detailed_results):
    for r in detailed_results:
        entropy_weight = 1.0 / (1.0 + entropy)
        reliability = 1.0 / (1 + error_penalty * python_errors)
        tool_bonus = 1.2 if (python_calls > 0 and python_errors == 0) else (0.8 if python_errors > 0 else 1.0)
        weight = entropy_weight * reliability * tool_bonus
        posterior[answer] += weight
```

**Value-of-information (VOI) early stopping**:
```python
submit_utility = max_prob
expected_improvement = entropy * voi_entropy_weight  # 0.6
continue_utility = max_prob + expected_improvement - voi_compute_cost  # 0.04

if submit_utility >= continue_utility:
    stop_event.set()  # stop if expected gain from more attempts < 0.04
```

**Key CFG params**:
- `temperature = 1.0`, `min_p = 0.02`
- `context_tokens = 65536`
- `attempts = 8`, `workers = 16`
- `posterior_stop_threshold = 0.82`
- `gpu_memory_utilization = 0.96`

---

## Recommendation

### Should we use huikang's fine-tuned model?

**Probably yes, worth trying.** The model is public, same size as base (drop-in replacement),
and is used by multiple competitive notebooks. However, the current evidence does NOT show
that it beats the base model:

- The highest public score using huikang's model: Jonathan Chan (score unknown, 185 votes)
- The highest public score NOT using huikang's model: 44/50 (nihilisticneuralnet)

The competition is still open and scores are dynamic. The fine-tuned model may yield a 1-2
point improvement over the base model if the fine-tuning specializes on IMO-style problems.

**Recommended action**: Try version 20 (latest) as a drop-in replacement in our v21/v22
notebook. The risk is low (same code, same protocol) and the potential gain is meaningful.

### Kaggle model source to add:
```json
"huikang/gpt-oss-120b-aimo3/Transformers/160a/20"
```

### Code change (single line in CFG):
```python
# Base model (current):
model_path = '/kaggle/input/gpt-oss-120b/transformers/default/1'

# huikang fine-tuned (to try):
model_path = '/kaggle/input/models/huikang/gpt-oss-120b-aimo3/transformers/160a/20'
```

Note the path difference: base model uses `/kaggle/input/gpt-oss-120b/` (dataset-style),
huikang's model uses `/kaggle/input/models/huikang/gpt-oss-120b-aimo3/` (model-style).

---

## Sources

- Kaggle model registry: `kaggle models list -s "huikang"`
- Kaggle model versions: `kaggle models instances versions list "huikang/gpt-oss-120b-aimo3/transformers/default"`
- Notebook `jonathanchan/aimo3-gpt-oss-120b-with-bayesian` (pulled and analyzed)
- Notebook `nihilisticneuralnet/44-50-let-me-over-cook` (pulled and analyzed)
- Notebook `zaynyu/40-50-gpt-oss-120b-tir-dynamictime-kernelpool` (pulled and analyzed)
- Notebook `huikang/finetuned-model-on-fork` (pulled and analyzed)
- Notebook `kienngx/aimo3-oss-120b-inference-only` (pulled and analyzed, credits huikang)
- AIMO3 leaderboard via `kaggle competitions leaderboard ai-mathematical-olympiad-progress-prize-3 --show`
