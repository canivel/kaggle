# arc3-lora-probe-adapters

Two **randomly initialized** LoRA adapters for Qwen3.6-27B-FP8, built to prove the vLLM
`--enable-lora` serve path before any training token is spent. **No training data was used and
no gradient was ever computed.**

| dir | `lora_B` | expected behaviour when served |
|---|---|---|
| `lora-noop/`  | all zeros | output token-identical to the base (measures the LoRA throughput tax with zero confound) |
| `lora-probe/` | ~1e-3 random | output must DIFFER from the base (proves the delta reaches the logits) |

Serving both and comparing separates "loaded and applied" from "silently ignored" — the failure
class Tufa's own `inference/tools/vllm_runtime_lora_guard.py` exists to catch.

Shape/key ground truth taken from the public `iseesmth/duck-harness-nca-qwen36-adapter-20260811`:
128 F32 tensors, 10,485,760 params, r=16, alpha=32, rsLoRA, on `{q,k,v,o}_proj` of the 16
full-attention layers (indices 3,7,...,63). `q_proj.lora_B` is [12288,16] — Qwen3.5 uses gated
attention, so `q_proj` is 2x `num_heads*head_dim`.

Built by `duck_eval/lora/make_probe_adapters.py`.
