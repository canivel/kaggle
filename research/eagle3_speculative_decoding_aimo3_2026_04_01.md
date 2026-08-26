# EAGLE-3 Speculative Decoding for AIMO3 — Research Report
Date: 2026-04-01

## Summary

EAGLE-3 speculative decoding for `openai/gpt-oss-120b` is **real, available, and battle-tested**.
The draft model is on HuggingFace (`wenliang1990/gpt-oss-120b-eagle3-aimo3`), vLLM supports it natively
via a single `--speculative-config` JSON flag, and it was benchmarked specifically on IMO math data.

**Key verdict: Drop-in, lossless, 36-42% throughput gain. Feasible to integrate in <1 hour.**

---

## 1. What Is It

EAGLE-3 is a speculative decoding technique where a tiny draft model proposes candidate tokens,
and the main model verifies them in parallel. If accepted, multiple tokens advance per forward
pass of the main model. The acceptance/rejection logic uses strict speculative sampling conditions
that provably preserve the target distribution — the output distribution is **mathematically identical**
to running the main model alone.

Key EAGLE-3 improvements over EAGLE-2:
- Abandons feature prediction, uses direct token prediction only
- Fuses low/mid/high-level features from 3 layers instead of top layer only (layers 1, 17, 33 for 120B)
- Hard distillation (top-1 teacher token as label) vs. soft KL divergence — reduces memory ~O(B*T) vs O(B*T*V)
- Tool outputs excluded from loss computation (critical for AIMO3's TIR workload)
- 1.4x faster than EAGLE-2 on same models

---

## 2. Reported Speedup

Measured on H800 GPUs with IMO data (same domain as AIMO3):

| Concurrency | Baseline tok/s | Eagle3 tok/s | Speedup |
|-------------|---------------|--------------|---------|
| 8           | 776.5         | 1059.4       | +36.4%  |
| 7           | 686.7         | 956.4        | +39.3%  |
| 6           | 596.6         | 851.6        | +42.8%  |
| 5           | 518.8         | 681.0        | +31.3%  |
| 4           | 465.7         | 657.7        | +41.2%  |
| 3           | 379.5         | 541.3        | +42.6%  |
| 2           | 297.6         | 422.2        | +41.9%  |
| 1           | 190.0         | 268.1        | +41.1%  |

Our notebook runs `batch_size=8` and `workers=8`. At concurrency 8, expect **+36%** throughput.
At concurrency 1 (single-threaded per-problem), expect **+41%**.

Baseten reported even higher: from ~400 to ~650 tok/s (+60%) in a different hardware config.

For AIMO3 specifically: if we currently process ~N tokens per problem, we will process ~1.4x*N tokens
in the same wall-clock time. This means more tool calls, longer reasoning chains, and more attempts
within the 4h55m time limit — all without changing any generation parameters.

---

## 3. Output Distribution

**Unchanged.** EAGLE-3 uses strict speculative sampling with token-level acceptance/rejection.
The paper guarantees lossless output distribution (same as running the main model alone). This is
not approximate — it is exact in expectation. No output quality difference.

The draft model just proposes; the main model accepts or rejects. If rejected, the main model
falls back to its own sampling, so the generation is always consistent with main model's distribution.

---

## 4. Draft Model Availability

### Option A: Community model (wenliang1990, AIMO3-specific)
- HuggingFace: `wenliang1990/gpt-oss-120b-eagle3-aimo3`
- Size: **0.3B parameters**, BF16, safetensors
- Downloads: 3,564/month (actively used)
- Trained specifically on IMO data (same domain as our competition)
- Architecture: 1 Llama decoder layer, hidden_size=2880, head_dim=64, MoE-aware

### Option B: NVIDIA official (long-context optimized)
- HuggingFace: `nvidia/gpt-oss-120b-Eagle3-long-context`
- Size: **0.2B parameters**, BF16
- Acceptance rate: 1.95-2.83 tokens/step on long-context tasks
- Uses TensorRT-LLM (NOT vLLM — incompatible with Kaggle kernel setup)

**Use Option A.** It uses vLLM natively, is the smaller binary, and was specifically trained on
IMO-domain math data with tool-call awareness (tool outputs excluded from training).

### Kaggle Model Registry
Neither model is on the Kaggle model registry as of 2026-04-01. You must add it as a custom
dataset or pre-download in your notebook from HuggingFace.

**To add to Kaggle:**
1. Download locally: `huggingface-cli download wenliang1990/gpt-oss-120b-eagle3-aimo3`
2. Create a Kaggle dataset from the downloaded files
3. Reference it as an input dataset in your notebook

---

## 5. VRAM Requirements

### Main model: openai/gpt-oss-120b
- 120B params, MoE (5B activated per token), quantized to FP8 for KV cache
- Estimated VRAM (bf16 weights): ~60GB → with FP8 KV cache: fits on single H100 80GB
- Current config: `gpu_memory_utilization=0.96` on single GPU (TP=1)

### Draft model: gpt-oss-120b-eagle3-aimo3
- 0.3B params in BF16 = ~0.6GB weights
- Shares KV cache infrastructure with main model
- vLLM's `draft_tensor_parallel_size=1` means draft runs on same GPU
- Total VRAM overhead for draft model: **~1-2GB** (negligible vs 80GB H100)
- KV cache is slightly smaller for draft model (single layer)

### Practical impact
With `gpu_memory_utilization=0.96` and TP=1 (single H100 80GB), the draft model adds
~1-2GB overhead (weights ~0.6GB + separate KV cache blocks for speculative steps at `num_speculative_tokens=3`).
Recommended: drop `gpu_memory_utilization` to 0.93. If OOM still occurs, diagnose the type:
- **Startup OOM** (before first request): weight loading issue — lower `gpu_memory_utilization` further
- **Runtime OOM** (during long sequences): KV cache exhaustion — reduce `max_model_len` or `batch_size`

---

## 6. vLLM Flags — Exact Implementation

The entire change is **one additional flag** to the vLLM server launch command:

```python
"--speculative-config",
'{"method": "eagle3", "model": "/kaggle/input/gpt-oss-120b-eagle3-aimo3", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1}'
```

Full server command (changes from baseline in CAPS comments):

```python
cmd = [
    sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
    '--seed', str(self.cfg.seed),
    '--model', self.cfg.model_path,                    # openai/gpt-oss-120b
    '--served-model-name', self.cfg.served_model_name,  # gpt-oss
    '--tensor-parallel-size', '1',
    '--max-num-seqs', str(self.cfg.batch_size),        # 8
    '--gpu-memory-utilization', '0.93',                # LOWERED from 0.96 to fit draft model
    '--host', '0.0.0.0',
    '--port', str(self.port),
    '--dtype', self.cfg.dtype,                         # auto
    '--kv-cache-dtype', self.cfg.kv_cache_dtype,       # fp8_e4m3
    '--max-model-len', str(self.cfg.context_tokens),   # 65536
    '--stream-interval', str(self.cfg.stream_interval),
    '--async-scheduling',
    '--speculative-config',                            # NEW FLAG
    '{"method": "eagle3", "model": "/kaggle/input/gpt-oss-120b-eagle3-aimo3", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1}'
]
```

The draft model path must be a local directory (not a HuggingFace ID) when running in Kaggle's
offline environment.

### Parameters explained
- `method: eagle3` — tells vLLM to use the EAGLE-3 acceptance algorithm
- `model` — path to the 0.3B draft model checkpoint
- `num_speculative_tokens: 3` — draft proposes 3 tokens per step; main verifies all 3 at once
- `draft_tensor_parallel_size: 1` — draft model uses 1 GPU (same as main model's TP)

### vLLM version compatibility
The benchmark.py uses vLLM's standard OpenAI-compatible server. The `--speculative-config` JSON
format was introduced in vLLM v0.6+. Check your vLLM version with `vllm --version` and ensure
it supports the `eagle3` method (added in vLLM v0.7+).

---

## 7. Integration with Our Existing Notebook

Our current notebooks (v14, v25) already use vLLM with the Harmony protocol. The integration
point is the `_start_server()` method in the `HarmonyTIRInferencer` class. No other changes
are needed — the speculative decoding is transparent to the client.

### Current _start_server (no speculative decoding):
```python
'--async-scheduling',
# END of current command
```

### Modified _start_server (with EAGLE-3):
```python
'--async-scheduling',
'--speculative-config',
'{"method": "eagle3", "model": "/kaggle/input/gpt-oss-120b-eagle3-aimo3", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1}'
```

The rest of the notebook (Harmony protocol, Jupyter kernels, voting, phase splitting) is
completely unchanged.

---

## 8. Compatibility with Our Novel Approaches

### Phase splitting (easy/hard routing)
**Fully compatible.** EAGLE-3 operates at the token generation level and is transparent to
higher-level routing. Whether a problem goes to the "easy" or "hard" phase, all generation
calls go through the same vLLM server which transparently uses speculative decoding.

### Domain routing
**Fully compatible.** Same reasoning — routing logic is above the LLM call layer.

### Reliability-weighted voting
**Fully compatible.** Output quality is mathematically identical; voting logic is unchanged.

### 12-attempt parallel inference (v26)
**Beneficial.** At concurrency=8 attempts in parallel, EAGLE-3 gives ~36-39% more tokens/s.
For 12 parallel attempts, expect similar gains as concurrency scales up within batch_size limit.

### Temperature settings
**Fully compatible.** Speculative decoding works for any temperature (including 0). The acceptance
criterion adjusts automatically. Higher temperature → slightly lower acceptance rate (draft must
match a more varied distribution), but still net positive throughput.

---

## 9. Does Anyone on AIMO3 Leaderboard Use EAGLE-3?

The GitHub repo `juemifuji/eagle3-aimo3` has only 3 stars and 0 forks (created 2026-02-03).
The `benchmark.py` in that repo IS structured as an AIMO3 solver, not just a benchmark —
it uses Harmony protocol, OpenAI client, system prompts targeting IMO. This suggests the author
is competing in AIMO3 and actively using it.

The model has 3,564 HuggingFace downloads/month, indicating broader community adoption.

No public Kaggle notebooks show the `--speculative-config eagle3` flag in their vLLM command.
The top public notebooks (seshurajup) do NOT use speculative decoding — this is an advantage
we can exploit by adopting it before others.

---

## 10. Implementation Steps (Priority Order)

### Step 0: Verify vLLM version (go/no-go gate)
The `eagle3` speculative method requires **vLLM 0.7+**. Before doing anything else, add this
as the first cell in your submission notebook:
```python
import subprocess
result = subprocess.run(['python', '-m', 'vllm', '--version'], capture_output=True, text=True)
print(result.stdout or result.stderr)
# If < 0.7, upgrade: pip install --upgrade vllm
```
If the kernel's vLLM is too old, none of the other steps will work. Abort and upgrade first.

### Step 1: Prepare the draft model as a Kaggle dataset
```bash
# On a machine with internet access (RunPod, local):
huggingface-cli download wenliang1990/gpt-oss-120b-eagle3-aimo3 \
  --local-dir /tmp/eagle3-draft

# Create Kaggle dataset:
kaggle datasets create -p /tmp/eagle3-draft \
  --name gpt-oss-120b-eagle3-aimo3 \
  --title "GPT-OSS-120B Eagle3 Draft Model for AIMO3"
```

### Step 2: Add dataset to notebook
In the Kaggle notebook editor, add the dataset as an input.
Path will be `/kaggle/input/gpt-oss-120b-eagle3-aimo3/`.

### Step 3: Modify _start_server() and preload in the notebook
In `CFG` class, add:
```python
draft_model_path = '/kaggle/input/gpt-oss-120b-eagle3-aimo3'
gpu_memory_utilization = 0.93  # lowered from 0.96 to fit draft model KV cache
```

In `_preload_model_weights()`, extend the preload loop to also warm the draft model:
```python
# After existing preload of main model, add:
for root, _, files in os.walk(self.cfg.draft_model_path):
    for file_name in files:
        files_to_load.append(os.path.join(root, file_name))
# The draft model is only ~0.6GB, adds negligible preload time
```

In `_start_server()`, add after `'--async-scheduling'`:
```python
'--speculative-config',
f'{{"method": "eagle3", "model": "{self.cfg.draft_model_path}", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 1}}'
```

### Step 4: Test locally or on RunPod A40
Note: A40 has 48GB VRAM. The 120B model in BF16 doesn't fit on 48GB. Test on RunPod A100 (80GB) 
or H100. Alternatively, test on Kaggle submission with a small timeout (abort after 2 problems)
to confirm server starts without OOM.

### Step 5: Monitor acceptance rate in vLLM logs
vLLM logs speculative token stats. Look for lines like:
```
Speculative tokens: avg_accepted=2.3/3, acceptance_rate=0.77
```
Acceptance rate > 0.7 means good speedup. On math tasks, expect 2.0-2.5 accepted/3 proposed.

### Step 6 (optional): Tune num_speculative_tokens
The benchmark uses 3, which is conservative. After confirming it works:
- Try `num_speculative_tokens=4` or `5` — more tokens verified per forward pass
- Higher values increase throughput IF acceptance rate stays high, but reduce it if the draft
  model diverges from the main model's distribution on longer speculative sequences
- Monitor acceptance rate: if it drops below 0.6 with 4 tokens, revert to 3
- Safe: lower acceptance rate only hurts throughput, never output quality

### Safety note on draft model trust
Since the acceptance/rejection criterion is enforced by the **main model**, a low-quality
draft model can only reduce throughput (low acceptance rate), never corrupt output quality.
This makes EAGLE-3 safe to adopt without fully auditing the community draft model's training.
Worst case: it offers no speedup. Best case: 40%+ throughput gain.

---

## 11. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| OOM on H100 80GB | Medium | Lower gpu_memory_utilization to 0.90, reduce max_model_len |
| Draft model path not found | Low | Pre-warm cache with preload script |
| vLLM version doesn't support eagle3 | Medium | Check vLLM version; update if needed |
| Kaggle dataset upload blocked | Low | Contact Kaggle support; try HF in internet mode |
| Lower acceptance rate on thinking traces | Low | Still net positive; thinking traces are verbose/predictable |
| Startup time increase | Very Low | Draft model is 0.3B; loads in seconds |

---

## 12. Expected Impact on Score

With 40% more tokens per problem:
- Problems that timeout can now complete verification steps
- Longer reasoning chains are possible within the per-problem budget
- More tool-call rounds per attempt
- More attempts fit in 4h55m

Our current best is 38/50 (v18), with validated notebooks at 44/50. The bottleneck is not
currently timeout-related failures (our timeout is 900s/problem). However:
- EAGLE-3 allows increasing attempts from 8→11 in the same time budget
- Or reducing per-attempt time limit and squeezing more diverse attempts
- At 44/50 level, each additional correct problem is from finding the right approach

Direct score gain estimate: +1 to +2 problems (from coverage of currently-timing-out edge cases
and more attempts). This is speculative — the main gain is insurance against timeout failures
at harder problem levels.

---

## Sources
- GitHub: https://github.com/juemifuji/eagle3-aimo3
- Draft model: https://huggingface.co/wenliang1990/gpt-oss-120b-eagle3-aimo3
- NVIDIA Eagle3: https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context
- EAGLE-3 paper: https://arxiv.org/html/2503.01840v1
- Baseten benchmark: https://www.baseten.co/blog/how-we-made-the-fastest-gpt-oss-on-nvidia-gpus-60-percent-faster/
- P-EAGLE in vLLM: https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/
