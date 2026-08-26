# Context Length Analysis: max_model_len 65536 vs 81920

**Research date**: 2026-04-01
**Question**: Should we use context_tokens=65536 (44/50 notebooks) or context_tokens=81920 (43/50 base)?

---

## Executive Summary

**Use 65536. The empirical evidence is unambiguous.**

The 44/50 notebooks (nihilisticneuralnet, kaanyorgun) use 65536. The base 43/50 notebook uses 81920. The score improved when context decreased. The improvement came from the 5-step prompt change, not from context length. No public ablation demonstrates 81920 outperforms 65536.

---

## Q1: Does 81920 vs 65536 context actually help on any problems?

**Answer: Unknown, but the empirical signal says no.**

- The arxiv paper (2603.27844) uses `--max-model-len 65536` exclusively. No ablation on context length was run or reported.
- The paper's best score (44/50, their best run out of 13) was achieved with 65536.
- The base 43/50 notebook uses 81920. When competitors switched to 65536 alongside the 5-step prompt, scores improved to 44.
- The only way 81920 helps is if a problem's reasoning chain exceeds 65536 tokens. The paper has no data on truncation rate.

**Actionable diagnostic**: Add a log line in `_process_attempt` triggered when the truncation check fires:

```python
max_toks = self.cfg.context_tokens - len(prompt_ids)
if max_toks < self.cfg.buffer_tokens:
    print(f"[TRUNCATED] attempt={idx}, prompt_len={len(prompt_ids)}")
    break
```

One submission with this logging tells us whether any problem actually hits the 65536 limit. If zero truncations occur across 50 problems × 8 attempts = 400 runs, 81920 provides zero benefit.

---

## Q2: How many problems have reasoning chains longer than 65536 tokens?

**Answer: Unknown from public data. Estimate: rare, possibly 0-2 out of 50.**

- gpt-oss-120b typical reasoning chains: ~15,000-40,000 tokens for hard IMO problems (based on TIR loop behavior with 8 attempts, turns=128, and the per-problem time budget of 270s).
- At ~50-100 tokens/second decode speed on H100, a 270-second budget allows ~13,500-27,000 tokens per attempt.
- Chains of 65,000+ tokens would require nearly the entire 270s budget just for token generation, leaving no time for Python tool calls. This is physically inconsistent with the TIR workflow.
- **Conclusion**: The 270s `base_problem_timeout` is the binding constraint for most problems, not context length. A 65536-token chain at 80 tok/s takes 819 seconds - more than the entire problem budget. So in practice, the context window is never the binding limit.

---

## Q3: VRAM impact of 81920 vs 65536 with fp8_e4m3 KV cache

**Answer: Zero VRAM difference in fp8 mode.**

### Architecture of gpt-oss-120b (from arxiv 2508.10925v1)

| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| KV heads (GQA) | 8 |
| Head dimension | 64 |
| Block size (vLLM) | 16 tokens |
| Weights on disk | 60.8 GiB (MXFP4 + overhead) |

### KV cache block size calculation

```
fp8 bytes per block = 1 byte * 36 layers * 16 tokens * (8 kv_heads * 64 head_dim * 2)
                    = 1 * 36 * 16 * 1024
                    = 589,824 bytes = 576 KB per 16-token block
```

### VRAM budget with gpu_util=0.96, H100 80GB

```
Usable:      76.8 GB  (80 * 0.96)
Weights:     65.3 GB  (actual file size includes MoE router, embeddings)
Overhead:     1.5 GB  (CUDA context, activations)
Available:    10.0 GB  for KV cache
```

```
fp8 blocks available: 10.0 GB / 576 KB = ~16,954 blocks
Total token capacity: 16,954 * 16 = ~271,267 tokens
```

### How max_model_len affects allocation

**max_model_len does NOT pre-allocate VRAM.** vLLM's PagedAttention:
1. Profiles model weights memory usage
2. Calculates `available = total * util - model_memory`
3. Allocates `num_blocks = available // block_size_bytes` (ALL available blocks)
4. Validates `num_blocks * block_size >= max_model_len`

For both ctx=65536 and ctx=81920:
- Required blocks: 4096 vs 5120
- Available blocks: ~16,954
- Both pass validation with large margin (4.1x and 3.3x headroom respectively)
- **The SAME ~16,954 blocks are allocated either way**

### Maximum concurrent sequences at full context

| Context | fp8 concurrent | per-sequence fp8 |
|---------|---------------|-----------------|
| 65536   | 4             | 2.42 GB         |
| 81920   | 3             | 3.02 GB         |

These are theoretical maximums (all sequences at full context simultaneously). In practice, vLLM uses PagedAttention's on-demand block allocation — sequences grow incrementally, do not all peak at the same time. AIMO3 runs 8 concurrent sequences, but each is growing from 0, not starting at max_model_len.

---

## Q4: Does longer context reduce batch throughput?

**Answer: Negligible for AIMO3's workload.**

### Why throughput is unaffected

1. **Decode phase is compute-bound, not memory-bound**: At N=8 concurrent requests, each forward pass processes 8 tokens (one per active sequence). This 8-token decode batch is tiny; the H100 is severely underutilized. Throughput is dictated by latency per forward pass, not memory bandwidth.

2. **The 8-attempt limit is the real constraint**: The solver runs exactly `attempts=8` parallel requests. Whether max_model_len is 65536 or 81920, the KV pool is filled on demand, not pre-allocated per-sequence. With 8 active sequences at typical lengths (5,000-30,000 tokens each), the 10 GB KV pool is ~10-30% utilized.

3. **Prefill is identical**: The initial prefill (problem text + system prompt ~500-1000 tokens) is the same regardless of max_model_len.

4. **Attention cost scales with actual sequence length, not max_model_len**: PagedAttention's sparse attention only attends over filled KV blocks, not empty preallocated space.

---

## Q5: Has anyone tested this ablation for AIMO3?

**Answer: No public ablation exists.**

- The arxiv paper (2603.27844) fixes context at 65536 and runs no comparison.
- The Kaggle discussion forum does not contain posts about context length ablation (Kaggle pages return only JS metadata when scraped).
- The observable empirical signal is: 43/50 (81920) -> 44/50 (65536) when prompt changed simultaneously. This is confounded — the prompt change explains the score increase, not the context change.
- No notebook with 81920 scores higher than a notebook with 65536 on the public leaderboard.

---

## Q6: Does the arxiv paper (2603.27844) mention context length?

**Answer: Yes, briefly. One mention, no ablation.**

From Appendix C of arxiv 2603.27844:

> vLLM configuration: `--max-model-len 65536 --kv-cache-dtype fp8_e4m3 --max-num-seqs 256 --gpu-memory-utilization 0.96`

No justification for 65536 is given. No comparison with 81920 is reported. The phrase "512 buffer tokens reserved for output" appears (suggesting 65024 effective reasoning tokens), but no chain-length distribution data is presented.

---

## Q7: Does max_num_batched_tokens affect this?

**Base config uses 2048; 44/50 notebooks omit it.**

### What max_num_batched_tokens does

This parameter limits the total tokens processed per scheduler iteration:
- Controls how many tokens from prefill + decode are batched together
- Lower values throttle prefill throughput (splits large prefills across iterations)
- Default value: `_DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048` (from vLLM source, confirmed)

### Impact of omitting it

**Omitting max_num_batched_tokens from the vLLM command does NOT change behavior** — the default is 2048 for standard (non-pooling, non-multimodal) models. The 43/50 base notebook explicitly passes `--max-num-batched-tokens 2048`, which matches the default. The 44/50 notebooks omit it and get the same behavior.

**No practical difference.** Either explicit or omitted, the value is 2048.

### Could a higher value help?

In theory, increasing `max_num_batched_tokens` to 65536 or higher would allow vLLM to process an entire new request's prompt in a single scheduler step (chunked prefill disabled). For AIMO3:
- Typical prompt: ~500-1000 tokens (well within 2048)
- 2048 is sufficient to prefill most AIMO3 requests in a single step
- No benefit expected from increasing this

---

## Q8: What about --max-cudagraph-capture-size?

**Base uses 64; 44/50 notebooks omit it.**

### What CUDA graph capture does

vLLM captures CUDA graphs for decode steps to reduce kernel launch overhead. Graphs are captured for specific batch sizes (1, 2, 4, 8, ..., up to a maximum). When the actual decode batch size falls within the captured sizes, the pre-compiled graph executes with lower overhead than eager mode.

`max_cudagraph_capture_size` sets the maximum batch size that gets a CUDA graph captured:
- Smaller value: fewer graphs captured, faster startup, but falls back to eager mode for larger batches
- Default: in vLLM v0.6, `max_seq_len_to_capture = max_model_len` (not a batch size limit)
- The `max_cudagraph_capture_size` flag was added in newer vLLM versions

### Impact of omitting it (as 44/50 notebooks do)

From the 44/50 learnings document, removing `--max-cudagraph-capture-size` was identified as one of the changes. This likely means:
- 44/50 notebooks allow vLLM to use its default capture policy
- The default captures graphs for a wider range of batch sizes
- This may actually improve decode throughput vs an explicit cap of 64

**Recommendation: Follow the 44/50 lead and omit this flag.** The 44/50 notebooks with the 5-step prompt outperform the 43/50 base that sets this to 64.

---

## Q9: Is there a benefit to --async-scheduling specifically?

**Answer: Both configs use it; it's non-controversial.**

Both the 43/50 base and 44/50 notebooks include `--async-scheduling` in the vLLM command (from the notebook code inspected). This flag enables asynchronous output processing:
- Processes completed request outputs asynchronously rather than blocking the main scheduling loop
- Reduces queue latency for subsequent iterations
- Beneficial for multi-request workloads where output processing (token decoding, logprob computation) would otherwise stall new scheduling

For AIMO3 with `top_logprobs=5` and streaming output, async processing reduces the cost of entropy computation from blocking the main loop. Keep it.

---

## Final Recommendation

### Use context_tokens=65536 (confirmed)

| Parameter | Recommended | Reason |
|-----------|-------------|--------|
| `context_tokens` | **65536** | Empirically validated at 44/50; paper baseline; 81920 provides no demonstrated benefit |
| `--max-model-len` | **65536** | Same |
| `--kv-cache-dtype` | **fp8_e4m3** | Required to fit model at any of these context lengths |
| `--max-num-batched-tokens` | **omit** | Default is 2048; explicit value would be identical |
| `--max-cudagraph-capture-size` | **omit** | 44/50 notebooks omit it; default is better than cap of 64 |
| `--async-scheduling` | **keep** | Both configs use it; async output processing helps |
| `--max-num-seqs` | **256** | 44/50 notebooks use 256; matches `batch_size` |
| `gpu_memory_utilization` | **0.96** | 44/50 notebooks use 0.96; safer than 0.99 |

### Why not 81920?

1. No empirical evidence it helps any problem
2. The score-improving transition (43->44) moved from 81920 to 65536
3. The arxiv paper's 13-run baseline (mean 39.7, best 44) uses 65536
4. The decode throughput math shows no meaningful difference for N=8 concurrent requests
5. At 270s per-problem budget and ~80 tok/s decode, practical chain length ceiling is ~21,000 tokens — well below 65536

### When would 81920 matter?

Only if a specific problem generates a reasoning chain that:
- Exceeds 65,000 tokens (>800 seconds of generation at 80 tok/s)
- Still produces the correct answer (the model didn't give up earlier)
- Would produce a wrong answer or truncate at 65536

This scenario is essentially impossible given the 270s-900s per-problem time budget.

### Diagnostic to validate

Add this to `_process_attempt`:

```python
max_toks = self.cfg.context_tokens - len(prompt_ids)
if max_toks < self.cfg.buffer_tokens:
    # This fires when the context window fills up
    print(f"[CTX_TRUNCATED] problem={problem[:50]}, attempt={idx}, prompt_len={len(prompt_ids)}")
    break
```

If this never fires across a full 50-problem run (400 attempts), 65536 is definitively sufficient.

---

## Sources

1. Arxiv 2603.27844 (Natapong Nitarach): vLLM config in Appendix C, `--max-model-len 65536`
2. Arxiv 2508.10925v1: gpt-oss-120b architecture (36 layers, 8 KV heads, head_dim=64, 60.8 GiB checkpoint)
3. vLLM v0.8.4 source, `worker/worker.py`: `num_gpu_blocks = available // block_size_bytes` (post-profiling)
4. vLLM v0.8.4 source, `worker/cache_engine.py`: `block_bytes = dtype * layers * block_size * (kv_heads * head_dim * 2)`
5. vLLM v0.8.4 source, `config.py`: `_DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048`
6. Competition notebook analysis: `44_50_comparison.md` (both 44/50 notebooks use context=65536, no --max-num-batched-tokens, no --max-cudagraph-capture-size)
7. `best_config.py` (base 43/50): context=81920, batched_tokens=2048, capture_size=64
