# A17 envelope one-pager — Qwen2.5-VL-72B-Instruct-AWQ on the 96 GB rail

Filed 2026-07-24 with the canary build. Discharges the A23 envelope-check
requirement (`stuck_review_v2_2026-07-23.md` §3/§5: "A17 starts under §3
envelope; ... envelope NO-GO is self-certifying"). All numbers cited from
artifacts on disk; nothing here conditions on an observed 72B measurement.

## Memory envelope (fits on paper; canary is the empirical check)

| item | value | source |
|---|---|---|
| card | NVIDIA RTX PRO 6000 Blackwell ×1, ~96 GB, sm_120 | `w0_eval_s1/…eval.log` CUDA check (build rail) = scored-rail SKU (`machine_shape: NvidiaRtxPro6000`) |
| 72B AWQ weights | **43.0 GB** (11 shards, 43,023,138,387 B, verified sum) | scope v1 §1; prereg amendment 2026-07-20 §1.1 |
| usable at `gpu_memory_utilization 0.92` | ~88 GB | `inference.json` (unchanged from 27B) |
| left for KV + vision tower + CUDA graphs | **~45 GB** | 88 − 43 |
| 27B comparison point | model 33.66 GiB + KV 43.62 GiB at max-model-len 65536 | `w0_eval_s1/vllm-openai-server.log` |

**Forced config change:** `--max-model-len 32768` (down from 65536). This
equals the harness's `ANALYZER_CONTEXT_WINDOW=32768`, so it costs nothing
behaviorally (scope v1 §0) while halving KV pre-allocation.

**KV arithmetic at 32768:** Qwen2.5-72B KV = 2(K,V) × 80 layers × 8 KV-heads ×
128 head-dim × 2 B ≈ **320 KiB/token** → ~10 GiB per full-32k sequence. The
~45 GB budget holds ~4.5 full-context sequences — and the canary runs exactly
**4 concurrent games**, with prefix caching shrinking the working set further.
Vision-tower weights are inside the 43 GB; its activations are transient and
small relative to the ~2 GB-class slack. **Verdict: feasible-but-tight, as
scoped (risk B).** Fallback ladder if load OOMs anyway: util 0.92 → 0.88
first; only then touch max-model-len; single-GPU impossibility is itself a
panel finding (no 2nd GPU exists on this rail).

## Serve-path envelope

- **Quantization kernel:** `--quantization awq_marlin` set EXPLICITLY (sm_120
  auto-detect wrinkle affects FP4, not AWQ, but we do not rely on
  auto-detect). vLLM pinned 0.19.0 by the wheelhouse; if awq_marlin errors for
  the VL arch on 0.19.0 that is a hard blocker → panel (risk C).
- **Parsers:** `--tool-call-parser hermes`, NO `--reasoning-parser`, NO
  thinking kwargs, `LOCAL_ANALYZER_ENABLE_THINKING=false` (risk D). Boot
  asserts in the canary exercise a forced tool-call round-trip AND a real
  image through the vision tower before any game starts; any miss dies loudly.

## Throughput envelope — the self-certifying NO-GO line (A23)

The 27B-FP8 baseline serves the full window at 192 tok/s job-wallclock with
480 actions/7920 s pooled over the 4 screen games (w0_eval_s1, frozen
numerator). The scoped decode penalty for 72B-AWQ is 2.5–3.0×. Per A23 /
stuck-review §3:

> **If the measured penalty exceeds 3.5× — i.e. the canary's pooled
> ρ_action = 480 / Σ N72B > 3.5 — the screen self-reports
> ENVELOPE-INFEASIBLE. That is a valid NO-GO datum by itself (physics), and
> unlike a capability NO-GO it requires NO panel ratification.**

The canary measures exactly this: `A17-CANARY rho_action_denominator=<Σ N72B>`
at the full 7920 s window, same SKU, 4-game concurrency (ρ_action ≤ 3.5 ⇔
Σ N72B ≥ 138). Per the R17 amendment §9.2 ρ_action is a planning diagnostic
for the GATE (null_adj evaluates at realized 72B actions), but it remains the
envelope self-certification statistic. Also void-on-sight conditions printed
by the canary itself: wrong GPU name (§0), game-window drift off ~7920 s
(risk A), MM cache stuck at zero (risk E).

## Cost envelope

One canary push ≈ 2.2 h window + ~0.25 h 72B load/init ≈ **2.5 GPU-h**, inside
the booked A17 AWQ-arm cap of 10.0 GPU-h (v2 §5) and the 30 GPU-h/wk quota
(week Jul 21–27 closes ≈ 23.2/30 with A17 booked). $0 cloud. Load/init sits
OUTSIDE the per-game 7920 s window (window clock = `max_runtime_s_per_game`,
starts at `bm.run()` after the server is already up), so it does not
contaminate the denominator.
