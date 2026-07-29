# Research Sweep — 2026-07-29 (ARC-AGI-3 campaign, step 1c)

Repo: F:\kaggle\arc-prize-2026 | Zero cloud budget, Kaggle-kernel-only builds.
Context frame: A17 72B bench measured ΣN=5 executed actions in a 7920s window — model
generated tokens continuously but actions stopped landing after ~720s (throughput/stall,
not a reasoning failure). Seed-2 confirmation today. Sweep prioritized (a) why long-run
vLLM+agent loops stall, (b) 72B AWQ multimodal throughput, (c) VLM exploration efficiency,
plus tool-call reliability and banking/replay.

Dedup: checked against prior briefs; excluded 2607.08716, 2607.13591, 2607.09493,
2607.20972, 2607.07196, 2607.08964, 2607.03441, 2607.15439, 2607.18754, 2607.05775,
2607.12227, 2607.08124, 2607.08233, and the ARC Prize Opus-5 30.2% blog item.

---

## 1. Robust KV Cache Management under Output-Length Uncertainty — arXiv:2607.16892 (Jul 18)
**VERDICT: ADOPT (diagnostic lens — this is very likely our A17 stall root cause).**
DRO-based KV reservation: output length is unknown at request arrival, so under-reservation
triggers **preemption → request termination + recomputation**, while over-reservation
shrinks the batch and collapses throughput. Their production traces (BurstGPT/Azure/ShareGPT)
show reservation policy alone drives up to 56% cost swings and P99/goodput regimes.
Why it matters to A17: an agent loop with a monotonically growing context (each landed action
appends observation + reasoning) is exactly the "long/uncertain output length" regime. After
~720s the running request's KV footprint likely crossed the scheduler's watermark, triggering
preempt/recompute cycles — the model keeps *decoding tokens* (looks alive) but *no new action
lands* because the request is thrashing on recompute. That matches ΣN=5 with continuous token
gen. One-line action: instrument the A17 vLLM server for `num_preemptions` / `swapped` /
`recompute` counters over the 7920s window before touching the model — the stall is almost
certainly scheduler preemption, not model quality. Free to check on the Kaggle build.

## 2. AgentTether: Graph-Guided Diagnosis + Runtime Repair — arXiv:2607.06273 (Jul 7)
**VERDICT: ADAPT (banking/replay design; do NOT adopt the framework wholesale).**
Abstracts runs into Transition Units on a dependency-aware Critical Transition Graph; a
**cross-iteration Repair Memory** supplies behavior-scoped corrective guidance on re-execution
without touching the agent weights. Repaired 59% (Banking) / 65% (GPT-5.4) of initially-failed
tau-bench tasks *while reducing turns and tokens*. Relevance: this is the cleanest recent
formalization of "banking" = persist localized failure→fix units across episodes and replay
guidance, not full trajectories. For our episodic-agent banking arm, the takeaway is store
repair units keyed on the *transition that failed* (state-aliasing candidate) rather than the
whole rollout. ADAPT the Repair-Memory keying idea; ignore the heavyweight graph tooling under
zero-budget constraints.

## 3. Cost-Effective Agent Harnesses for ARC (Explorer-Definer) — arXiv:2607.06764 (Jul 7)
**VERDICT: ADOPT (direct harness lesson for our zero-budget frozen fork).**
Open-weight DeepSeek V3.2, **no ARC fine-tuning**, strict budget. Explorer-Definer Pipeline
(separate pattern-discovery from program-synthesis) hit 57.5% pass@2 at $0.25/task; Reflective
Orchestrator 67.25% at $0.62/task — ~52 pts over a 15.5% one-shot baseline. Key empirical
claim: **"generation-bound, not selection-bound"** — gains came from broader candidate
generation and adaptive re-exploration when hypotheses fail, not better ranking. This is
ARC-AGI-1 (offline grids) not AGI-3 (interactive), so it does not transfer directly, BUT the
architectural lesson maps onto our loop: when actions stall/fail, spend budget re-exploring new
transformation hypotheses rather than re-ranking existing ones. Reinforces the uncapped-BFS
memory note. ADOPT the "widen generation before you improve selection" prior for the harness.

## 4. PoTRE: Test-Time Reasoning via Cognitive Heterogeneity — arXiv:2607.20268 (Jul)
**VERDICT: IGNORE (for now — wrong cost regime).**
Decouples inference into four heterogeneous agents + a Task-Adaptive Aggregation Layer;
evaluated on ARC-AGI-2 among other frontier benchmarks. The multi-agent aggregation is a
throughput multiplier — the opposite of what a 72B single-GPU, already-stalling, single-request
harness can afford right now. Revisit only after the A17 throughput/stall issue (finding 1) is
resolved. IGNORE until serving is stable.

## 5. HyMCache: KV Cache for Multi-Turn Serving w/ CXL-Hybrid Memory — arXiv:2607.18141 (Jul 18)
**VERDICT: IGNORE (hardware-dependent; not on Kaggle).**
Tiers KV cache across CXL-hybrid memory to sustain multi-turn (agentic) serving as context
grows. Correct diagnosis (multi-turn agents exhaust GPU KV), wrong remedy for us — requires CXL
memory tiering we do not have on Kaggle's single-GPU kernel. Useful only as corroboration that
"growing multi-turn context exhausts KV and degrades serving" is the field-consensus stall
mechanism, reinforcing finding 1. IGNORE the method.

## 6. GLANCE — Visual-Linguistic Curiosity for VLM Exploration — arXiv:2605.03782 (May; surfaced now)
**VERDICT: ADAPT (exploration prior; flagged as slightly stale — May, not this week).**
Uses the discrepancy between the agent's *linguistic* world-model prediction and the *visual*
reality (via an evolving target network) as an intrinsic curiosity signal for sparse-reward VLM
tasks. Directly on-topic for AGI-3's "explore, infer goals, no instructions" regime. Not new
(May submission) so reported only as an exploration-design reference, not a fresh result. The
adaptable kernel: score candidate actions by predicted-vs-observed frame mismatch to drive the
VLM toward "known unknowns" — cheap to approximate with our existing frame-diff signal without
the RL training loop. ADAPT the mismatch-as-novelty heuristic; skip the RL machinery.

---

### Net
Finding 1 is the headline: the A17 stall is textbook KV-cache preemption/recompute thrash under
growing agent context, and it is *measurable for free* on the next Kaggle build via vLLM
preemption counters — do that before any model-quality hypothesis. Findings 2–3 give concrete,
budget-compatible harness upgrades (transition-scoped repair banking; widen generation before
improving selection). 4–6 logged but deprioritized (cost regime / hardware / staleness).
