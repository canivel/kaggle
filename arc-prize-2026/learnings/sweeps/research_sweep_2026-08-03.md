# ARC-AGI-3 Research Sweep — 2026-08-03

Window: ~2026-07-30 .. 2026-08-03 (items new since the 08-02 sweep).
**arXiv indexing status: the 2026-08 (`2608.*`) listing is EMPTY — "No updates for this time
period." Newest indexed submission date across cs.AI is 2026-07-31.** So the genuinely in-window
arXiv slice is 07-29..07-31 only; nothing from Aug 1-3 exists yet to sweep.

Fields swept: ARC-AGI-3 / ARC Prize results+writeups+interviews; LLM agents on interactive
benchmarks (test-time learning, exploration, world models); **agent context management /
compaction / memory eviction for long-horizon agents (A22 lane)**; agentic harness design and
banking/replay strategies for stochastic evals. Sources: arXiv API (submittedDate-sorted, 4
distinct query families), arXiv cs.AI monthly listings (2026-07, 2026-08), arcprize.org/blog,
web search.

Campaign anchors for verdicts: **A22 compaction is the single active lane** (retained-reasoning
dead-key detection + rolling-cut eviction inside a constrained rerun); frozen filler runs daily;
boristown A/B DECLINED (NC-14 mechanism-null), prereg dormant; control ledger n=19 STATIONARY
(best 1.33); 72B route DEAD; **no cloud spend** — evals are free Kaggle kernel builds on the RTX
PRO 6000; Kaggle 9h wall, no-final-rerun; private track = **no code-exec tools, single model**.

Already-triaged, NOT re-reported: Living-Harness 2607.26598; MemoHarness 2607.14159; Agentic
Context Management 2607.21503; Tycho 2607.28287; Plans-Don't-Persist 2606.22953; MemHarness
2607.28272; Self-GC 2607.00692; Addressable Recall Compaction 2607.25066; CompactionRL 2607.05378;
OpenAI 07-29/07-30 retained-reasoning + compaction (13.3→38.3); Opus 5 30.2%; aTTT 2607.03441;
TTA-via-Env 2511.04847; GSME/HarnessBank 2607.13683; EvoAgentBench 2607.05202; Rodionov
exec-sims 2605.05138; Rudakov 2512.24156; vLLM #39056.

---

## A. NEW IN-WINDOW (arXiv 2026-07-29..07-31)

### 1. Reproducing LightMem: Naive RAG Is Just as Good for Memory Management  ⭐ MOST DECISION-RELEVANT
- arXiv:2607.29104 (v1 **2026-07-31**) — https://arxiv.org/abs/2607.29104
- Gist: An independent reproduction of LightMem (compress past interactions into compact
  retrievable entries). Findings: **constructed memories do NOT consistently outperform raw-turn
  retrieval**; naive RAG generally wins at matched retrieval depth; **LightMem wins mainly under
  tight answering-token budgets**; oracle eval shows memory *construction removes answer-relevant
  information*; and swapping only the retriever over a fixed store moves accuracy **58.1% → 75.5%**
  (17.4 pts) — i.e. the retriever, not the memory representation, is the dominant lever.
- **Verdict: ADAPT (prereg framing + a falsifiable prediction for A22).** This is the cleanest
  external prior on *why* a compaction arm would win or lose, and it cuts both ways for us:
  (a) **supportive** — our regime IS the tight-budget regime (9h wall + finite context + no
  final rerun), which is exactly where constructed/compacted memory beat raw retention; (b)
  **cautionary** — the mechanism of the win is *budget relief*, not information gain, and
  construction actively *deletes* answer-relevant content. Two prereg deltas: (i) attribute the
  A22 hypothesis explicitly to **budget relief under the wall**, not to "better memory," so a null
  is interpretable; (ii) **prefer deterministic retention of raw turns over LLM-constructed
  summaries** wherever both fit the budget — this converges with CWL (§B1) and MemDecay (§B2) and
  argues against any summarize-then-reinject variant of the retained-reasoning arm.

### 2. Filesystem-Based Memory for LLM Agents: Organization, Evolution, and Sustainability
- arXiv:2607.26637 (v1 **2026-07-29**) — https://arxiv.org/abs/2607.26637
- Gist: Systematic study of the default "directory tree of markdown files" agent-memory pattern.
  Result is a negative one: **organization erodes for all but the strongest management agent, and
  no measured agent converts organization itself into better answers.** The only durable benefit
  is efficiency — organized stores roughly halve retrieval cost when material is large.
- **Verdict: IGNORE for build (ADAPT as a scope guard).** Confirms we should NOT spend A22 effort
  on an elaborate structured memory store maintained by the model — the maintenance burden is
  real and the answer-quality payoff is not. Reinforces keeping A22 to a **single-surface,
  deterministic** context/memory change. Tempers the Living-Harness (2607.26598) enthusiasm from
  yesterday's sweep: keep the state-graph framing conceptual, do not build a model-maintained store.

### 3. Zero-Mem: Zero-Token Memory Operations for LLM Agents
- arXiv:2607.29377 (v1 **2026-07-31**) — https://arxiv.org/abs/2607.29377
- Gist: Inference-only memory system where **no step outside final question answering invokes an
  LLM or consumes tokens**. Organizes traces into an entity-context graph + a temporal hierarchy,
  retrieves from both, applies deterministic calibration to discard conflicting evidence. −57.6%
  memory-operation time vs baselines on long-memory / long-context QA.
- **Verdict: ADAPT (design constraint, not a drop-in).** The load-bearing idea for A22 is the
  hard constraint itself: **zero LLM calls in the memory/eviction path.** Under a 9h wall with no
  final rerun, every compaction LLM call is stolen from gameplay actions — so the A22 eviction
  policy should be provably token-free (deterministic scoring), which is exactly the CWL/MemDecay
  shape. Evaluated on QA, not interactive ARC, and the entity-graph construction is more machinery
  than we want — take the constraint, not the architecture.

### 4. Aries: Rethinking AI Cloud Infrastructure for Agentic Serving Systems
- arXiv:2607.29069 (v1 **2026-07-31**) — https://arxiv.org/abs/2607.29069
- Gist: Full-stack agentic-serving experimentation framework. Key qualitative finding: **retaining
  additional context yields diminishing accuracy benefits while reducing serving capacity**; also
  notes tool sandboxes alternate long idle periods with sudden spikes. No hard numbers in abstract.
- **Verdict: IGNORE (weak corroboration only).** Directionally supports the A22 premise (there is
  a knee in the retain-more curve, so cutting is cheap in accuracy and expensive to skip), but the
  paper is a cloud-serving/multi-tenant capacity study — our binding constraint is a single 9h
  Kaggle kernel, not serving throughput. No build, no cloud spend implication.

### 5. ChronoMem: Version Control and Semantic Rollback for LLM Agent Memory
- arXiv:2607.27773 (v1 **2026-07-30**) — https://arxiv.org/abs/2607.27773
- Gist: Semantic version-control layer over agent memory (Google ADK integration). Whole-memory
  snapshot at each write, natural-language undo mapped to a historical version via hybrid lexical+
  semantic retrieval, rank fusion and reranking; agent can then answer counterfactually "as if
  future updates never occurred."
- **Verdict: IGNORE.** Snapshot-per-write is the opposite of our cost profile (we are cutting
  context to survive the wall, not versioning it), the NL-undo path requires extra model calls,
  and the use case (user-driven rollback in conversational agents) has no analog in a no-human,
  no-rerun ARC kernel. Filed only so the compaction shelf is complete.

### 6. TraceViT: Grounded Trace Supervision for Visual Abstract Reasoning
- arXiv:2607.29586 (v1 **2026-07-31**) — https://arxiv.org/abs/2607.29586
- Gist: Looped visual reasoner **trained** on semantically monotonic transformation chains derived
  from programmatic task implementations; decomposes solutions into intermediate grid states.
  **67.8% pass@2 on ARC-AGI-1, 24.3% pass@2 on ARC-AGI-2. No ARC-AGI-3 results.** Ablations: trace
  supervision helps only when paired with grounding.
- **Verdict: IGNORE.** Only ARC-AGI-*3*-adjacent by name. It is a static-grid (AGI-1/2) method,
  it requires training (violates the zero-cloud-spend rule), and it does not touch the interactive
  exploration/agentic setting our track scores. Logged so the ARC-side coverage is on record.

### 7. DungeonBench: Rules-Rich Tactical Reasoning in D&D Combat
- arXiv:2607.29577 (v1 **2026-07-31**) — https://arxiv.org/abs/2607.29577
- Gist: Simulator benchmark with an *Encounter* track (single fight) and a *Day* track (multiple
  encounters linked by persistent resources — HP, spell slots, consumables). Headline: **frontier
  LLMs often win isolated encounters but fail at cross-encounter resource budgeting, rest timing,
  and rule-aware discipline.**
- **Verdict: IGNORE (one useful analogy).** Not adoptable, but it is a second independent
  demonstration of the failure mode our whole campaign lives inside: competence *within* an episode
  does not carry across episodes when the binding resource is a budget. That is structurally the
  ARC-AGI-3 level-to-level carryover problem, and it is a rhetorical support for A22 (budget
  discipline is the scarce skill), not new evidence.

### 8. ARC Prize / ARC-AGI-3 ecosystem — nothing new
- **arcprize.org/blog: no new post.** Latest is still 07.06.26 "ARC Prize 2026: ARC-AGI-3
  Milestone Prize #1" (previous: 05.01.26 GPT-5.5/Opus-4.7 analysis). **Second consecutive sweep
  with no ARC Prize publication — absence confirmed.**
- Public snapshot (data verified 07-31, third-party aggregator): **Claude Opus 5 30.2%**, GPT-5.6
  Sol 7.8%, Claude Opus 4.8 1.5%. Unchanged vs our record — no re-triage needed.
- TechTimes, 07-31: "ARC-AGI-3 Gets Open-Source Agent That Writes Python World Models Instead of
  Neural Weights" — **press coverage of the already-triaged Rodionov exec-sims line** (2605.05138 /
  15-of-25 public games with GPT-5.5), no new technical content. **IGNORE** — and note it remains
  private-track-illegal for us regardless (no code-exec tools).
- No new Tufa Labs duck-harness writeup or MLST episode in-window (the Zurich MLST episode and the
  duck-harness blog both predate this window).

---

## B. NEW-TO-US BUT OLDER (surfaced this sweep; all directly on the A22 lane)

These are not in the 3-4 day window, but none appears anywhere in `learnings/` and all three
speak straight to the A22 mechanism. Reporting them because the in-window arXiv slice is thin.

### B1. Beyond Compaction: Structured Context Eviction for Long-Horizon Agents (CWL)  ⭐⭐ STRONGEST A22 MATCH
- arXiv:2606.11213 (v1 2026; Semenov & Dorofeev) — https://arxiv.org/abs/2606.11213
- Gist: **Context Window Lifecycle (CWL)** — the agent annotates its trajectory as *typed,
  dependency-linked episodes*; a **deterministic, LLM-free policy** evicts in priority order when
  the token budget is exceeded. It preserves user turns and *active* reasoning context while
  **aggressively removing action episodes whose effects are already persisted in the environment.**
  Explicitly positioned against summarization-based compaction on four counts: unpredictable
  lossiness, destruction of causal structure, blocking model cost, compression-induced
  hallucination. Demo: one agent completing 89 sequential tasks across 80M tokens without
  accuracy loss.
- **Verdict: ADAPT — this is the reference design for A22's eviction half.** The key insight
  transfers to ARC-AGI-3 almost verbatim and is stronger here than in the paper's own setting:
  **in a game environment the effects of past actions ARE persisted in the observable frame**, so
  past action episodes are the provably-safest thing to cut, while *active* reasoning is the thing
  to keep. That is precisely the A22 split (retained-reasoning kept, rolling-cut applied to stale
  action turns), now with an external, published justification and a priority ordering to copy.
  It is deterministic and LLM-free → zero token cost inside the 9h wall, no code-exec, single
  model → fully private-track- and budget-legal.

### B2. MemDecay: Region-Aware KV Cache Eviction for Efficient LLM Agent Inference  ⭐
- arXiv:2607.10582 (2026-07-12) — https://arxiv.org/abs/2607.10582
- Gist: **Training-free** region-aware eviction. Assigns region-specific base priorities and decay
  rates, refreshes retention scores when tokens receive attention, evicts lowest-scoring pages,
  and **pins** critical regions. Measured token half-lives are the headline: **system instructions
  stay useful for 148-189 decoding steps; scratchpad tokens decay in 14-16 steps** (~10x gap).
  Pinning critical regions preserved full accuracy in all configs, and **region-aware retention
  beat recency-based retention as context expanded** (Qwen 1.5B/3B).
- **Verdict: ADAPT — most actionable single item in this sweep (see §C).** Two things land: (i)
  the 10x half-life gap between instruction-region and scratchpad-region tokens is *direct
  empirical support for dead-key detection by region* — you do not need attention introspection to
  know reasoning/scratchpad content goes stale ~10x faster than instructions; (ii) **"region-aware
  beats recency as context expands" is a claim about exactly the policy the duck harness already
  runs.** The duck's stock behaviour is "infinite play via eviction" — pop the oldest messages,
  keep system prompt + recent history — i.e. **pure recency eviction**. MemDecay says that is the
  weaker policy in the long-context regime the duck lives in. Caveat: MemDecay operates at the
  KV-page level on small Qwen models, whereas A22 operates at the message level, so the *result*
  is analogical, not transferable — but the *policy shape* (region priorities + decay + pinning)
  ports cleanly to message-level rolling-cut with zero token cost.

### B3. Self-Compacting Language Model Agents (SelfCompact)
- arXiv:2606.23525 (v1 2026-06-22, v2 2026-07-10) — https://arxiv.org/abs/2606.23525
- Gist: Inference-time, **no fine-tuning or external supervision**. Two components: a model-invoked
  compaction *tool*, plus a lightweight **rubric for when to fire it** — compact at task resolution
  or trajectory convergence; **suppress mid-derivation or when the agent is stuck.** Up to +18.1
  pts on math, +5-9 on agentic search, and **30-70% lower per-question cost vs fixed-interval
  summarization.**
- **Verdict: ADAPT (the trigger policy, not the summarizer).** A22 currently frames rolling-cut as
  a budget-triggered mechanism; SelfCompact's evidence is that **when you cut matters as much as
  what you cut**, and specifically that cutting mid-derivation or while stuck is harmful. In
  ARC-AGI-3 "stuck" is a common and detectable state (no score change over N actions), so the
  suppress-while-stuck rule is cheap to implement deterministically and is a plausible confound if
  we *don't* control for it. Do NOT adopt the model-invoked summarization tool itself — that
  spends tokens and conflicts with §A1's raw-over-constructed finding and §A3's zero-token
  constraint.

---

## Summary

- **Relevant items this sweep: 11** — 8 in-window (7 arXiv 07-29..07-31 + the ARC Prize/ecosystem
  null) and 3 new-to-us older papers on the A22 lane.
- **Explicit absences (each is a finding):**
  - **arXiv `2608.*` is completely unindexed** (2026-08 listing returns "No updates for this time
    period"); newest indexed cs.AI submission is 2026-07-31. Nothing from Aug 1-3 exists to sweep.
  - **ARC Prize blog: no new post for the second consecutive sweep** (latest still 07-06 Milestone
    Prize #1). No new ARC-AGI-3 results, writeups, or interviews in-window; no new Tufa Labs or
    MLST material. Public snapshot unchanged (Opus 5 30.2%).
  - **Nothing new on banking/replay strategies for stochastic evals.** Searches returned only
    already-triaged or pre-window material (pass^k reliability, HarnessBank/GSME 2607.13683,
    ReliabilityBench). Our shelved EWM+banking lane gets no new evidence in either direction.
- **Non-IGNORE verdicts: 5 ADAPT, 0 ADOPT** (nothing is drop-in adoptable under no-cloud-spend +
  no-code-exec + single-model + 9h wall).
  1. **CWL / Beyond Compaction (2606.11213)** — reference design for A22 eviction: deterministic,
     LLM-free, typed episodes, evict actions whose effects are already persisted in the env.
  2. **MemDecay (2607.10582)** — region priorities + decay + pinning beat recency; 148-189 vs
     14-16 step half-lives justify region-based dead-key detection.
  3. **LightMem reproduction (2607.29104)** — the win comes from *budget relief*, not better
     memory; prefer raw-turn retention over constructed summaries.
  4. **SelfCompact (2606.23525)** — adopt the *trigger rubric* (suppress mid-derivation / while
     stuck), not the summarizer tool.
  5. **Zero-Mem (2607.29377)** — adopt the hard constraint: zero LLM calls in the eviction path.

## Effect on today's A22 work

One substantive change, three prereg deltas, no change to the frozen filler daily and no new spend.

- **CHANGE — reframe the A22 eviction arm from recency to region.** The duck harness baseline
  already does rolling-cut *by recency* ("infinite play via eviction": pop oldest, keep system
  prompt + recent window). A pure recency rolling-cut is therefore **not a clean intervention — it
  is the control.** MemDecay (§B2) plus CWL (§B1) converge on the alternative: cut by **region /
  episode type with pinning**, not by age. Concretely for A22: pin the system prompt and the
  *active* reasoning span; make **stale action episodes whose effects are already visible in the
  current frame** the first eviction class (they are recoverable from the observation, so the cut
  is information-preserving); give reasoning/scratchpad content a fast decay and instructions a
  slow one. This sharpens the arm from "cut more/less" into a falsifiable **region-aware vs
  recency** A/B against a baseline the harness already ships — which is a much better-powered
  comparison than an unanchored compaction knob.
- **Prereg delta 1 (attribution):** state the hypothesis as *budget relief under the 9h wall*, not
  "better memory" (§A1) — otherwise a null is uninterpretable.
- **Prereg delta 2 (mechanism purity):** the eviction path must make **zero LLM calls** (§A3, §B1);
  drop any summarize-then-reinject variant — it costs actions, and construction provably deletes
  answer-relevant content (§A1 oracle result).
- **Prereg delta 3 (confound control):** add a deterministic **suppress-cut-while-stuck** rule
  (no score change over N actions) per §B3, and log when it fires, so cut-timing is not a hidden
  confound in the arm.
- **Scope guard:** do NOT build a model-maintained structured memory store (§A2) — organization
  erodes and buys no accuracy. This narrows, but does not retract, yesterday's Living-Harness
  (2607.26598) framing: keep the failure→recovery / state-graph idea as *framing*, implement only
  the deterministic single-surface context/memory change.
- **Unchanged:** frozen filler runs daily; boristown prereg stays dormant (nothing this window
  bears on NC-14); ledger n=19 stationary; no cloud spend; evals stay free Kaggle kernel builds.
