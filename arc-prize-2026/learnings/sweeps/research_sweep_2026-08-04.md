# ARC-AGI-3 Research Sweep — 2026-08-04

Window: ~2026-08-01 .. 2026-08-04 (items new since the 08-03 sweep).

> **METHODOLOGY CORRECTION — yesterday's arXiv "empty month" was a listing-page artifact.**
> The 08-03 sweep concluded that "arXiv `2608.*` is completely unindexed … nothing from Aug 1-3
> exists to sweep." That conclusion came from `arxiv.org/list/cs.AI/2026-08`, which **still today
> returns "No updates for this time period."** But `export.arxiv.org/api/query` sorted by
> `submittedDate` **does** return `2608.*` entries continuously through **2026-08-03**. The monthly
> *listing* page lags; the *API* does not. **Standing rule for future sweeps: use the API, never the
> monthly listing, to establish the frontier.** As a result this sweep recovers a real and dense
> 08-01..08-03 slice that yesterday reported as nonexistent — twelve in-window items below, several
> landing directly on yesterday's v1 failure modes.

Fields swept: ARC-AGI-3 / ARC Prize results+writeups; LLM agents on interactive benchmarks
(test-time learning, exploration, world models); **agent context management / compaction / memory
eviction (A22 lane)**; agentic harness design; banking/replay for stochastic evals. Sources: arXiv
API (5 distinct query families across cs.AI / cs.CL / cs.LG, submittedDate-descending), arXiv cs.AI
monthly listing (as a control), arcprize.org/blog + /leaderboard, web search.

Campaign anchors for verdicts: **A22 compaction v2 is the single active lane** — region-aware
eviction with pinning (CWL 2606.11213, MemDecay 2607.10582), hygiene-gated digest (no
hedged/truncated facts, refuted-list never elided, no self-ingestion), RETAIN off by default,
suppress-eviction-while-stuck. **v1 seed-1 screen FAILED on two modes: (F1) toxic digest and
(F2) blind action batching.** Frozen filler runs daily; boristown prereg dormant (NC-14
mechanism-null); ledger n=19 STATIONARY, best 1.33; 72B route DEAD; **no cloud spend** — evals are
free Kaggle kernel builds; Kaggle 9h wall, no-final-rerun; private track = **no code-exec tools,
single model**.

Already-triaged, NOT re-reported: Living-Harness 2607.26598; MemoHarness 2607.14159; Agentic
Context Management 2607.21503; Tycho 2607.28287; Plans-Don't-Persist 2606.22953; MemHarness
2607.28272; Self-GC 2607.00692; Addressable Recall Compaction 2607.25066; CompactionRL 2607.05378;
OpenAI 07-29/07-30 retained-reasoning + compaction; Opus 5 30.2%; aTTT 2607.03441; TTA-via-Env
2511.04847; GSME/HarnessBank 2607.13683; EvoAgentBench 2607.05202; Rodionov exec-sims 2605.05138;
Rudakov 2512.24156; vLLM #39056; CWL 2606.11213; MemDecay 2607.10582; SelfCompact 2606.23525;
LightMem-repro 2607.29104; Zero-Mem 2607.29377; Filesystem-Memory 2607.26637; Aries 2607.29069;
ChronoMem 2607.27773; TraceViT 2607.29586; DungeonBench 2607.29577.

---

## A. NEW IN-WINDOW (arXiv 2026-08-01..08-03)

### A1. When Memory Becomes Authority: Benchmarking Authority Collapse at the Memory Consolidation Boundary  ⭐⭐ TOP HIT FOR F1 (toxic digest)
- arXiv:2608.01679 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.01679
- Gist: Defines **authority collapse** — consolidation "preserves a claim while erasing the source
  constraints governing its authorized use," so a stored fact is later treated as more authoritative
  than it was ever licensed to be. Observed in **48 of 49 configurations** across seven consolidators
  and LLM backbones. Collapsed memories without authority metadata produced a mean
  **unauthorized-action rate of 50.3%**. Fix: **automatically predict and persist an authority label
  alongside each consolidated claim** — end-to-end unauthorized actions **16.9% → 0.0%** with benign
  task success essentially unchanged.
- **Verdict: ADOPT (the authority-label field; it is a one-line schema change to the v2 digest).**
  This is the precise, published name for what killed A22 v1: the digest did not lose the facts, it
  lost the *epistemic status* of the facts, and the agent then acted on them as settled. Our v2
  hygiene gate already forbids hedged/truncated facts and forbids eliding the refuted-list — but
  those are *filters*, and 2608.01679 shows filtering is the wrong shape: the surviving claims still
  arrive stripped of provenance. **Concretely: every digest line must carry an explicit status tag
  (`CONFIRMED` / `HYPOTHESIS` / `REFUTED`) and the action-count or frame at which it was established;
  lines that cannot be tagged are dropped rather than emitted bare.** This is deterministic, costs no
  extra LLM call (the tag is part of the same emission), and turns "refuted-list never elided" from a
  special case into the general mechanism. Their 16.9→0.0 is on tool-authorization, not ARC — the
  transfer is by analogy, but the failure shape is identical and the mitigation is nearly free.

### A2. Judging Is Not Enumerating: Silent Omissions in LLM-Authored Acceptable Sets  ⭐⭐ SECOND TOP HIT FOR F1
- arXiv:2608.01000 (v1 **2026-08-02**) — https://arxiv.org/abs/2608.01000
- Gist: When an LLM *authors* a set (test suite, acceptable-answer set, reward spec) rather than
  *judges* membership, it omits silently and catastrophically. Numbers: models **detect planted
  over-inclusions 6-7x more often than planted omissions**; an LLM-authored suite admits only
  **19-42% of oracle-correct solutions** despite judging at F1 **0.74-0.90**; a production system of
  43,227 items **fails omission-first at 10:1**. Three interventions: (i) **emit the predicate, not
  the extension** → F1 ≈ **0.99**; (ii) gate with a known-correct probe → false rejection
  **58-92% → ≤5%**, but keeps only 5-39% of suites; (iii) reference-based repair → **3.3-10.6x** yield.
- **Verdict: ADAPT — restructure the digest from an enumeration into a predicate + a probe gate.**
  The 6-7x asymmetry is the mechanism behind F1 that our current hygiene gate cannot see: **our gate
  inspects what the digest *says*, and omission is invisible to that inspection by construction.**
  Two deltas: (1) **prefer rules over lists** — a digest line "ACTION5 toggles the gate colour" (a
  predicate) is far more robustly emitted than "the following 9 cells changed" (an extension); bias
  the v2 digest prompt toward mechanism statements and away from enumerated state, which the frame
  already carries anyway. (2) **Add a known-correct probe**: seed the digest input with one fact we
  independently know is true from the run log, and **discard the whole digest if that fact does not
  survive** — a deterministic, zero-extra-call canary for exactly the silent-omission mode, and the
  paper's cheapest intervention (58-92% → ≤5%). Their caveat applies: probe-gating rejects a lot of
  suites, so the fallback must be "emit no digest" (which is safe — that is the control), not "emit
  the ungated one."

### A3. Real-Time Detection and Repair of LLM Agent Failures  ⭐⭐ TOP HIT FOR F2 (blind action batching)
- arXiv:2608.02464 (v1 **2026-08-03**, Sunny Dubey) — https://arxiv.org/abs/2608.02464
- Gist: Mid-episode monitors for agents that "loop, cascade tool errors, drift off goal, fabricate
  results, or **silently absorb corrupted content**." Two **LLM-free** detectors: (i) a one-class
  echo-state-network ensemble with CUSUM alarms trained only on healthy runs — **0.71 detection at a
  5% false-alarm budget, AUROC 0.872**; (ii) **deterministic verification** that recomputes stated
  totals against actual tool results and confirms required calls executed — catches **60% of failures
  (96% with a coverage check) at 0 of 63 false positives**. Overhead **~200 microseconds/step, three
  orders of magnitude below a judge call.** Repair = rollback + re-execute: recovers **45% of failures
  vs a 16% resampling control (p=0.0005)**, task success **52% → 73%** for ~one extra model call per
  run. 2,823 episodes, three frameworks, Qwen/Llama/Gemini.
- **Verdict: ADOPT (the deterministic-verification half + the coverage check).** This is the single
  most directly usable item in the sweep and it is the missing control for F2. Blind action batching
  is *precisely* "confirms required calls were executed" inverted: the agent emitted a batch and never
  verified the batch landed. The deterministic half needs no training, no model, and no tokens — **for
  each emitted action batch, verify post-hoc against the frame that (a) the expected number of actions
  registered and (b) the claimed effect matches the observed delta; on mismatch, abort the batch and
  fall back to single-action mode.** 0/63 false positives at 60-96% catch is an extraordinary
  operating point and the ~200µs cost is free under a 9h wall. Their statistical ESN detector needs
  healthy-run training data we do not have at n=19 — **skip that half**, take the deterministic half.
  Also note "silently absorb corrupted content" is a named failure mode covering F1 as well.

### A4. Context Compaction Theory
- arXiv:2608.01326 (v1 **2026-08-02**; Tirmazi, Markelon, Bishop, Mitzenmacher) — https://arxiv.org/abs/2608.01326
- Gist: First formal foundation for compaction. Two games: the **Context Selection Game** (retain a
  subset of accumulated state — i.e. *eviction*) and the **Context Generation Game** (summarize into a
  bounded message — i.e. *digest*). Main theorem: **the Generation Game is equivalent to one-way
  communication complexity**, so known communication-complexity bounds transfer directly to compaction.
  Separation result: **there exists a query family for which generation needs strictly less budget than
  selection.** Includes a case study on Anthropic's compaction endpoint over set-membership queries.
- **Verdict: ADAPT (framing + one honest concession).** Two things land. (i) It gives our two A22
  halves their formal names and confirms they are *different objects*, not two settings of one knob —
  which supports reporting the eviction arm and the digest arm as separable effects rather than one
  fused "compaction" treatment. (ii) The separation result is a **genuine counterweight to the
  raw-over-constructed consensus** we have been accumulating (LightMem-repro §A1 08-03, PRO-LONG §B1,
  2605.12978 §B3): there are query classes where a *generated* summary is provably cheaper than any
  *selection*. So the correct prereg stance is not "digests are bad" but "digests buy budget only when
  the query is compression-friendly, and ARC-AGI-3's queries are frame-recoverable and therefore
  selection-friendly" — a falsifiable claim rather than a prejudice. Theory paper, no experiment for
  us to copy; do not build anything from it.

### A5. Practical Online KV Cache Compaction for LLM Agents  ⭐
- arXiv:2608.00902 (v1 **2026-08-02**) — https://arxiv.org/abs/2608.00902
- Gist: Systems study of online compaction on real agent trajectories. **Token Eviction (TE) preserves
  most accuracy at 80% KV reduction** and beats **Attention Matching (AM)** under imperfect proxies.
  Headline behavioural finding: **immediate compaction often hurts; delaying compaction so it can use
  the agent's actual future queries recovers much of the gap.** Proxy-query choice (boundary /
  repeat-prefill / delayed future-generation) is the core design lever.
- **Verdict: ADAPT (adds a second, independent leg to the timing rule).** Yesterday's prereg delta 3
  adopted *suppress-cut-while-stuck* from SelfCompact 2606.23525. This paper supplies a different and
  stronger reason for the same family of rule: **cutting early is bad because the cut is made without
  knowing what the agent will ask for next.** In ARC-AGI-3 that argues for **cutting lazily — at the
  budget ceiling only, never on a fixed interval or at level boundaries** — since the later the cut,
  the more of the agent's actual query distribution is observable. Also independent corroboration that
  **eviction > summarization under an imperfect relevance signal**, which is exactly our regime (we have
  no oracle for what matters). Caveat: KV-page level on serving workloads, not message-level; the
  policy shape transfers, the 80% number does not.

### A6. When Replanning Becomes the Bottleneck: Budgeted Replanning for Embodied Agents
- arXiv:2608.01428 (v1 **2026-08-02**) — https://arxiv.org/abs/2608.01428
- Gist: Two components. **E-RECAP** = cost-aware progressive token pruning that predicts token utility
  and prunes across transformer layers **while preserving critical head and tail tokens**. **BRACE** =
  a controller deciding *whether* to replan, *which mode*, and allocating an explicit token budget +
  latency SLO. Results: **62-92% fewer tokens per replanning call**; SLO violations **85.5-100% →
  4.7-50%**; in the hard setting **80.0% success at 4.6% violations**. The framed problem is that
  frequent LLM replanning creates latency spikes invisible to average success metrics.
- **Verdict: ADAPT (head+tail pinning; and one measurement idea).** "Preserve critical **head and
  tail** tokens" is an independent third vote for pinning-the-extremes (with CWL and MemDecay) and it
  matches the A22 v2 pin-set exactly: system prompt (head) + active reasoning span (tail). More
  interesting is the framing: **a failure invisible to average metrics.** Our ledger is a mean over
  runs; if compaction cost manifests as occasional wall-clock blowups rather than uniformly worse
  play, our n=19 mean would not show it. Cheap addition: **log per-decision latency and the count of
  compaction events per run**, so a null on score can still be diagnosed. BRACE's learned controller is
  out of scope (needs training, needs spend).

### A7. Diagnosing Search Behavior and Failure Modes in Long-Horizon Search Agents
- arXiv:2608.01913 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.01913
- Gist: Splits long-horizon failure into **retrieval gaps** (evidence never found) vs **utilization
  gaps** (evidence found but misused). Answer accuracy correlates **better with cumulative retrieval
  recall than with number of searches or context consumed**. And: **useful evidence usually appears
  early in the trajectory, yet agents keep searching, producing a long tail of low-yield steps.**
- **Verdict: ADAPT (diagnostic vocabulary for the A22 post-mortem).** The retrieval/utilization split
  is the right axis for reading an A22 null: if the region-aware arm loses, we need to know whether the
  needed fact was *evicted* (retrieval gap) or *retained and ignored* (utilization gap) — and those
  imply opposite next moves. Cheap to instrument by replaying which digest/context lines were present
  when a wrong action fired. The "long tail of low-yield steps" is also a second independent statement
  of the ARC exploration problem, but is not itself new evidence for us. Search-QA domain, not
  interactive; take the taxonomy, not the results.

### A8. When Memory Updates but Behavior Does Not: Repairing Implicit Stale Dependencies (StateAuditor)
- arXiv:2608.01619 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.01619
- Gist: The **implicit policy adaptation (IPA) gap** — the agent's memory *has* the updated value and
  the agent still plans around the old one. StateAuditor audits **from stored state to draft** rather
  than checking what the response says: an **LLM proposes** candidate state transitions with timestamped
  evidence, then **deterministic code validates each quotation against source entries and verifies the
  evidence is genuinely newer**; only verified transitions repair. STALE benchmark (400 scenarios,
  50-session histories): **0.736 vs 0.686 baseline, +5.0 pts paired, 95% CI [+2.9, +7.2]**; replication
  0.738 vs 0.680. Modest on HorizonBench.
- **Verdict: ADAPT (the audit *direction*, not the pipeline).** The load-bearing idea is cheap and
  ours already half-implements it: **stale beliefs are best caught by auditing state→plan, not by
  reading the plan.** In ARC terms, a REFUTED entry that keeps driving actions is exactly the IPA gap,
  and it is deterministically checkable — if an action is consistent with a refuted hypothesis, flag it.
  **But note the disqualifier: StateAuditor's proposal step is an LLM call**, violating the zero-LLM-in-
  the-memory-path constraint (Zero-Mem 2607.29377, CWL 2606.11213). Take the chronology/provenance
  verification (deterministic, we already need timestamps for §A1's status tags); leave the proposer.
  +5.0 pts with a CI that excludes zero is a real but small effect in a non-interactive domain.

### A9. HarnessCompass: Guiding Automatic Harness Evolution toward Generalizable and Effective Agent Harnesses
- arXiv:2608.01918 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.01918
- Gist: Automatic harness evolution with three ingredients: **global constraints to prevent
  task-overfitting**, **agent feedback beyond trajectory data**, and **per-component optimization before
  consolidation**. SWE-bench Verified with GPT-5.4: **Pass@1 54% → 66% in 5 evolution iterations**,
  beating prior AHE on both effectiveness and evolution efficiency; **the evolved harness transfers to
  held-out tasks and other models.**
- **Verdict: IGNORE for build (ADAPT one design rule).** The method needs many full agent runs to
  score candidate harnesses — that is precisely the budget we do not have (no cloud spend, free Kaggle
  kernel builds, ~1 screen/day). But the *one* transferable finding is the anti-overfit constraint:
  their gains generalize **because** they constrained evolution globally rather than per-task, whereas
  prior harness-evolution work overfits. That is a direct rhyme with our own history of public-LB
  luck-chasing, and it argues for keeping A22's intervention **mechanism-level and game-agnostic**
  (region types, not game-specific pins). No spend implication; nothing to build.

### A10. LiveMem: Maintaining Memory State Continuity in Long-Running LLM Inference
- arXiv:2608.02515 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.02515
- Gist: Carries computation forward through a **fixed-capacity memory state whose lifetime is
  independent of the active context** — context turnover with memory-state maintenance, **memory-oriented
  post-training**, and state-aware serving. On LongMemEval it answers from the memory state **even when
  the supporting evidence has been removed from the current context**; evidence-distance analysis shows
  information persisting beyond the active window. Claims leading performance among intrinsic-memory methods.
- **Verdict: IGNORE.** The mechanism requires **post-training** the model, which violates the
  zero-cloud-spend rule outright, and we run a fixed base model in a single Kaggle kernel. Filed because
  it is the strongest in-window statement of the goal A22 pursues by prompt-level means — worth knowing
  the intrinsic-memory route exists and is closed to us, so we do not mistake its results for evidence
  that our prompt-level version should work.

### A11. MemSIF: From Structured Interactions to Dual-Track Fact Memory
- arXiv:2608.01742 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.01742
- Gist: Two tracks — **CoreFact** (stable, schema-guided, consolidated at write time) and **ActiveFact**
  (formed on demand, promoted when supported by multiple historical sources *and* recurring query demand).
  LoCoMo **+2.29-8.79%**, LongMemEval-S **+2.87-6.15%**, five backbones, best Total ACC in all settings.
- **Verdict: IGNORE (one idea noted).** Gains are small, the domain is conversational QA, and write-time
  consolidation is the thing 2605.12978 (§B3) and the LightMem reproduction warn against. The one idea
  worth remembering is **promotion requires multi-source support** — i.e. a fact earns durability by
  being independently re-observed. That is a plausible future refinement of our CONFIRMED tag (confirm
  only on second independent observation) but it is not this week's change.

### A12. Learning What to Remember: Test-Time Training via Context Distillation
- arXiv:2608.01672 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.01672
- Gist: Test-time training that uses a long-window teacher's supervision to teach a model **how to
  allocate its memory capacity toward future relevance** — i.e. learning the retention policy rather
  than hand-designing it.
- **Verdict: IGNORE.** Requires gradient updates at test time and a long-window teacher; both are out
  under no-cloud-spend + single-model + 9h wall. Logged for the test-time-learning field coverage: the
  learned-retention direction is real and active, and it is the thing our deterministic policy is a
  cheap proxy for.

### A13. Also-ran in-window (logged, no verdict needed)
Screened and set aside: **2608.02110 IACM-RL** (intent-aware context management — RL post-training,
out of budget); **2608.01285 Router-Mem** (evidence-sufficiency early stop, −25-27% latency; QA-only,
but the "stop when memory suffices" trigger rhymes with our suppress-while-stuck rule); **2608.01543
V-Mem** / **2608.00962 PMMC** / **2608.01456** (multimodal memory — ARC is visual, but all three
require built retrieval machinery we have ruled out); **2608.02097 Fetch-then-Explore** (persistent
workspace, decouples selection from extraction); **2608.01247 RestoreKV**, **2608.00528 S⁴R** (KV
compression, serving-level); **2608.01534 R-Qwen** (recursive VLM, **+27.6% on static ARC-AGI**, needs
training — same class as TraceViT yesterday); **2608.00355 CurveShift** (separating capability gains
from ceiling effects — mildly interesting for how we read our own flat ledger); **2608.02442 Shortcut
Hacking** (eval-validity); **2608.00028** (flat vs hierarchical multi-agent resource accounting).

### A14. ARC Prize / ARC-AGI-3 ecosystem — nothing new (third consecutive null)
- **arcprize.org/blog: no new post.** Latest is still **07.06.26 "ARC Prize 2026: ARC-AGI-3 Milestone
  Prize #1"** (previous 05.01.26). **Third consecutive sweep with no ARC Prize publication.**
- **arXiv: no new ARC-AGI-3 paper since Tycho 2607.28287 (07-30).** A dedicated `abs:"ARC-AGI-3"`
  query returns nothing at all in the 08-01..08-03 window. The 2608 slice's only ARC-touching item is
  R-Qwen 2608.01534 (static ARC-AGI, training-based) — see §A13.
- **Leaderboard: could not be read.** `arcprize.org/leaderboard` renders its table client-side and
  returns only the scatter-plot description to a fetch. **No change detectable; no re-triage.** (Noted
  as a tooling gap — if leaderboard drift starts mattering, this needs a browser-driven check.)
- Milestone #2 deadline confirmed **2026-09-30**; competition close **2026-11-02**, winners **12-04**.
  Duck-harness public figures reconfirmed (1.21% ARC eval / 1.03 Kaggle, Qwen 3.6 27B FP8) — unchanged,
  no new Tufa Labs writeup in-window.
- **Nothing new on banking/replay for stochastic evals**, for the second consecutive sweep. The shelved
  EWM+banking lane gets no new evidence either way.

---

## B. NEW-TO-US BUT OLDER (surfaced this sweep; all load-bearing for the A22 lane)

### B1. PRO-LONG: Programmatic Memory Enables Long-Horizon Reasoning  ⭐⭐⭐ MOST DECISION-RELEVANT ITEM IN THE SWEEP
- arXiv:2607.20064 (v1 2026-07-22, v2 07-23; Fox, Wang, Rosu, Dhingra) — https://arxiv.org/abs/2607.20064
- **This is an ARC-AGI-3 context-management result and it does not appear anywhere in `learnings/`.**
- Gist: The agent keeps a **complete, structured, lossless interaction log** — structured headers plus
  the agent's plan, action, and resulting board state per entry — and queries it **programmatically**
  (regex or Python; 60.6% of its tool calls are Python). The paper is explicit: *"losslessness; nothing
  is compressed or summarized when writing, so the log is a faithful, ground-truth record of environment
  state."* Results on the **full ARC-AGI-3 public game set**: **+18.0 percentage points over the no-log
  coding-agent baseline**, **up to 76.1% pass@1** (matching SOTA harnesses) and **97.4% best@2 with
  Fable 5**, using **4.2-5.8x fewer tokens** than specialized alternatives.
- **Verdict: ADAPT — and it substantially raises the prior on the A22 lane while re-pointing it.**
  Three consequences, in order of importance:
  1. **It is the first hard number attaching context management to ARC-AGI-3 specifically: +18.0 pts,
     4-6x fewer tokens.** Every other paper on our compaction shelf argues from QA or serving. This one
     argues from our benchmark. The A22 lane is now the best-evidenced lane we have ever run.
  2. **The winning policy is lossless retention + cheap indexed search — the exact opposite of
     digesting.** Combined with LightMem-repro (retriever, not representation, is the lever: 58.1→75.5),
     2605.12978 (§B3), and §A5 (eviction beats summarization under imperfect proxies), **four independent
     lines now converge on: do not construct summaries; retain raw and improve access.** This is a direct
     indictment of the digest half of A22 v2 and an endorsement of the eviction half. Yesterday's v1
     toxic-digest failure was not bad luck — it was the predicted outcome.
  3. **Its literal method is private-track-illegal for us** (regex/Python over the log = code-exec tools,
     forbidden; and the log must fit somewhere). So the adaptation is: **keep the lossless-log *discipline*
     — never overwrite or summarize the record — and make the eviction policy govern only what is
     *presented in-context*, not what is *destroyed*.** That is a real and cheap change: an evicted
     region should be recoverable and re-presentable (e.g. a deterministic pinned index line naming what
     was cut and where), rather than gone. Under §A4's vocabulary this keeps us firmly in the Selection
     Game, where our query class lives.
- **Open action:** the paper says code and logs are on GitHub. Worth a follow-up sweep to see whether
  the log *schema* (structured header + plan + action + board state) is published — the schema itself is
  legal for us to copy even though the search tool is not.

### B2. Governance Decay: How Context Compaction Silently Erases Safety Constraints  ⭐⭐ MECHANISM PAPER FOR F1
- arXiv:2606.22528 (v1 2026-06-21, v2 06-27; Shiyang Chen) — https://arxiv.org/abs/2606.22528
- Gist: Constraints an agent **reliably obeys while visible** are silently deleted by compaction, and
  the same agent then performs prohibited actions. **1,323 episodes**: violations **0% with full
  visibility → 30% after compaction, up to 59%** for some models. The conditional is the striking part:
  **when the constraint survived summarization, violations stayed at 0%; when it was removed, 38%.** So
  the harm is *entirely* mediated by elision, not by degraded reasoning. Also introduces a
  **Compaction-Eviction Attack** (adversarial content biases the summarizer into omitting legitimate
  policy) and a **training-free mitigation, "Constraint Pinning," which restores violations to 0%.**
- **Verdict: ADOPT (Constraint Pinning by name — it is what our pin-set should be sized around).**
  This is the cleanest causal account of F1 available: **the digest does not corrupt the agent, it
  deletes the invariants and the agent then behaves rationally on what remains.** Two operational
  consequences. (i) Our pin-set is currently system prompt + active reasoning span; **the refuted-list
  and the game-invariant lines must be *pinned*, not merely *gate-protected***. "Never elided" as a
  hygiene check is a filter that can fail open; pinning cannot. (ii) The 0%/38% conditional gives us a
  clean falsifiable prediction for the A22 post-mortem: **if a run fails, check first whether the
  relevant invariant was present in-context at the failing step** — if it was present and the agent
  violated it anyway, our whole compaction hypothesis is wrong; if it was absent, the mechanism is
  confirmed and the fix is pinning. That is a strong, cheap discriminator between the retrieval and
  utilization gaps of §A7. Training-free and zero-token → fully budget- and private-track-legal.

### B3. Useful Memories Become Faulty When Continuously Updated by LLMs  ⭐⭐
- arXiv:2605.12978 (2026-05-13) — https://arxiv.org/abs/2605.12978
- Gist: Memory utility **rises then degrades, potentially falling below the no-memory baseline**, and the
  authors localize the cause to **the consolidation process itself, not the underlying experiences**.
  Headline number, and it is on our benchmark family: **even when consolidating from ground-truth
  solutions, GPT-5.4 fails on 54% of ARC-AGI problems it had previously solved without memory.** An
  **episodic-only control that simply retains trajectories remains competitive**, and with consolidation
  disabled in a controlled setting, agents **double the accuracy of their forced-consolidation
  counterparts.** Recommendation: *"treat raw episodes as first-class evidence and gate consolidation
  explicitly rather than firing it after every interaction."*
- **Verdict: ADOPT (as the justification for keeping RETAIN off and for gating the digest hard).** The
  "consolidating from ground-truth still loses 54%" result is the strongest single sentence in this
  sweep: **it removes the "our digest was just low-quality" explanation for yesterday's v1 failure.**
  Even a perfect digest degrades performance in this family, because the act of rewriting an episode
  into a claim destroys the grounding that made the episode useful. This retroactively validates the v2
  decisions to keep **RETAIN off by default** and to add **no-self-ingestion**, and it upgrades them from
  cautious defaults to the main hypothesis. It also supplies the *frequency* rule we lacked: **gate
  consolidation explicitly — never fire the digest per-interaction or per-level, only at a budget
  ceiling** (converging with §A5's delay-the-cut finding from a completely different direction).

### B4. Workspace Optimization: How to Train Your Agent (DreamTeam)
- arXiv:2605.09650 (2026-05-10) — https://arxiv.org/abs/2605.09650
- Gist: Frames agent improvement as **workspace evolution mirroring weight-space training** — artifacts,
  evidence, and textual feedback take the role of gradients. The **DreamTeam** multi-agent harness
  improves **ARC-AGI-3 SOTA from 36% → 38.4% with 31% fewer actions.**
- **Verdict: IGNORE (calibration only).** Multi-agent and workspace-evolution machinery is out of scope
  under single-model + no-code-exec + one-screen-per-day. Its value is as a **scale calibrator**: the
  public ARC-AGI-3 frontier moves in ~2-point increments from substantial harness engineering. Useful
  when sizing what an A22 effect could plausibly look like against our n=19 stationary ledger — we should
  not be preregistering hopes of large jumps.

### B5. ACM: Agentic Context Management for Long Horizon Tasks
- arXiv:2607.23809 (2026-07-26) — https://arxiv.org/abs/2607.23809
- Gist: Agents autonomously **compress, archive, and retrieve** their own context, reducing token
  pressure and enabling extended exploration. (Distinct paper from the already-triaged 2607.21503 of a
  near-identical name.)
- **Verdict: IGNORE.** Model-driven compression is the mechanism §B1/§B3/§A2 all indict, and
  self-directed archiving spends tokens inside the wall. Logged so the compaction shelf is complete and
  so it is not confused with 2607.21503.

---

## Summary

- **Relevant items this sweep: 18** — 13 in-window (12 arXiv 08-01..08-03 + the ARC ecosystem null) and
  5 new-to-us older papers, all on the A22 lane.
- **Methodology finding (most important non-paper result):** yesterday's "arXiv 2608 is empty" was
  **wrong** — an artifact of the monthly listing page, which still shows "No updates for this time
  period" while the API returns entries through 08-03. **Future sweeps must use the API to set the
  frontier.** Recovered slice was dense and directly on-lane.
- **Explicit absences (each is a finding):**
  - **ARC Prize blog: no new post for the third consecutive sweep** (latest still 07-06 Milestone #1).
  - **No new ARC-AGI-3 arXiv paper since Tycho (07-30).** Dedicated query returns nothing in-window.
  - **Leaderboard unreadable by fetch** (client-side rendering) — no change detectable; flagged as a
    tooling gap rather than a null.
  - **Nothing new on banking/replay for stochastic evals**, second consecutive sweep.
- **Non-IGNORE verdicts: 3 ADOPT, 7 ADAPT.**
  1. **ADOPT — Authority Collapse (2608.01679):** tag every digest line with epistemic status +
     establishing frame; drop untaggable lines. 48/49 configs affected; mitigation 16.9% → 0.0%.
  2. **ADOPT — Real-Time Detection & Repair (2608.02464):** deterministic post-batch verification that
     the emitted actions registered and the claimed effect matches the frame delta. 60-96% catch at
     **0/63 false positives**, ~200µs/step, LLM-free. Direct counter to F2.
  3. **ADOPT — Governance Decay / Constraint Pinning (2606.22528) + Useful-Memories-Become-Faulty
     (2605.12978):** pin invariants and the refuted-list rather than gate-protecting them; keep RETAIN
     off; gate consolidation to a budget ceiling. 0%-vs-38% conditional; **GPT-5.4 loses 54% of
     previously-solved ARC-AGI tasks even consolidating from ground truth.**
  4. **ADAPT — PRO-LONG (2607.20064):** **+18.0 pts on the full ARC-AGI-3 public set at 4.2-5.8x fewer
     tokens via a lossless log**; adopt lossless-record discipline, evict from *presentation* not from
     *the record*. Its programmatic search is private-track-illegal.
  5. **ADAPT — Silent Omissions (2608.01000):** digest as predicate not enumeration; add a known-correct
     probe and discard the digest wholesale if the probe does not survive.
  6. **ADAPT — Online KV Compaction (2608.00902):** cut lazily at the budget ceiling, never on a fixed
     schedule; eviction beats summarization under imperfect relevance signals.
  7. **ADAPT — Budgeted Replanning (2608.01428):** pin head *and* tail; log per-decision latency and
     compaction-event counts so a score-null is still diagnosable.
  8. **ADAPT — Context Compaction Theory (2608.01326):** selection and generation are formally distinct
     objects; report the two A22 halves separately, and state the selection-friendliness of ARC queries
     as a falsifiable claim.
  9. **ADAPT — Search Failure Taxonomy (2608.01913):** read any A22 null through retrieval-gap vs
     utilization-gap; they imply opposite next moves.
  10. **ADAPT — StateAuditor (2608.01619):** audit state→plan, not plan→text; take the deterministic
      chronology check, leave the LLM proposer (violates zero-LLM-in-memory-path).

## Effect on today's A22 work

The sweep is unusually decisive, and it points **against the digest half and for the eviction half** of
A22 v2. Four changes, no new spend, no change to the frozen filler daily.

- **CHANGE 1 — demote the digest, promote pinning.** Convergent evidence from four independent lines
  (§B1 lossless log wins on *our* benchmark; §B3 ground-truth consolidation still loses 54%; §A2
  omission is 6-7x harder to detect than over-inclusion; §A5 eviction beats summarization under imperfect
  proxies) says our v1 toxic-digest failure was **the predicted outcome, not an execution error.** The
  v2 hygiene gate is the right instinct implemented at the wrong layer: filters inspect what survives and
  are blind to what was dropped. **Replace "refuted-list never elided" (a filter) with pinning the
  refuted-list and the game invariants (a guarantee)** — §B2's Constraint Pinning, training-free, 0%
  violations restored. If effort is scarce today, ship pinning and ship the digest smaller.
- **CHANGE 2 — add deterministic post-batch verification (the F2 fix).** Per §A3: after each action
  batch, check against the frame that the expected number of actions registered and the claimed effect
  matches the observed delta; on mismatch **abort the batch and drop to single-action mode.** LLM-free,
  ~200µs, and the published operating point is 60-96% catch at **zero** false positives in 63. This is
  the cheapest concrete fix in the sweep and it addresses the failure mode we have no control for.
- **CHANGE 3 — status tags on every retained claim.** Per §A1: `CONFIRMED` / `HYPOTHESIS` / `REFUTED`
  plus the establishing frame or action-count; untaggable lines are dropped rather than emitted bare.
  This makes §B2's pinning implementable (you can only pin what is typed) and makes §A7's
  retrieval-vs-utilization post-mortem mechanical.
- **CHANGE 4 — cut lazily, at the ceiling only.** §A5 (delayed compaction recovers most of the gap) and
  §B3 (gate consolidation explicitly, never per-interaction) independently converge with yesterday's
  SelfCompact-derived suppress-while-stuck rule. **Fire eviction only at the budget ceiling — not per
  level, not per interval** — and keep logging when suppression fires.
- **Prereg deltas (carried + new):** yesterday's three stand (attribution = budget relief; zero LLM calls
  in the eviction path; suppress-cut-while-stuck). New: (a) report **eviction and digest as separable
  arms**, not one fused treatment (§A4); (b) prespecify the **invariant-present-at-failing-step check**
  as the primary post-mortem discriminator — invariant present + violated ⇒ the compaction hypothesis is
  falsified; invariant absent ⇒ mechanism confirmed (§B2); (c) **log per-decision latency and
  compaction-event counts** so a score-null remains diagnosable (§A6); (d) size expectations modestly —
  the public ARC-AGI-3 frontier moves ~2 pts per major harness effort (§B4).
- **Scope guards (carried):** no model-maintained structured store; no summarize-then-reinject; no
  post-training route (§A10) and no harness-evolution search (§A9) — both exceed the budget.
- **Follow-up for the next sweep:** check whether PRO-LONG's GitHub publishes the **log schema**
  (structured header + plan + action + board state). The schema is legal for us even though its
  regex/Python search tool is not.
- **Unchanged:** frozen filler runs daily; boristown prereg dormant (nothing this window bears on NC-14);
  ledger n=19 stationary; no cloud spend; evals stay free Kaggle kernel builds.
