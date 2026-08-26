# ARC-AGI-3 Research Sweep — 2026-08-05

Window: arXiv submissions ~2026-08-03T13:00Z .. 2026-08-04T18:00Z (items new since the 08-04
sweep, which covered 08-01..08-03). Per the standing rule, the frontier was established via
`export.arxiv.org/api/query` (NOT the monthly listing).

> **METHODOLOGY NOTES.** (1) `http://export.arxiv.org` now 301-redirects to HTTPS with an empty
> body — scripts must query `https://export.arxiv.org/api/query` directly or silently get nothing
> (today's first query did). (2) The API frontier at sweep time is **2608.04007, 08-04T17:59Z**:
> **08-05 submissions are not yet announced** — the next sweep must re-cover 08-05 explicitly.
> (3) Long OR-chains inside `submittedDate:[..] AND (A OR B OR ...)` silently degrade to
> date-range-only matching (unrelated physics/math returned); use one quoted phrase per query.

Fields swept: ARC-AGI-3 / ARC Prize results+writeups; LLM agents on interactive benchmarks;
**agent context management / compaction / memory eviction (A22 lane)**; test-time
learning/adaptation; agentic harness design; banking/replay for stochastic evals. Sources: arXiv
API (12+ single-phrase query families across cs.AI/cs.CL/cs.LG, submittedDate-bounded),
arcprize.org/blog, web search, GitHub (PRO-LONG follow-up).

Campaign anchors for verdicts: **A22 compaction v2 is SEALED pre-build**
(`learnings/war_room/a22_compaction_v2_prereg_2026-08-04.md`): region-aware eviction with
pinning, digest demoted + hygiene-gated, RETAIN off, suppress-cut-while-stuck K=5, zero LLM calls
in the eviction path. **Per sealed-spec discipline, NOTHING in this sweep may change the v2
build mid-flight; all bearing items are routed to the R24 Sunday agenda as v3 candidates.**
Frozen filler runs daily; boristown dormant (NC-14); ledger n=19 STATIONARY, best 1.33; no cloud
spend; Kaggle 9h wall; private track = no code-exec tools, single model.

Already-triaged, NOT re-reported: everything in the 08-04 sweep's §A/§B (2608.01679, 2608.01000,
2608.02464, 2608.01326, 2608.00902, 2608.01428, 2608.01913, 2608.01619, 2608.01918, 2608.02515,
2608.01742, 2608.01672, the §A13 also-rans, PRO-LONG 2607.20064, 2606.22528, 2605.12978,
2605.09650, 2607.23809) plus the pre-08-01 shelf. Also skipped: **AERA / Explore-Before-You-Solve
2605.25931** surfaced in web search but is already known campaign-wide (16 files in `learnings/`).

---

## A. NEW IN-WINDOW (arXiv 2026-08-03T13:00Z..08-04)

### A1. LeanMem: Simple and Efficient Long-Term Memory for LLM Agents  ⭐ closest to the A22 lane this window
- arXiv:2608.03463 (v1 **2026-08-04**) — https://arxiv.org/abs/2608.03463
- Gist: Handle history **differently by compressibility, temporal dynamics, and fidelity
  requirement**: filter low-value content, then store as compact **profile** memory, temporally
  structured **event** memory, or source-grounded **record** memory. Maintenance updates **only
  the dynamically evolving event memories — stable profiles and immutable records are never
  re-consolidated.** Query-adaptive retrieval budgets at inference. LoCoMo/LongMemEval-S: up to
  **+15.1 pts over the strongest memory baseline at lowest or near-lowest cost.**
- **Verdict: ADAPT (R24 v3 candidate — write-once records).** The transferable principle is
  **immutability by type**: records whose fidelity matters are written once and never pass
  through consolidation again. That is the clean generalization of what our v2 already does ad
  hoc (deterministic extraction; refuted-list never rewritten) and it composes with 2605.12978's
  "gate consolidation" and 2608.01679's status tags: **REFUTED and established-FACT entries should
  be formally write-once in the store — later passes may append, never rewrite.** Conversational
  QA domain, so the +15.1 does not transfer as a number. NOT a mid-build change; agenda R24.

### A2. ParEvalLayer: When Partial LLM-Agent Evaluations Support a Decision  ⭐ methodology, directly our gate discipline
- arXiv:2608.02444 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.02444
- Gist: A decision layer over **partial** benchmark runs: given paired outcomes for two systems
  and a **pre-chosen comparison policy**, it emits one of four states — better-by-the-required-
  amount / not-better / **needs more evidence** / **abstain**. Replaying completed public
  benchmarks as if stopped early: **3 benchmarks reach the completed-run decision after only
  15-25% of task outcomes; others need far more** — i.e. a partial score alone is meaningless
  without stating what remains unresolved.
- **Verdict: ADAPT (R24 methodology agenda).** This is a published formalization of exactly our
  seed-1 screen + kill-rule practice (prereg thresholds, paired vs frozen baseline, VOID rules).
  Two uptakes for the gate-eval readout format, neither touching the sealed build: (i) adopt the
  **four-state output** — our current PASS/FAIL collapses "needs more evidence" into FAIL, which
  is how a K3 pause and a genuine harm signal get conflated; (ii) **report the unresolved-
  comparison mass** alongside any partial verdict (which games/pairs the decision does NOT yet
  cover). Cheap, deterministic, and it sharpens the K3-at-seed-1 vs K3-at-seed-2 distinction we
  already operate. Also the nearest-neighbor hit for the quiet banking/replay thread — but it is
  eval-economics, not banking (see §A15).

### A3. ScrambleToolBench: Agents Search Exhaustively Even When Their Own Map Points to the Next Step  ⭐ the utilization gap, interactive
- arXiv:2608.02358 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.02358
- Gist: Interactive terminal benchmark with **semantic cues removed** — tool behavior must be
  discovered by trial and error; adds mapping drift, stochastic action failures, temporal
  windows. Findings: under structural change, agents show **belief inertia or fall back to
  exhaustive search even when their own recorded map contains the next step**; **more test-time
  reasoning amplifies the brute-force search** rather than enabling deductive recovery;
  **persistent memory reduces compounding errors but does not produce efficient re-inference.**
- **Verdict: ADAPT (post-mortem instrumentation, R24).** This is 2608.01913's utilization gap
  demonstrated in a trial-and-error interactive setting close to ARC-AGI-3 mechanics, and it is a
  warning shaped like our lane: **retaining the right facts is necessary but demonstrably not
  sufficient — agents ignore their own map under drift.** Concrete uptake for the A22 forensics
  (not the build): alongside the sealed invariant-present-at-failing-step check, add its dual —
  **at each failure, check whether the retained notes/digest contained the correct next step**;
  if yes, the failure is utilization, and more retention (v3 work) would be aimed at the wrong
  gap. Also a caution on "more reasoning fixes stuck": it amplified brute force here, which is
  consistent with our suppress-cut-while-stuck being a *protection* not a *cure*.

### A4. Screenshots or Tools? Managing Multimodal Context in Hybrid GUI-MCP Computer-Use Agents
- arXiv:2608.03327 (v1 **2026-08-04**) — https://arxiv.org/abs/2608.03327
- Gist: Same MCP tools **help a reasoning model (+4.0pp) and hurt a non-reasoning one (−5.9pp)**
  under an identical harness (OSWorld-MCP, 309 tasks). Context-level finding: **after a
  successful tool call the next screenshot is often redundant; dropping it and halving image
  history cuts input tokens ~1/3** at small accuracy cost (retraining under the same observation
  rule removes the cost: 37.8% vs 33.0% at 53% of tokens).
- **Verdict: ADAPT (corroboration only — v2 already implements the analog).** "The newest
  observation supersedes prior observations" is precisely our v2 staleness proxy for episode
  eviction (every executed action's effect is visible in the current frame). This is an
  independent multimodal vote that dropping superseded visual state is nearly free. The
  retraining half is out of budget. Nothing to change; cite it as external support in the v2
  writeup. The tool-help-is-model-conditional result is also a neat rhyme with our
  feedback_kaggle_env_match discipline: identical scaffold, different model class, opposite sign.

### A5. ContinualSkillBench: Can LLM Agents Truly Evolve Their Capabilities?
- arXiv:2608.03874 (v1 **2026-08-04**) — https://arxiv.org/abs/2608.03874
- Gist: Dynamic benchmark for in-context continual skill learning (5 domains x 100 ordered
  subtasks with cross-task reuse). Findings: sequential execution helps, but **in-context
  learning performs comparably to explicit skill maintenance on average** — gains come from
  adaptation to prior context and feedback, not from reusable skill abstraction; **less capable
  models accumulate larger, fragmented, task-specific skill collections.**
- **Verdict: ADAPT (evidence for the lane's premise; nothing to build).** Two useful sentences:
  (i) keeping the right context in-window buys most of what an explicit skill library buys —
  supports A22-over-skill-store prioritization; (ii) weaker models fragment their skill stores —
  a caution against ever bolting a self-maintained library onto our mid-size model
  (feedback_simplicity_wins, again). Files under test-time-learning coverage.

### A6. Verifiable Memory (VerMem): Unified Memory Management with Local and Global Verifiers
- arXiv:2608.03137 (v1 **2026-08-04**) — https://arxiv.org/abs/2608.03137
- Gist: One policy controls LTM, active context, and episodic history via **seven atomic ops**
  (add/revise/soft-delete LTM; retrieve; filter/summarize active context; **restore episodic
  fragments**); SFT + 3-stage RL with train-time-only verifiers. Best on most metrics across five
  benchmarks.
- **Verdict: IGNORE (direction-confirming).** SFT+RL training route violates
  zero-spend/single-model. Noteworthy anyway: its op inventory is our region classes with a
  learned controller, and **"restore selected episodic fragments" is the recoverable-eviction
  discipline we adopted from PRO-LONG** — the learned version of the lane exists; ours is the
  deterministic proxy.

### A7. Harness-R1: Learning to Edit Executable Runtime Harnesses from Agent Failure Trajectories
- arXiv:2608.02276 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.02276
- Gist: A separate 9B "harness engineer" is RL-post-trained to convert target-agent failures into
  validated executable harness patches (frozen target reruns provide reward). WebShop/ALFWorld/
  DBBench: vanilla Qwen3.5-9B **44.3% → 53.6%**; after target fine-tuning, 59.2% → 64.2%.
- **Verdict: IGNORE (same class as HarnessCompass 2608.01918 yesterday — training a harness
  optimizer is out of budget).** One reusable checklist: its editable lifecycle = **context
  construction / tool mediation / action validation / execution recovery** — note that the two
  we have never systematically engineered (action validation, execution recovery) are exactly
  yesterday's 2608.02464 ADOPT (deterministic post-batch verification). Converging evidence that
  that ADOPT was the right pick.

### A8. RoMeRL: The Memory-Reward Trap in Self-Evolving Agent Memory
- arXiv:2608.02508 (v2 **2026-08-03**) — https://arxiv.org/abs/2608.02508
- Gist: Names the **memory-reward trap** — trajectory-level rewards are jointly credited to
  co-retrieved memories, so irrelevant experiences receive misleading utility and persist; fixes
  it with a fixed-dimensional per-task utility state (ALFWorld/LifelongAgentBench: −84.4% memory
  size, −21.1% LLM calls).
- **Verdict: IGNORE.** RL-learned memory utility is out of scope. Logged as a standing caution:
  **any future "learned eviction score" inherits this trap**; our deterministic class-based
  policy does not, which is now a citable advantage rather than a mere economy.

### A9. MutMem: Cryptographically Authorized Mutation in Persistent Agent Memory
- arXiv:2608.02843 (v1 **2026-08-03**) — https://arxiv.org/abs/2608.02843
- Gist: Every nontrivial memory-weight change is a signed, provenance-bound transition;
  poison-likely content is **retained with revisable labels** rather than deleted; 91.8%
  LongMemEval under LLM judgment.
- **Verdict: IGNORE.** Cryptographic machinery solves an adversarial-reviewer problem we do not
  have. It is, however, a second independent vote (after 2608.01679 authority collapse) for
  **provenance-carrying, label-not-delete memory entries** — both already adopted 08-04 (status
  tags; REFUTED retained forever). No delta.

### A10. MAFIA: Query-Only Memory Attacks against Audited LLM Agents
- arXiv:2608.03844 (v1 **2026-08-04**) — https://arxiv.org/abs/2608.03844
- Gist: Memory-poisoning attack that beats input auditing (90.7% ASR; audit detection 83.3% →
  7.4%) via retrieval-competitive placement and factual cloaks.
- **Verdict: IGNORE.** No adversary writes to our store; the one self-poisoning channel we have
  (digest self-ingestion) is already closed by the v2 round-trip break. Filed for field coverage:
  "compact factual cloaks evade audits" is the adversarial mirror of our hygiene gate's known
  blindness to well-formed falsehoods — the gate checks form, not truth; the probe-canary
  (2608.01000 ADAPT) is the partial answer.

### A11. ExpG: Robust Tool Use via Experience-Driven Adaptive Guidance
- arXiv:2608.03403 (v1 **2026-08-04**) — https://arxiv.org/abs/2608.03403
- Gist: Build and refine tool-usage guidance from historical trajectories (acquisition →
  distillation → reuse); smaller agents with ExpG beat larger ones without.
- **Verdict: IGNORE.** LLM-maintained experience pool with summarization-based distillation is
  the exact mechanism the shelf indicts (2605.12978; LightMem-repro), plus extra LLM calls inside
  the wall.

### A12. TurnSight: Turn-Level Hindsight Self-Distillation for Tool-Integrated Reasoning
- arXiv:2608.04007 (v1 **2026-08-04**) — https://arxiv.org/abs/2608.04007
- **Verdict: IGNORE.** RL post-training with hindsight teacher branches; out of budget. Logged as
  the current frontier of turn-level (vs trajectory-level) credit assignment — same failure
  framing our per-event sidecars are designed to diagnose deterministically.

### A13. Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility
- arXiv:2608.04001 (v1 **2026-08-04**) — https://arxiv.org/abs/2608.04001
- Gist: Formalizes TTS as budgeted inference over the prefix tree; three regimes
  (single-trajectory / leaf-level / prefix-level); prescribes protocol-matched reporting of
  compute and uncertainty.
- **Verdict: IGNORE (methodology shelf).** Nothing operational for a single-trajectory 9h-wall
  agent, but the "report the protocol, not just the scalar budget" norm matches our prereg
  practice; one-line citation if the A22 writeup ever needs it.

### A14. Also-ran in-window (logged, no verdict needed)
Screened and set aside: **2608.03276 TaskPress** (query-agnostic KV eviction conditioned on a
task guide — serving-level, but "meta-query-conditioned retention" rhymes with region classes);
**2608.02901 AnchorKV** (20x KV compression with zero discard — serving-level; the
never-discard-only-approximate stance is the KV-level cousin of recoverable eviction);
**2608.03764 GDPevo** (self-evolution benchmark with attributable-by-design train/test rule
splits — nice attribution idea, enterprise domain); **2608.02392 GROVE** (temporally stratified
memory from streaming video); **2608.02302** (agent-declared trajectory boundaries — curiosity
for block parsing); **2608.02880** (field-aware skill retrieval); **2608.02560** (O(1) SSM state
injection — architecture route, closed to us); **2608.03571** (environment-distribution design
for agent training); **2608.03502** (hybrid LLM-RL agents); **2608.03961 / 2608.03119** (TTS
sampling / label-free RLVR); **2608.02216 / 2608.02883 / 2608.03885** (vision TTA, out of
domain); **2608.02878 VeriTrace** (complete debugging action space → 100% Pass@1 VerilogEval —
"incomplete action space caps the ceiling" is a good slogan, domain far).

### A15. ARC Prize / ARC-AGI-3 ecosystem — still quiet (fourth consecutive null)
- **arcprize.org/blog: no new post.** Latest remains **07.06.26 "ARC Prize 2026: ARC-AGI-3
  Milestone Prize #1."** Fourth consecutive sweep without an ARC Prize publication.
- **arXiv: no new ARC-AGI-3 paper since Tycho 2607.28287 (07-30) — 6 days quiet.** The dedicated
  ARC-AGI query's frontier is still R-Qwen 2608.01534 (08-02, static ARC, triaged yesterday).
- **Leaderboard:** third-party snapshot (BenchLM, "verified 08-04") shows **Opus 5 (High) 30.2%
  leading, GPT-5.6 Sol 7.8%** — consistent with what we already carry; no re-triage. The official
  page remains unreadable by fetch (client-side render; standing tooling gap).
- **Banking/replay for stochastic evals: nothing new — THIRD consecutive quiet sweep.** Dedicated
  replay/trajectory-bank/variance queries returned only ParEvalLayer (§A2), which is partial-eval
  decision theory, not banking. The shelved EWM+banking lane continues to accrue no external
  evidence either way.

---

## B. FOLLOW-UP DISCHARGED — PRO-LONG log schema recovered (from 08-04 sweep's open action)

Repo is public: **https://github.com/alexisfox7/PRO-LONG** ("97.4% on ARC-AGI-3"), with agent
code (`prolong_agent/`), official scorecards, and **full sanitized Fable 5 online run logs**
(`release_logs/<model>/<cohort>/<game>/rep<N>/{logs.txt, logs_analyzer.txt, scorecard.json,
workspace/}`; spec in `release_logs/FORMAT.md`). The log schema, extracted from
`environment/runner.py::_log_action` + `agent/game_state.py`:

- Append-only `logs.txt`; entries separated by an 80-char `=` rule.
- Header line: **`Action N | Level L | Attempt A [| Plan Step i/j] | Score: S`**.
- Optional **`[PLAN]` block** — the agent's own plan/hint, carried forward in the log until the
  next analysis (their "stateless" ablation removes exactly this carry-forward, isolating the
  objective trace).
- **`Tool Call: NAME({json})`** — ACTION6 logged with explicit `{"x":..,"y":..}`.
- Board state as ASCII grid (hex mode), with animation layers labeled `[frame i/n]` and the
  settled grid `[settled]`; initial state logged as `[INITIAL BOARD STATE]`.
- Sidecars: `current_board.txt` (Score/Action header + current grid, overwritten each step) and a
  workspace `notes.md` / `actions.json`.

**Assessment:** the schema itself is deterministic-write and thus fully legal for us (it is the
*search* over it that is private-track-illegal). It is close kin to our mechanical store +
per-event sidecars; deltas worth considering are the **header format** (Action|Level|Attempt|
Score on every entry — cheap, greppable provenance, and it is exactly the "establishing frame"
field the 08-04 status-tag ADOPT needs) and the **explicit [PLAN] carry-forward as a typed
block**. License: none visible in the repo — flag before copying code verbatim (schema imitation
is fine; code reuse is not needed anyway). **R24 v3 candidate; NOT a mid-build change.**

---

## Summary

- **Relevant items this sweep: 16** — 13 in-window arXiv with verdicts + the also-ran batch + the
  ARC-ecosystem null + the PRO-LONG schema follow-up (discharged).
- **Non-IGNORE verdicts: 0 ADOPT, 5 ADAPT** — a lighter window than 08-04, and per sealed-spec
  discipline every ADAPT routes to the **R24 Sunday agenda as a v3 candidate**, none to the build:
  1. **ADAPT — LeanMem (2608.03463):** type-differentiated store; REFUTED/established-FACT
     entries formally write-once (append-only, never re-consolidated).
  2. **ADAPT — ParEvalLayer (2608.02444):** four-state gate readouts (better / not-better /
     needs-more-evidence / abstain) + report unresolved-comparison mass with any partial verdict.
  3. **ADAPT — ScrambleToolBench (2608.02358):** add the utilization-gap dual to the post-mortem —
     at failure, check whether the retained notes contained the correct next step; belief inertia
     is a policy failure retention cannot fix.
  4. **ADAPT — Screenshots-or-Tools (2608.03327):** independent multimodal confirmation that
     dropping superseded observations is nearly free — external support for the v2 staleness
     proxy; nothing to change.
  5. **ADAPT — ContinualSkillBench (2608.03874):** in-context adaptation ≈ explicit skill
     maintenance; supports A22-over-skill-store and warns against a self-maintained library.
- **Explicit absences (each is a finding):**
  - **ARC Prize blog: 4th consecutive sweep with no new post** (latest still 07-06).
  - **No new ARC-AGI-3 arXiv since Tycho 2607.28287 — 6 days.** PRO-LONG's public repo (§B) is
    the only ARC-AGI-3 ecosystem movement, and it is an artifact drop, not a new paper.
  - **Banking/replay: 3rd consecutive quiet sweep.** Nearest item (ParEvalLayer) is
    eval-economics, not banking.
  - **08-05 arXiv not yet announced** at sweep time (frontier 08-04T17:59Z) — next sweep must
    cover 08-05 as new territory, not assume it was swept today.
- **Nothing in this window contradicts the sealed v2 design.** If anything the window reinforces
  it: VerMem independently converges on our op set with a training budget we don't have; Harness-R1's
  lifecycle checklist lands on the same post-batch-verification gap yesterday's ADOPT filled;
  RoMeRL names the trap our deterministic eviction policy structurally avoids.

## Effect on today's A22 work

**None permitted, none needed.** The v2 spec is sealed pre-build; this window produced no
evidence of a design error in it, and no ADOPT-grade mid-build item. Standing list for **R24
Sunday (v3 candidates + methodology):**
1. Write-once immutable records in the store (LeanMem §A1 + carried 2605.12978/2608.01679).
2. PRO-LONG header schema (`Action N | Level L | Score: S` per entry) as the establishing-frame
   field for status tags; typed [PLAN] carry-forward block (§B).
3. Four-state partial-eval readout + unresolved-mass reporting for gate evals (ParEvalLayer §A2).
4. Utilization-gap forensics line: retained-notes-contained-the-answer check at each failure
   (ScrambleToolBench §A3, complements the sealed invariant-present-at-failing-step check).
5. Carried from 08-04: eviction/digest as separable arms; per-decision latency + compaction-event
   logging; modest effect sizing (~2 pts per major harness effort).
- **Scope guards (carried, reinforced):** no learned eviction scores (memory-reward trap §A8), no
  LLM-maintained experience pools (§A11), no post-training routes (§A6/§A7/§A12).
- **Follow-ups for next sweep:** (a) cover arXiv 08-05 (not yet announced today); (b) PRO-LONG
  repo license check before any code-level reuse (schema imitation needs none); (c) ARC Prize
  blog — a Milestone Prize #2-related post is increasingly overdue (deadline 09-30).
- **Unchanged:** frozen filler daily; boristown dormant; ledger n=19 stationary; zero cloud
  spend; evals stay free Kaggle kernel builds.
