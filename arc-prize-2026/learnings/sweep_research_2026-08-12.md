# Research Sweep — 2026-08-12 (ARC-AGI-3 campaign, step 1c)

Repo: F:\kaggle\arc-prize-2026 | Zero cloud spend, Kaggle build-rail only.

**Context frame (what makes an item relevant today).** Today's diagnosis
(`learnings/war_room/efficiency_diagnosis_2026-08-12.md`) established: (i) action
efficiency, not level depth, is the binding constraint, but (ii) only **20%** of the
0.914-point gap is bookkeeping (+0.184), 40% is search policy, 40% is genuine capability;
(iii) the efficiency lane's ceiling is **~2.19 local ≈ 1.26–1.36 LB, SHORT of gold
1.48–1.58**; (iv) the **root cause of re-exploration is FORGETTING** — `context_budget_tokens:
31744` / 33 history messages on levels running 225 actions, while the harness already exposes
`history` / `transitions` / `last_transition` as preloaded Python globals **that the agent
never queries**. The arm on the rail right now (`canivel/arc3-duck-p1-eval` v1, prereg
`learnings/war_room/p1_prereg_2026-08-12.md`) is spending its slot on **mechanism C** — a
≤900-char non-truncatable derived-state block injected every turn. So the sweep was aimed
first and hardest at **memory / state-externalization under a small context budget**, and
graded everything else against "does this survive contact with a 27B agent on a free build
kernel."

**Bar applied.** ADOPT/ADAPT requires a named runner-side change AND a named endpoint on our
rail. Everything else is IGNORE. Default is IGNORE.

**Dedup.** Excluded as already on the campaign record: 2604.08224 (Externalization review,
already the KAOS SkillStore reference), 2608.07077 (Qwen3.6-27B world-model maintenance
deficit) and 2608.07429 (TEPA append-only memory) — both swept 08-10 and routed to P3;
VISTA (`vista-research.github.io`) — killed today via 734369; Brüggen animation-awareness —
built and KILLED today; the ARC Prize Opus-5 30.2% snapshot (swept 07-29).

**Sweep result: 15 items · 0 ADOPT · 3 ADAPT · 12 IGNORE.**

---

## 1. Prime Agent (Prime Intellect) — self-reported **95.5% RHAE on ARC-AGI-3**, MIT-licensed
`https://github.com/PrimeIntellect-ai/prime-agent` (MIT, 14.6k★, verified live) ·
announcement `https://x.com/PrimeIntellect/status/2085087000764568010` · writeup
`https://www.marktechpost.com/2026/08/06/prime-intellect-releases-prime-agent/`
**VERDICT: ADAPT (one mechanism only — the pull-side transition query surface).**

Released 2026-08-05. Architecture = **RLM + Continual Harness**: there are no fixed tool
schemas; the *only* tool is a **persistent IPython kernel**, and "skills, tools, and
sub-agents are pre-imported modules." Harness state is formalised `H = (ρ, G, K, M)`
(prompt, sub-agents, skills, memory), each CRUD-able; `/refine` reads the trajectory and
patches supplemental harness state (never the base prompt) with rollback. Sub-agents are
`rlm("sub-task")` calls returning asynchronously. Reported **95.5% RHAE Best@1 on ARC-AGI-3
with Opus 5** (runs 95.0 / 95.2 / 95.5, Best@3 99.97, all 183 levels), above the 95.4 human
baseline.

**Discount this number hard before reading anything into it.** It is self-reported; the
repo README does **not** mention ARC-AGI-3 (I checked); the official ARC-AGI-3 leaderboard
still shows Opus 5 at 30.2%; and the run is a frontier API model on the public API games,
not our 27B-on-Kaggle private rail. **Nothing about the score transfers.**

**What does transfer, and it is exactly today's root cause.** Prime Agent's central design
choice is that the agent reaches its own state *programmatically*, by writing code against a
live kernel, rather than by reading a rendered summary. **The duck harness already gives our
agent a Python tool and already holds `transitions` in that namespace — we have the
substrate and are not using it.** P1's mechanism C is the **push** side (harness renders a
block at the agent). This is the **pull** side.

**Concrete runner-side change:** add to the Python tool preamble a handful of pre-defined,
zero-LLM helper functions over the existing `transitions` global — `untried_here()`,
`tried_here()`, `dead_pairs()`, `path_to(board_hash)`, `distinct_boards()` — plus one
worked call in the preamble so the calling convention is unambiguous. No prompt-philosophy
edit (so `feedback_prompt_is_noise` does not bite), no model change, no new dependency,
~50 lines.
**Endpoint on our rail:** primary must be **delivery**, exactly as for P1 M0 — `fraction of
analysis steps whose emitted code calls ≥1 helper`, pre-registered with a band. Secondary:
dup-`(s,a)` rate and actions-per-cleared-level vs the P1 arm.
**Named risk:** a 27B model may simply never call them — the same failure class as "the agent
never queries `transitions`." That is precisely why delivery, not Δlc, is the endpoint.
**Sequencing:** strictly AFTER the P1 arm is pulled and read. C (push) and this (pull) must
not be confounded in one build.

## 2. Addressable Recall Compaction (ARC) — arXiv:2607.25066 (Jul 27, 2026)
Dang, Ichikawa, Fatima, Shirahata · `https://arxiv.org/abs/2607.25066`
**VERDICT: ADAPT (the lossless-eviction half only; route to the A22 lane as a mechanism claim).**

Separates archival storage from active-context presentation: tool observations go into an
**append-only, ID-addressable log**, and when space is needed older observations are
**replaced by compact citations** rather than dropped; the agent can then recall content by
identifier instead of re-executing the tool. NIAH 99.40% vs best baseline 88.12%;
LongBench-v2 Hard 29.97% vs 28.25%. **Evaluated on Qwen3-8B @16k and Qwen3-32B @32k** — i.e.
our exact model class and our exact context budget (31,744).

This is the closest published match to our defect all sweep. But take only half of it: the
`recall(id)` half **requires the agent to issue a query, and our documented root cause is
that our agent does not query things it has been handed.**

**Concrete runner-side change:** when the duck harness evicts a history message, do not drop
it — substitute a single-line, ID-addressable stub (`[s12 CLICK(3,4)@b#a71f -> no change]`).
Cost is bounded statically in characters (the same discipline that P1 adopted after the
animation arm died on a mis-specified token denominator), zero LLM calls.
**Endpoint on our rail:** dup-`(s,a)` re-execution rate (countable, already instrumented by
the P1 canary) and `levels_completed` ≥ 16 on the local 25.
**Routing:** this is a **compaction-lane** change, and A22 is formally open/unworked with
**no builds bought** and revival condition **R2 = a surviving mechanism claim is required**.
This is a candidate R2 mechanism claim. It does **not** on its own justify buying a build,
and it must not be smuggled into the P1 successor as an unpriced extra.

## 3. BeliefMem — Belief Memory: Agent Memory Under Partial Observability — arXiv:2605.05583 (May 7, 2026)
Liao, Wang, Zhu, Du, Yan, Chen · `https://arxiv.org/abs/2605.05583`
**VERDICT: ADAPT (narrow — the latent-state games only).**

Instead of storing one deterministic conclusion per fact, stores **multiple candidate
conclusions with probabilities**, updated by **Noisy-OR** as observations arrive, and
surfaces all candidates with confidence at retrieval. Targets the "self-reinforcing error"
failure: an agent commits to one reading of an ambiguous observation and never revisits it.
Evaluated on LoCoMo and ALFWorld; best average performance, no model sizes disclosed.

Relevance is specific and real: **8 of our 25 games carry ambiguous `(board_hash, action)`
pairs** (m0r0 55, re86 19, sk48 11, ka59 10, cd82 8, g50t 4, dc22 3, wa30 2), P1 handles them
by **hard-disabling** mechanism A, and the diagnosis calls **m0r0 (19.9% of the whole 56%)
essentially irreducible** for exactly this reason. Our current mechanism C says only "this
game has latent state" — one line, no per-pair information.

**Concrete runner-side change:** in block C, for games the online detector has flagged,
report per-pair **outcome multiplicity** (`tried n=3, outcomes 2 distinct`) instead of a
binary dead/alive label, and **never** emit "confirmed dead" for a pair with >1 observed
outcome. Pure arithmetic over the memo P1 already maintains; fits the 900-char budget;
zero LLM calls.
**Endpoint on our rail:** `levels_completed` on the 8 flagged games vs the family (must not
fall — this is already P1 kill rule #3), and dup-`(s,a)` rate on flagged games must stay
**above zero** (if it goes to zero we have suppressed legitimate re-probing of latent state,
which is the harm this change exists to avoid).
**Do NOT** import the Noisy-OR probability machinery — we have no calibration data for it
and it would put an unvalidated numeric model in the action path.

## 4. Memory in the Loop: In-Process Retrieval as Extended Working Memory — arXiv:2607.05690 (Jul 6, rev Jul 19, 2026)
Khan, Lipizzi · `https://arxiv.org/abs/2607.05690`
**VERDICT: IGNORE.**
Moves memory retrieval *inside* the reasoning loop with in-process storage (~100 µs vs 100+
ms networked); reports recall 0/5 → 3.6–4.8/5 across four GPT-5-class models and, notably,
**redundant actions 7.2 → 0.0 per 12 interactions**. That redundancy number is tantalising
and it is the reason this item is listed rather than dropped — but **their independent
variable is retrieval latency, and ours is already zero**: `transitions` is a preloaded
Python global in the same process. We have the paper's endpoint condition already and still
show 4.33× redundancy. There is no change to make. Corroborative only.

## 5. Less Context, Better Agents — arXiv:2606.10209 (Jun 8, 2026)
Lodha, Pahlavikhah Varnosfaderani, Chakraborty, Mithal · `https://arxiv.org/abs/2606.10209`
**VERDICT: IGNORE (corroborative; the one actionable variant spends our scarcest resource).**
On a 50-task enterprise tool-use benchmark (GPT-5, cross-checked on Sonnet 4.5): full history
71.0% @ 1,480,996 tok / 14.56 h; **pruned to last 5 tool-call pairs 79.0% @ 535,274 tok /
5.39 h**; **pruned + summarization 91.6% @ 553,374 tok**. Useful as external evidence that
our 33-message truncation is not itself the error — *unsummarised* truncation is. But the
delta they monetise is an **LLM summarization pass**, and our measured binding constraint is
the **token budget** (every game terminated `gave_up` at ~66–69k tokens). Buying summary
tokens to save action tokens is a trade we cannot price, and P1's block C already delivers a
deterministic summary at **zero** LLM calls. No change proposed.

## 6. JAMEL — Joint Agent Memory and Exploration Learning via Novelty Signals — arXiv:2606.01528 (Jun 1, 2026)
Tian, Weng, Kong et al. (13 authors) · `https://arxiv.org/abs/2606.01528`
**VERDICT: IGNORE (requires training; latent memory is unauditable).**
Correctly names our symptom — "over long trajectories, ineffective memory causes agents to
revisit exhausted behaviors" — and **trains** agentic memory and exploration policy jointly
against deterministic novelty signals (e.g. code coverage), in the GUI domain, claiming
open-weight beats and reduced token consumption. Two disqualifiers: (a) it is a training
framework and we have **zero cloud budget** and no path to fine-tuning a 27B FP8 model on the
Kaggle build rail; (b) it compresses to **latent** memory, which is exactly the opposite of
the auditable, canary-able, non-truncatable text block our screen protocol can actually read.

## 7. OLIVIA — Online Learning via Inference-time Action Adaptation — arXiv:2605.11169 (May 11, 2026)
Yu, Wu, Li et al. · `https://arxiv.org/abs/2605.11169`
**VERDICT: IGNORE (needs hidden states; and its premise is falsified in our traces).**
Inference-time only, no fine-tuning: models action selection as a **contextual linear bandit
over candidate actions with frozen hidden states as contexts**, UCB exploration, four
benchmarks. Two hard blockers. (1) It needs per-candidate **hidden states** from the serving
stack; our agent emits Python code through a vLLM chat endpoint and we have no such
interface. (2) Its motivating premise is "agents revisit similar decision states and repeat
the same local mistakes" — but our diagnosis measured **798/1,110 (72%) of actions on cleared
levels as non-repeating, information-producing probes**, and vc33 is provably cycle-free.
A bandit over repeated states has little to bite on here.

## 8. Redundant or Necessary? RedundancyBench — arXiv:2605.29893 (May 28, 2026)
Hu, Yang, Zhou, Liang, Guo, Yin, Han · `https://arxiv.org/abs/2605.29893`
**VERDICT: IGNORE as a method — but keep it as the citation that defends P1's design.**
Benchmark of manually-annotated redundant steps in agent trajectories; three LLM-based
detection strategies; **best method scores 24.88%, and some perform worse than random
guessing.** No deterministic detector offered. The lesson is a warning, not a mechanism:
**LLM-judged redundancy does not work.** P1 does not ask an LLM what was redundant — it
proves it with a `(board_hash, action)` memo under determinism, with an online latent-state
detector to guard the assumption. External support for that choice; nothing to import.

## 9. When Agents Do Not Stop: Infinite Agentic Loops (IAL-Scan) — arXiv:2607.01641 (Jul 2, 2026)
Hou, Wang, Zhao, Wang · `https://arxiv.org/abs/2607.01641`
**VERDICT: IGNORE (static analysis of source code, not runtime behaviour).**
IAL-Scan lifts agent code to a framework-independent IR, builds an Agentic Loop Dependence
Graph and checks whether feedback paths can repeatedly reach costly operations; 6,549 repos,
68 confirmed failures across 47 projects, 91.9% precision. It finds loops in **code**. Our
loops are **semantic** — a well-formed harness whose LLM re-derives a hypothesis it already
tested. Nothing to run against our runner.

## 10. Life-Harness — Adapting the Interface, Not the Model — arXiv:2605.22166 (May 21, rev May 27, 2026)
Xu, Wen, Li · `https://arxiv.org/abs/2605.22166`
**VERDICT: IGNORE (its gains live in failure modes our action space does not have).**
The most on-thesis-sounding item of the sweep — frozen weights, harness-only, "adapt the
interface" — with big numbers: 7 deterministic environments (τ-bench, τ²-bench, AgentBench),
18 backbones, 126 settings, improved 116/126, **+88.5% average relative**, and harnesses
learned on Qwen3-4B transferred to 17 other models. It is graded IGNORE because of *where*
those gains come from: **environment contracts, action realization, tool-contract
misunderstanding** — i.e. rich tool APIs being called wrongly. **Our action space is five
primitives plus a click.** We have no tool-contract failures to repair; our recorded failure
is state tracking. Adopting this would be importing a fix for a defect we do not have, which
is precisely this campaign's named failure mode.

## 11. MemoHarness: Agent Harnesses That Learn from Experience — arXiv:2607.14159 (Jul 14, 2026)
Huang, Wang, Bao, Ma, Luo, Nian, Zhuang, Liu, Zhao, Zhang · `https://arxiv.org/abs/2607.14159`
**VERDICT: IGNORE (no retrievable quantitative result; wrong task family).**
Decomposes the harness into six editable control dimensions and adapts it per test case from
a dual-layer experience bank, with no test-time labels or search. Shell-agent, code-gen and
analytical-reasoning benchmarks. **The abstract states improvement over fixed harnesses but
I could not retrieve a single number**, and per-case harness rewriting on a rail where one
arm costs a build slot and our screens are already under-powered at m=2 is unbuyable.

## 12. TraceCompiler — arXiv:2608.02680 (Aug 3, 2026)
El Yadouni (EPFL), Li (Binome) · `https://arxiv.org/abs/2608.02680`
**VERDICT: IGNORE (it is a competition-legal overfitting machine).**
Mines clusters of noisy agent traces and compiles them into mostly-deterministic workflows;
classifies value bindings; **0.928 P / 0.943 R on 15,775 dependency edges** vs 0.711 F1
baseline; 0.993 P on AppWorld; one intent went from 34 API calls to **11**. A 3× action
reduction is exactly our currency and we hold 5,151 recorded actions over 25 games — so this
is the item most likely to be mis-adopted. It must not be: it needs **repeated procedures**
to compile, our private-LB games are **different games**, and compiling workflows from our
25 local games is the definition of what `feedback_arc_generalization_first` forbids. It also
requires training on a labelled split.

## 13. Explore Before You Solve: Speed–Depth Trade-off in Epistemic Agents for ARC-AGI-3 — arXiv:2605.25931 (May 25, 2026)
Liew Keong Han · `https://arxiv.org/abs/2605.25931`
**VERDICT: IGNORE (its agent is 4× weaker than ours) — logged as prior art for P2.**
The only ARC-AGI-3-specific *method* paper found this sweep. Proposes **AERA** with a
three-phase EXPLORE → VERIFY → PLAN structure and formalises a Pareto trade-off between
action efficiency and information gain, under which **RHAE's quadratic form is a second-order
penalty for deviating from the frontier** — an independent derivation of exactly the quadratic
argument the campaign made today. But the results are far below ours: **RHAE 0.2116 with
Qwen2.5-0.5B, 4/25 public games solved; 0.30 private (code track)**, against our 1.33 public
max. Nothing to lift. Two things worth carrying: (a) EXPLORE/VERIFY/PLAN is our **P2
verified-plan batch gating** independently arrived at, which raises P2's prior slightly;
(b) the paper claims **24 of 25 public games are solvable by non-intelligent strategies** —
we clear 17 levels, so if that claim held on the same game set it would be a large public-set
headroom signal. It is an unreplicated claim from a 0.5B-model paper; **verify against our own
25 local games before it is ever cited as headroom.**

## 14. Graph-Based Exploration for ARC-AGI-3 (Rudakov, Shock, Cowley) — arXiv:2512.24156 (Dec 30, 2025)
`https://arxiv.org/abs/2512.24156`
**VERDICT: IGNORE as new (already on the campaign record) — but it is the published prior for P3.**
Training-free: segments frames, prioritises actions by **visual salience**, maintains a
**directed graph of explored states and transitions**; median **30/52 levels across six
games, 3rd on the ARC-AGI-3 Preview private leaderboard**. Surfaced by search this sweep;
it is the "Rudakov visual-priors" item already in the campaign memory, so it is not a new
finding. Recorded here because it is the closest published analogue of **P3 (frontier-first
exploration over the observed transition graph)** and its salience-prioritised ordering is
the one component P3 does not currently specify. Its vision front-end is out — two independent
same-model negatives on vision-in-the-loop at 27B (VISTA, and our own).

## 15. Quo Vadis, World Modeling? — arXiv:2608.02713 (Aug 3, 2026)
Yang, Yang, Wen, Kong et al. (20 authors) · `https://arxiv.org/abs/2608.02713`
**VERDICT: IGNORE (position paper, no method, no results).**
Argues world modeling should shift from physical state transitions to **agent-usable
information transitions** — execution outcomes, retrieved experience, verification signals —
organised into six proxy forms (dynamics, spatial, execution, memory/experience, skill,
reward/verification). Our P1 block C is, in their taxonomy, a memory/experience proxy at the
inference-time-guidance level, so the framing is congenial. It is a survey with no
experiment. Zero adoptable content.

---

## Searched and found nothing adoptable (recorded so the gap is visible, not silently skipped)

- **Plan-vs-act / when-to-replan / multi-action batching, empirical.** Searched
  specifically for a result that could rescue or refute P2's batch gating. Everything
  returned was either pre-2026 (Pre-Act 2505.09970, hierarchical planning 2504.16563) or
  DAG-orchestration for multi-agent query resolution (2603.11445), none of it measuring
  actions-per-decision against redundancy in a stateful game. **Our own +0.452 correlation
  between actions-per-analysis-step and log(redundancy) remains the only measurement that
  bears on P2, and it is ours.**
- **Test-time training / test-time adaptation for ≤32B models on novel tasks.** The category
  returned only 2025 work (Test-Time Adaptation of Tiny Recursive Models 2511.02886,
  Test-Time Adaptation for LLM Agents via Environment Interaction 2511.04847, In-Place TTT
  at ICLR 2026). All require **gradient updates at inference**, which is unavailable to a
  27B FP8 vLLM server inside a Kaggle build kernel under a zero-spend rule. **No new
  July–August 2026 item in this category.** Category closed for this campaign unless the
  serving constraint changes.
- **BALROG / Crafter / ALFWorld / TextWorld, July–August 2026.** Nothing new returned;
  results were the ICLR 2025 BALROG paper and its documentation. No fresh interactive-agent
  benchmark result in the window.
- **Kaggle discussion feed.** Not swept here — it was swept separately today (frontier
  734585, forum route restored) and is covered in ITERATION_LOG.

---

### Net

**Zero ADOPTs, and that is the correct outcome.** The three ADAPTs are all small,
runner-side, zero-LLM-cost refinements of work already in flight, and **none of them should
be built before `canivel/arc3-duck-p1-eval` v1 is pulled and read** — they are all
modifications to, or complements of, mechanism C, and building them now would confound the
one endpoint we are currently paying for.

**The single most important item is #1.** Prime Agent's 95.5% is not a number we can use,
believe, or chase — but its architecture independently converges on today's root-cause
finding from the opposite direction: it makes the agent reach its own state **by writing code
against a live kernel**, and the duck harness already has that kernel and already holds
`transitions` in it. P1's block C is the push side of that idea; the pull side is
~50 lines of helper functions in the tool preamble and one build slot, and it is the natural
successor arm **if and only if** C's delivery endpoint reads well.

**The single most useful negative is #8.** RedundancyBench's best LLM-based redundancy
detector scores **24.88%, with some strategies below random.** Every plausible-sounding
"have the model notice it is repeating itself" proposal that arrives from here on should be
answered with that number. Our deterministic `(board_hash, action)` memo is not the crude
version of a smarter LLM mechanism — it is the only version that works.

**The one item to be actively suspicious of is #12.** TraceCompiler's 34→11 action reduction
is precisely the shape of result this campaign is primed to over-adopt, and it would be
compiling workflows out of the 25 games we can see, for a private leaderboard made of games
we cannot.
