# Research Sweep — 2026-08-15 (ARC-AGI-3 campaign, step 1c)

Repo: F:\kaggle\arc-prize-2026 | Zero cloud spend, Kaggle build-rail only. Research only —
no builds, no Kaggle writes, no pushes were made in producing this file.

**Context frame (what makes an item relevant today).** The campaign's root-cause finding is
unchanged and everything below is graded against it: **our agent FORGETS.** 31,744-token
context, 33 history messages, levels running 225 actions, and a `transitions` history that the
harness already holds as a preloaded Python global **which the agent never queries**. The
efficiency reframe was raised and refuted the same day (ceiling ≈1.26–1.36 LB, short of gold;
~40% of the gap is real capability), so an item only earns a verdict above IGNORE if it attacks
**forgetting** or **real capability**, and only earns ADOPT if it is additionally implementable
**locally/CPU-first or as a free Kaggle build**. Model-level lanes (122B MoE brain, LoRA serve)
have consumed ~4 push slots and ~3 GPU-h since 08-13 for **zero measurements**, so anything
that opens another infra-heavy lane is graded down on that basis alone, regardless of merit.

**Retired beliefs, not re-litigated here.** "Let the model notice it's repeating"
(RedundancyBench 24.88%); prompt A/B testing is noise; JEPA on ARC-AGI-3 Kaggle (3 strikes);
the exec-sim substrate as an execution substrate (S1 fired 08-10, E1 = 0 of 13). Items whose
only content reduces to one of these are IGNORE by default and are not given a paragraph.

**Dedup.** Yesterday's sweep (`learnings/ressweep_2026-08-14.md`) was thorough and batched a
large share of the Aug 11–13 cohort. Excluded here as already on the record: 2608.11588
(CoAdapt-GUI, ADAPT), 2608.12626 (EpicStar, ADAPT — abstract re-verified today, yesterday's
characterisation is correct), 2608.11775, 2608.11786, 2608.11949, 2608.12847, 2608.12888,
2608.10333, 2608.09251/09292/09819, 2608.12307, 2603.08960 (*qs* inequality), 2608.04066,
2607.28287 (Tycho), and the whole batched in-window memory cohort (RippleMem 2608.13334,
MindMemOS 2608.12428, ERSkill 2608.12720, Spatial Memory Agent 2608.12743, Governed
Persistent Memory 2608.12476, Formal Definition of Agent Memory 2608.11654, LycheeMemory V2,
Total Recall at What Cost?, 2608.10676, 2608.10502). I re-verified each of these against the
arXiv API and found no reason to reopen any of them.

**Sweep result: 16 items · 0 ADOPT · 2 ADAPT · 14 IGNORE.**

---

## ★ A structural finding about the sweep itself, stated before the items

**The daily 1c cadence has outrun arXiv's announcement cadence, and today is the proof.**
Date-range queries `submittedDate:[20260814 TO 20260818]` over `cs.AI ∪ cs.LG ∪ cs.CL` with
`abs:"agent"` return **zero results**; the same query from Aug 13 returns 46. Aug 14 was a
Friday and Aug 15 a Saturday, so the entire genuinely-new window available to this sweep is
**Aug 12–13**, which yesterday's sweep already read most of. Roughly 60% of today's search
effort went into re-confirming yesterday's dedup list.

This is not a complaint, it is a scheduling number: **step 1c should move to every 2–3 days**,
and the freed slot should go to the one thing today's brief says has no lane at all (66% of
turns emitting no world-model update). Recommended for Sunday's panel; not acted on here.

---

# FIELD 1 — Agentic harness design: instruction surfaces, delivery, and auditing

## 1. Harness-IF — Evaluating Instruction Following Across Instruction Surfaces in Coding Agents — arXiv:2608.11727 (v1 **Aug 12, 2026**)
Huang, Que, Zeng, Zhang, Wang, Chen, Wang, Hou, Pu, Yan, Huang · `https://arxiv.org/abs/2608.11727`
**VERDICT: ADAPT — as an endpoint-design correction to P1/P2's delivery metric, not as a mechanism.**
**Not previously on our record (0 hits).**

Opening sentence of the abstract is, almost verbatim, the objection our own screen protocol has
never answered: *"When a coding agent obeys a rule, it may simply have been going to do that
anyway."* They introduce **Against-Prior Accuracy (AP-Acc)**: score compliance **only on rules
labelled as opposing the agent's unprompted default**, where "unprompted default" is established
empirically by **re-running the same tasks with the rule withheld** across nine probe builds.
60 multi-turn coding items, 642-rule library, 256 rules receiving verdicts, 12 frontier models.
Result: raw accuracy 72.1–85.9%, **AP-Acc 66.1–78.6%**; **every one of the 12 models is worse on
against-prior rules**, by 3.6–7.4 points (mean 5.81), and the direction survives a common-support
analysis with item-clustered intervals. Prior control does not change the top build but
**exchanges three adjacent rank pairs**.

**Why this bites us specifically.** The entire P1 programme is pre-registered on a **delivery**
endpoint — `M0 ∈ [3%, 30%]`, "fraction of turns on which the block is delivered/used" — and the
08-12 sweep's successor arm (the pull-side `untried_here()` / `dead_pairs()` helpers) is
pre-registered on **exactly the same kind of quantity**: "fraction of analysis steps whose
emitted code calls ≥1 helper." Harness-IF's finding is that a raw rate of that shape **overstates
the mechanism's contribution by a model-specific margin**, because some fraction of those turns
would have gone the same way with the block withheld. We already hold the withheld arm:
`runs/null10` and the pre-block family runs *are* the "rule withheld" condition. We have simply
never differenced against them at the turn level.

**Second, sharper finding, and it lands on an arm that is not yet built.** Their counterbalanced
conflict pilot (nine separate builds) reports that **precedence does not follow prompt depth**:
system prompts, project files and user instructions all rank **ahead of tool and skill
descriptions**. The 08-12 sweep's #1 ADAPT proposes putting the pull-side helper functions and
their worked call **in the Python tool preamble** — i.e. on the *lowest-precedence surface these
authors measured*. That does not kill the arm, but it converts "a 27B may simply never call
them" from a hand-wave into a **named, published prior**, and it says where to put the
instruction if the arm is ever built. This is a placement constraint derived from an ablation,
not a prompt-wording change, so `feedback_prompt_is_noise` does not bite.

**Cheapest possible local test (CPU, no GPU, no push slot):** re-score the archived P1 canary
event logs against the archived no-block runs at matched `(game, board_hash)`, and report
**AP-delivery** = delivery rate restricted to turns where the block's implied action *differs*
from the action the no-block arm took at the same board state. Pure arithmetic over JSON already
on disk; ~1 hour; produces a number that can be compared to the sealed [3%, 30%] band before the
band ever has to fire.
**Named risk:** the against-prior subset may be small enough that the restricted rate is
uninterpretable at our n. If so, that is itself the finding — it means the sealed M0 band was
never powered to distinguish delivery from coincidence, and it should be reported to the panel
as an instrument defect rather than silently widened.
**Sequencing:** this is a *read* of data we already own. It does not queue behind the P1 arm and
it costs nothing. It should be done *before* the pull-side arm is priced.

## 2. AI Guardrail Survival under Single-Cycle Agentic Self-Summarization — arXiv:2608.11392 (v1 Aug 11, **v2 Aug 13, 2026**)
`https://arxiv.org/abs/2608.11392`
**VERDICT: ADAPT — the auditing half only; and as a pre-registered kill condition on A22.**
**Not previously on our record (0 hits).**

Studies what happens to a standing constraint across **one** compaction cycle, where a transcript
is replaced by a model-generated summary. Central claim, stated flatly: **"a presence check is
not a safety check."** When compaction does not drop a rule outright, it frequently leaves
**something that looks like a rule but does not act like one**. On behavioural replay, a degraded
residue leads the model to perform the prohibited action **far more often than an intact rule
does — all-case gaps of +34 and +57 points under two replay models, both positive.** Even intact
rules sometimes fail to fire. Two further results matter to us: **rule-form items are retained
substantially more often than prominence-matched facts**, and the loss is **silent at runtime**,
detectable only against a retained external ground truth (a constraint registry) — which reveals
textual absence but *still* not whether a surviving rule fires.

**Why this is not a duplicate of 2608.11775 (The Sleeping Agent, IGNORE'd yesterday).** Sleeping
Agent's claim is *what information* gist compression loses (temporal). This paper's claim is
*that surviving text stops functioning*, and it attaches a magnitude (+34 / +57) and an auditing
prescription. Yesterday kept one sentence of Sleeping Agent as a design warning; this is the
version of that warning with a number on it.

**Why it bites us, in two places.**
(a) **Our derived-state block C emits facts, not rules** — `dead_pairs`, `untried_here`,
board hashes. This paper's measured asymmetry says the fact class is **preferentially lost**
under compaction while the rule class survives. If A22 (the compaction lane, formally open, no
builds bought, revival condition R2 = a surviving mechanism claim) is ever revived with an
LLM summarizer in it, block C is in the class that degrades.
(b) **It generalises past compaction to any delivery endpoint.** We audit that the block was
*emitted*. This paper is the published statement that emission-auditing "gives false assurance."
Read together with item #1, the two papers converge on the same instrument defect from opposite
directions: #1 says a raw delivery rate overstates because of coincidence, #2 says it overstates
because presence ≠ firing. Our campaign already has a standing feedback entry — *audit the
instrument before trusting its verdict* — and this is the second consecutive sweep to surface
external support for it.

**Cheapest possible local test (CPU, no GPU, no push slot):** over the archived event logs,
partition turns by whether history truncation had already occurred, and for each turn where a
block-C fact is **still textually present**, check whether the action taken **contradicts** that
fact (e.g. re-executes a pair the block listed as dead). The `(board_hash, action)` memo needed
to adjudicate this already exists. The output is a single number — *contradiction rate given
presence* — which is precisely the quantity a presence-based delivery metric cannot see.
**Do NOT import** the constraint-registry machinery or reframe block C as rule-form text to
exploit the retention asymmetry: turning facts into rules is a prompt-wording change, and it
would trade an auditable deterministic fact for an instruction whose firing we cannot verify.
**Scope limit, stated honestly:** our harness **truncates**, it does not summarize. The
compaction-cycle mechanism therefore does not literally apply to the deployed runner today. The
transferable half is the *auditing* claim (which is harness-independent) and the *A22 kill
condition* (which applies the moment a summarizer is introduced). Graded ADAPT on those two,
not on the mechanism.

## 3. AutoDesign — Meta-Harness Optimization for Long-Horizon Agentic Design — arXiv:2608.13560 (Aug 13, 2026)
**VERDICT: IGNORE (per-task harness rewriting, and the task family is generation, not state tracking).**
A meta-harness optimizer directs a code agent to recursively improve its own harness from rollout
feedback; PosterBench Main Track 78.32, beating a closed commercial system by 7.45; the learned
DesignHarness lifts seven controlled code-agent configurations from 54.99 → 67.39 (+12.4%);
253 tool calls / 11 editing turns / 40 minutes / under $3 per run. It is the third
harness-self-improvement paper in eight days (after MemoHarness 2607.14159 and EvoHarness-RL
2608.05446) and it fails on the same two grounds each time: **the gains live in an open-ended
generation task with a rubric**, not in state tracking under a hard token budget, and
**recursive harness rewriting costs rollouts**, which is the resource our games already run out
of at 66–69k tokens. Logged so the harness-evolution family is visibly saturating with no item
that survives contact with our constraints.

## 4. Capability Sheaves for Compositional Agent-Harness Repair — arXiv:2608.13228 (Aug 13, 2026)
**VERDICT: IGNORE — but it is the most intellectually honest paper in the sweep and is worth one line of imitation.**
Models harness-component disagreement over shared state as a finite capability sheaf; controlled
experiment halves a candidate budget from 2,000 to 1,000 per cluster. Then, on the real split
(160 SWE-bench Multilingual issues, 875 candidate patches): **118 issues resolved vs 116 for a
matched non-cohomological selector, exact sign-flip p = 0.75; the abstention gate ties the strong
anchor at 127/160 with p = 1.0.** The authors write, in their own abstract, *"The discovery gate
therefore fails and the confirmatory split remains sealed"* and *"the study supports the
controlled invariance mechanism ... but not a real-world cohomological advantage."*
Nothing to lift — the machinery is heavy, the domain is code repair, and there is no ARC surface.
It is recorded because a paper that reports its own discovery gate failing and keeps its
confirmatory split sealed is the exact behaviour our screen protocol is trying to enforce, and it
is a useful external example to point at the next time an arm wants its gate widened after the
fact.

## 5. QuoteBench — How Matched Scores Can Hide Command-Path Failures — arXiv:2608.13547 (Aug 13, 2026)
**VERDICT: IGNORE as a method — one transferable sentence.**
56 one-shot tasks; replaying the *same model reply* through an added shell parser drops success
by **55.4–73.2 points**, and disclosure of the boundary recovers 30.4–60.7 for six of eight
configurations. Headline: **GPT-5.6-sol's matched gap of −3.6 points conceals −64.3 points of
transport damage and +60.7 points of model compensation.** The domain is Bash quoting and we have
no shell transport, so there is nothing to build. The transferable sentence is their closing
recommendation — report *generation contract, execution path, operating point, and final-state
validator*, never a matched score treated as an intrinsic model property. Our own agent emits
**Python through a chat endpoint into a harness that parses it**, and we have never separately
measured parse-loss from generation-loss. That is a latent instrument question, not a lane; noted
for the panel, not proposed as work.

## 6. Beyond Final Scores: Systematic Evaluation of Agents for Long-Horizon AI R&D — arXiv:2608.13417 (Aug 13, 2026)
**VERDICT: IGNORE (evaluation framework for a different task family; zero mechanism).**
Seven frontier models, 36 long-horizon tasks, rule-based within-run metrics across Solution
Framing / Execution / Feedback Control, plus controlled comparisons of **experience reuse within
and across tasks**. Conclusion: agents "operate more like engineering optimizers than fully
autonomous researchers", performance varies substantially run-to-run, and **experience reuse can
help or mislead**. That last clause is mildly corroborative of our position (accumulated context
is not automatically an asset), but the paper offers no mechanism, no ARC surface, and its
metrics are rubric-scored research tasks. Nothing to run.

## 7. DARC — Diagnosis Before Recovery: Turning Agent Failures into Selective Self-Correction — arXiv:2608.11772 (Aug 12, 2026)
**VERDICT: IGNORE (needs a development-set failure profile we do not have for unseen games).**
The premise is congenial and even quotable: *"failures need not trigger uniformly more
context"*, and generic recovery playbooks "broaden the agent's context precisely when the system
needs a narrower repair interface." DARC profiles task-family failure modes on a **development
set**, prunes mismatched interventions from a shared recovery library, and freezes a
verifier-selected policy for deployment; on ALFWorld / AppWorld / XBRL it improves task
performance **while reducing environment steps**. Two disqualifiers, and the second is the one
that matters. (i) It is a *development-set* method — the whole causal order depends on having
profiled the failure family before deployment. (ii) Our private-LB games are **different games**,
and profiling failure families on our 25 local games and freezing a policy from them is the
definition of what `feedback_arc_generalization_first` forbids — the same objection that killed
TraceCompiler (2608.02680) on 08-12. Same trap, new shape.

## 8. BENCH2ROBUST — Retry, Switch, or Abstain? — arXiv:2608.11977 (Aug 12, 2026)
**VERDICT: IGNORE (its failure model is stochastic tool failure; our environment is deterministic).**
Converts failure-free tool benchmarks into stochastic environments; Bayesian Tool Memory improves
robustness by up to **16.8 points without retraining**, RL adds complementary behaviour, combined
40.8–45.5% under injection. The "without retraining" half is the only part that could ever fit
our rail, but BTM's content is *recovery knowledge about which tools fail transiently* — and our
action space is five primitives plus a click, in an environment we have **proven deterministic**
(that determinism is the load-bearing assumption under the entire `(board_hash, action)` memo).
There is no transient-failure distribution for BTM to estimate. Adopting it would be importing a
fix for a defect we have measured ourselves not to have.

---

# FIELD 2 — Memory, context management, state carrying

## 9. In-window memory cohort — **IGNORE, and already batched yesterday.**
`RippleMem 2608.13334` (associative recollection; +3.95% LoCoMo, +11.87% LongMemEval-S, 30×
cheaper graph construction), `MindMemOS 2608.12428` (94.03% LOCOMO, 70.63% PersonaMem),
`ERSkill 2608.12720` (+31.3% with Qwen3-Next-80B, +28.1% with GPT-5.4-nano), `Spatial Memory
Agent 2608.12743`, `Governed Persistent Memory 2608.12476`, `Towards a Formal Definition of
Agent Memory 2608.11654`. I pulled full abstracts on all six today rather than trust yesterday's
one-liners, and **yesterday's batched verdict holds without amendment**: every one is
conversational QA, personalisation, embodied-video, or contract verification, and every one
assumes the agent **issues a retrieval query**. Our documented root cause is an agent that does
not consult state **it already holds in-process at zero latency**. That is still our finding and
not the literature's — now for the fourth consecutive sweep.

One refinement worth carrying forward, from 2608.11654: it formalises memory as a
**capacity-constrained coverage maximiser** tracing a *utility–capacity frontier*. Our block C is
a 900-character non-truncatable budget chosen by fiat. If the panel ever wants to justify that
number rather than assert it, this is the vocabulary — but it is a framing, it is instantiated on
Homer's *Odyssey*, and it buys nothing today.

## 10. LoongReflect — arXiv:2608.11967 (Aug 12, 2026)
**VERDICT: IGNORE (training framework; needs a privileged teacher and GRPO).**
Formulates reflection as a **memory-control policy** over a reversible trajectory tree with
explicit reflect/backtrack actions, where backtracking removes an unreliable branch from active
context while preserving a concise corrective lesson. The *representation* is genuinely close to
what P3 (frontier-first exploration over the observed transition graph) would want. But the
contribution is a two-channel training recipe — a fast channel distilling from a privileged
teacher and a slow channel running outcome-based GRPO — on multi-hop RAG and math. We have zero
cloud budget, no path to fine-tuning a 27B FP8 model on the Kaggle build rail, and no privileged
teacher. Same disqualifier as JAMEL (2606.01528) on 08-12.

## 11. StateBridge — arXiv:2608.13317 (Aug 13, 2026)
**VERDICT: IGNORE (needs hidden-state access we do not have).**
Training-free closed-form orthogonal alignment of one agent's final-layer hidden states into
another's input space, prepended as a continuous prefix; best or tied-best on 22 of 26
model-task pairs. Blocked by the identical constraint that killed OLIVIA (2605.11169) on 08-12:
**our agent emits Python through a vLLM chat endpoint and we have no hidden-state interface.**
It is also a *multi-agent* communication method and we run one agent. Two independent blockers.

---

# FIELD 3 — Skill banking and self-evolution

## 12. Skill/self-evolution cohort — **IGNORE, batched.**
`SkillEvo 2608.13120` (multi-turn user simulation as a feedback *generator* rather than an
evaluation endpoint; +23.0 over self-reflection, +15.4 over single-turn-QA-driven evolution, on
9 production cloud-service Skills), `SkillShapley 2608.13173` (Shapley attribution of individual
steps within a skill), `LOPD 2608.13040` (makes the teacher's privileged context itself learnable;
beats GRPO with <30% of the rollout budget), `Practice Makes Unsafe / SkillMisevo 2608.12851`
(all 21 evolved configurations author unsafe artifacts; carryover ASR 16.0% → 35.3% under three
malicious tasks).

All four IGNORE on the same two grounds: they require **either training or many rollouts**, and
they operate on a *skill library* accumulated across tasks — which, on a private leaderboard made
of games we have never seen, is the generalisation trap `feedback_arc_generalization_first`
exists to block. 2608.12851 is worth one sentence as a warning rather than a method: **an
unsafe/wrong success becomes reusable policy after its triggering input disappears.** Substitute
"wrong" for "unsafe" and that is the failure mode any bank-what-worked proposal for ARC has to
answer, including P3's.

---

# FIELD 4 — ARC-AGI-3 and interactive/game benchmarks

## 13. **Nothing new. Second consecutive day.**
The full-text date-sorted `"ARC-AGI-3"` query still returns **16 papers total**, most recent
**2608.04066 (Aug 4)**. There is **no ARC-AGI-3 paper in the Aug 12–15 window**. The
BALROG / NetHack / Crafter / TextWorld query returns nothing newer than 2608.08466 (Aug 9), and
that item is a harness-self-improvement paper, not a game result. No new ARC Prize blog post
surfaced. The third-party mirror still reads Opus 5 30.2% / GPT-5.6 Sol 7.8% / Opus 4.8 1.5%
(UNVERIFIED against the official board; leaderboard work belongs to 1b, not here).

## 14. ★ **Record gap: MAP — A Map-then-Act Paradigm for Long-Horizon Interactive Agent Reasoning — arXiv:2605.13037 (May 13, 2026)**
Liu, Ye, Sun, Zhu, Xiao, Han, Gu, Cai, Zhang · `https://arxiv.org/abs/2605.13037`
**VERDICT: IGNORE as a method — but it is a genuine hole in our record and it changes a count.**
**Zero hits across `learnings/`, `docs/` and ITERATION_LOG** — this ARC-AGI-3 paper has never
been on the campaign record despite twelve prior sweeps, and it is one of only sixteen ARC-AGI-3
papers in existence.

MAP names our exact pathology in its own vocabulary: current interactive agents acquire
environmental understanding **reactively during execution**, a "temporal inversion" producing
**Delayed Environmental Perception** and an **Epistemic Bottleneck** that "traps them in
inefficient failure cycles." Its remedy is three-phase and plug-and-play — Global Exploration →
Task-Specific Mapping (an explicit structured cognitive map) → Knowledge-Augmented Execution.
**On ARC-AGI-3 it enables frontier models to surpass near-zero baseline performance in 22 of 25
game environments.** It also ships MAP-2K and reports that training on map-then-act trajectories
**outperforms training on expert execution traces**.

**Why still IGNORE.** (i) The result is *frontier models*, on the public games — the same
discount we applied to Prime Agent's 95.5% and Tycho's 100.0 RHAE; nothing about the number
transfers to a 27B on our private rail. (ii) Its three-phase structure is a re-derivation of
AERA's EXPLORE → VERIFY → PLAN (2605.25931), which we already logged on 08-12 as independent
prior art for P2 — so it raises P2's prior slightly and adds no new mechanism. (iii) The
training half is unbuyable.

**What it does change: a count.** The published ARC-AGI-3 record now contains **at least four**
independent systems clearing >20 of 25 public games or equivalent — OPINE-World 20/25
(2607.01531), **MAP 22/25**, Tycho 100.0 RHAE (2607.28287), and the verification coding agent at
~99% RHAE (2607.15439) — against **one** published null, the maximal-deterministic-executive
agent with 0 completions across 52 runs (2608.04066). Yesterday's cross-paper regularity
therefore holds and strengthens: **on ARC-AGI-3 the published wins come from agents that build
and maintain an explicit external model of the game; the published null came from an agent that
took the choosing away from the model.** MAP is the entry in that pattern that requires no code
execution — its externalised object is a *map*, not an executable — which is the cheapest variant
of the pattern and the one closest to block C. That is the single sentence worth carrying to
Sunday.

## 15. ★ **Re-surfaced: ScrambleToolBench — arXiv:2608.02358 (Aug 3, 2026)** — on our record since the 08-05 sweep, and under-read
`https://arxiv.org/abs/2608.02358` · **VERDICT: IGNORE as new — elevated as the strongest published statement of our root cause, and as a warning about P1.**
It is already logged in `daily_brief_2026-08-05.md` and `sweeps/research_sweep_2026-08-05.md`, so
it is not a find. I re-read the full abstract today because its title is a one-line description of
our defect, and the body contains a claim we appear never to have extracted:

> *"When faced with structural changes such as mapping drift, agents fail to use deductive
> strategies such as cycle tracing, and instead exhibit **belief inertia** or fall back to
> **exhaustive search**. Increasing test-time reasoning only **amplifies this expensive
> brute-force search** rather than enabling deductive recovery. **While equipping agents with
> persistent memory reduces compounding errors, they remain unable to efficiently infer
> structural changes.**"*

Three things follow, none of which are new work, all of which are free.
(a) It is independent external confirmation, on an interactive terminal benchmark with hidden
tool behaviours, that the failure is **belief inertia plus exhaustive fallback** — our measured
re-exploration, named identically.
(b) *"Increasing test-time reasoning only amplifies brute-force search"* is a published prior
against any proposal to buy more thinking tokens per turn as a fix. Worth having on hand.
(c) **The clause that should worry us: persistent memory helps but is documented insufficient.**
P1's mechanism C *is* persistent memory (a derived-state block). This paper says that class of
intervention reduces compounding error but does not restore deductive recovery under structural
change — which is exactly the regime of the 8 latent-state games where P1 hard-disables
mechanism A. It is not a reason to pull the arm; it is a reason to be unsurprised if C delivers
and Δlc stays flat, and to have said so **before** the readout rather than after.

---

# FIELD 5 — Test-time learning / TTT, and exploration for sparse-reward interactive envs

## 16. **Nothing new in either field. Both categories stay closed.**

**(b) Test-time learning / TTT.** Zero in-window items. The date-sorted `test-time × agent` query
returns nothing after 2608.12313 (AVA-Encoder, video representation — irrelevant), and the two
nearest agentic items, CoAdapt-GUI (2608.11588) and DARC (2608.11772), are already adjudicated
above and yesterday. The category verdict recorded on 08-12 is unchanged and now three sweeps
old: **everything in TTT requires gradient updates at inference or a development-set profile,
and neither is available to a 27B FP8 vLLM server inside a Kaggle build kernel under zero
spend.** Closed unless the serving constraint changes.

**(d) Exploration for sparse-reward interactive envs (Go-Explore lineage, RND/curiosity).**
**Zero in-window items — third consecutive dry sweep.** A date-restricted
`intrinsic motivation ∪ novelty ∪ frontier ∪ sparse reward` query over cs.AI ∪ cs.LG for
Aug 11–18 returns 40 results of which **not one** is an exploration method; they are benchmarks,
finance, governance and infrastructure. The Go-Explore lineage query returns nothing dated 2026-08
at all. The most recent live items in this field remain JAMEL (2606.01528, IGNORE'd 08-12, needs
training) and Rudakov's graph-based ARC-AGI-3 exploration (2512.24156, Dec 2025, the published
prior for P3). **P3's design has no competition in the current literature and no new evidence for
or against it.** That is worth saying plainly: if P3 is built, it will be built on a December 2025
paper and our own measurements, and no amount of further sweeping is going to improve that
position.

**Also swept and rejected without a paragraph:** `Ready Cohorts 2608.12123` (GPU-resident agent
control paths; cluster-scale serving economics, irrelevant to a single-GPU build kernel);
`OmniScientist 2608.13558`, `Vero 2608.13522`, `VibeLifeBench 2608.10875`, `Harness security /
governance cohort 2608.12977 / 2608.12761 / 2608.12789`; `Latent On-Policy Self-Distillation
2608.13040` (folded into #12); the Aug 13 finance/clinical/circuit multi-agent applications.

---

### Net

**Zero ADOPTs, and the two ADAPTs are both instrument corrections rather than mechanisms — which
is the right shape for today.** Neither costs a GPU-hour, a push slot, or a build. Both are reads
of data already sitting on disk, and both can be done before the P1 arm is pulled without
confounding it, because neither touches the runner.

**The two ADAPTs say the same thing from opposite directions, and it is a thing this campaign has
already been told once.** Harness-IF (#1) says a raw delivery rate **overstates** the mechanism
because some compliance is coincidence — measured at 3.6–7.4 points across 12 models, every one.
Guardrail Survival (#2) says a presence check **overstates** the mechanism because surviving text
stops firing — measured at +34 and +57 points. Our P1 endpoint is a raw presence-flavoured
delivery rate with a sealed band of [3%, 30%]. On 08-14 we discovered a *different* pre-registered
band (the memory-channel rider's [0.5%, 3%]) was wrong by an order of magnitude and would have
failed while working correctly. That is now **three separate instrument defects in four days**,
and today's literature independently predicts a fourth in the same family. The cheap
against-prior differencing in #1 should be run before the P1 readout, not after it.

**The most useful negative is #15, and it is a paper we already had.** ScrambleToolBench's
sentence — *persistent memory reduces compounding errors, but agents remain unable to efficiently
infer structural changes* — is the pre-registered explanation for a P1 result where **C delivers
and Δlc stays flat**. Writing it down today means that outcome cannot later be read as either a
surprise or a vindication. It also retires, in advance, the reflex of answering a flat C with
"more thinking tokens": the same paper reports that increasing test-time reasoning **amplifies**
brute-force search rather than replacing it.

**The most useful positive is #14, and it is a hole in our own record.** MAP (2605.13037) reports
frontier models clearing **22 of 25 ARC-AGI-3 games** and has never appeared in any of our twelve
sweeps. Its number does not transfer, but its existence moves the published >20/25 count to four
systems against one null, and it is the cheapest member of that family — its externalised object
is a **map**, not an executable — which matters because our exec-sim substrate is CLOSED and
block C is already a degenerate map. That the record had a hole this size, on our own benchmark,
after twelve sweeps, is itself the finding: the gap was not in the literature, it was in us.

**Process recommendation, restated so it is not lost in the items:** step 1c should drop to every
2–3 days. arXiv announced **zero** in-scope agent papers on Aug 14–15, and roughly 60% of today's
effort re-confirmed yesterday's dedup list. The freed slot has an obvious claimant — the 66% of
turns emitting no world-model update, which today's brief records as having **no open lane**.

---
---

# MAP audit

*Added 2026-08-15, second pass, at the coordinator's direction. Research only — no builds, no
Kaggle writes, no cloud spend. Sources: `arxiv.org/abs/2605.13037`, `arxiv.org/html/2605.13037v1`
(method + Tables 2, 7, 8), plus abstract/protocol pulls on 2607.01531, 2607.28287, 2607.15439,
2608.04066 via the arXiv API.*

## 0. Verdict first

**MAP's "22 of 25" is NOT a solve count and the number is NOT competition-comparable. I
overstated it in the first pass and I am correcting that here.**

The paper's own sentence, quoted exactly from the results section, is:

> *"MAP achieves consistent improvements over ReAct in **22 out of 25 games** in total, with ReAct
> scoring near-zero across virtually all environments."*

**22/25 counts the games in which MAP beat its own ReAct baseline.** It is a win-rate against a
baseline that the same paper describes as scoring near-zero almost everywhere. It is not
"22 games solved," and I reported it in a way that invited that reading. On the arithmetic that
*is* commensurable — levels completed — summing Tables 2 and 7 gives **MAP ≈ 41 levels across the
25 public games, against ReAct ≈ 4.** For scale, Tycho establishes the public set contains
**183 levels**, so MAP clears roughly **22% of the public set**. Our campaign clears **17 levels**
on the local 25.

So the honest one-line comparison is: *a frontier API model with an explicit mapping phase clears
~41 of 183 public levels; we clear 17.* That is a real gap and it is worth about 2.4×. It is not
"a published system solves 22 of 25 games and we have never read it," which is what my first-pass
wording implied.

## 1. What the number actually measures — the comparability audit

Five checks, done against the paper rather than assumed.

**(a) Same games? YES — and this is the one place MAP is unusually well matched to us.** Table 2
reports TU93, SB26, VC33, RE86, AR25, WA30; Table 7 adds BP35, CD82, CN04, DC22, FT09, G50T,
KA59, LF52, LP85, LS20, M0R0, R11L, S5I5, SK48, SP80, SU15, TN36, TR87, SC25. **These are our game
IDs.** m0r0, re86, sk48, ka59, cd82, g50t, dc22, wa30 and vc33 all appear in our own
latent-state and diagnosis records under the same names. The evaluation set is the public 25 we
run locally. This is the strongest single point of comparability in the whole ARC-AGI-3
literature and it is why the paper deserved the audit.

**(b) Same metric? NO — and the paper does not define its own.** Section 4.1 says performance is
"measured via Relative Human Action Efficiency (RHAE), which assesses action efficiency relative
to human baselines," and **that is the entire definition given.** No formula, no normalisation, no
range appears in the text or in the Table 2 caption. The reported values are then internally
irreconcilable with every other paper's RHAE: MAP reports `RE86: Level 3, Score 11.59`,
`SB26: Level 3, Score 7.59`, `AR25: Level 3, Score 7.66`, while **Tycho reports RHAE on a 0–100
scale with 100.00 as saturation** and 2607.15439 reports "about 99% RHAE." A score of 11.59
cannot be the same quantity as a score capped at 100 unless it means something entirely
different. **MAP's "Score" column is an undefined quantity that shares a name with RHAE.** It is
not comparable to the other papers, and it is certainly not comparable to our Kaggle
leaderboard number (public max 1.33, gold line 1.58). **Only the "Level" column is usable.**

**(c) Same action budget? NO, and the difference is load-bearing.** MAP spends **30 actions per
episode on the mapping phase alone** on ARC-AGI-3 (vs 10 for ALFWorld and 15 for TextCraft /
ScienceWorld — ARC-AGI-3 gets the largest mapping allocation in the paper), then "remaining
budget" for acting. Our levels run ~225 actions and our games terminate `gave_up` at
**66–69k tokens**, i.e. we are token-bound, not action-bound. A 30-action upfront survey is
priced very differently in the two regimes, and the paper's own cost claim — Figure 4, "total
costs comparable to or lower than ReAct" — is a claim about a frontier model that does not have
our 31,744-token ceiling.

**(d) Same harness / same conditions as the Kaggle rerun? NO, and the paper is silent where it
matters most.** It **never mentions** the Kaggle competition, the private leaderboard, offline or
no-internet execution, or a local model. It refers only to "the standard evaluation protocol."
Worse, **Stage 1 (Cross-Task Global Exploration) is not specified for ARC-AGI-3 at all.** The
method text describes Stage 1 building an environment-general knowledge base `K_g` from
"manual trajectories" and "training tasks", run **once per environment, offline**, explicitly
"does not consume the per-episode budget" — but that description is instantiated for ALFWorld,
TextCraft and ScienceWorld, and **ARC-AGI-3 receives no comparable discussion.** I could not
establish from the paper whether `K_g` for ARC-AGI-3 is built per-game or per-benchmark, nor what
the "manual trajectories" are. **This is disqualifying on its own.** If `K_g` is per-game and
offline, the method has seen each game before it is scored on it, which the Kaggle rerun on
unseen private games structurally forbids. If it is per-benchmark, the paper does not say so. An
unspecified pre-execution offline phase on a benchmark whose entire premise is
skill-acquisition-from-scratch is exactly the thing that has to be nailed down, and it is not.

**(e) Same model? NO.** **Claude 4.6 Opus.** A frontier API model. Our rail is a 27B FP8 vLLM
server inside a free Kaggle build kernel with no internet. Nothing about a frontier-model number
transfers, and we have applied this discount consistently to Prime Agent (95.5%), Tycho (100.0)
and OPINE-World.

**Comparability verdict: NOT competition-comparable. Four independent grounds — undefined
metric, frontier model, unspecified offline pre-exploration phase, and no private-set or offline
protocol — any one of which is sufficient.** Per the coordinator's instruction, it is therefore
treated as **uncomparable, not impressive**, and it **does not change the campaign.** The single
number that survives the audit is the levels figure (~41 vs our 17), and it survives only as a
frontier-model reference point, not as a target.

## 2. Mechanism: does it attack our forgetting root cause?

**Partly — and the part that does is a mechanism we are already paying for.**

MAP's Stage 2 artifact `M_t` is a **text-based structured cognitive map** (Table 8 shows the
ARC-AGI-3 instance containing "Environment Layout", "Action Effects", "Game Rules"), built during
a dedicated mapping phase and then carried into execution as a conditioning input:
`a_t ~ π_θ(a_t | u, M_t, K_g, h_t)`. Strip the vocabulary and that is **a push-side,
non-truncatable derived-state block delivered every turn — which is P1's mechanism C**, currently
on the rail. MAP is therefore best read as **independent external corroboration of block C's
design at a much larger budget**, not as a new mechanism. It is the third such convergence
(after Prime Agent's kernel-state design and AERA's EXPLORE→VERIFY→PLAN), and its distinguishing
feature is the cheapest one: **its externalised object is text, not executable code**, which
matters because our exec-sim substrate is CLOSED (S1 fired 08-10, E1 = 0 of 13).

**The one genuinely liftable component I found is the stopping rule, and it is CPU-testable.**
Stage 2 terminates on a **dual-convergence criterion**: `Cond_A` = knowledge increment
`Δ|M_t| = 0` for k consecutive steps, **and** `Cond_B` = state novelty `r(o_t) = 1/√N(o_t)` below
threshold ε for k steps, with `T_stop = min{t | (Cond_A ∧ Cond_B) ∨ (t ≥ T_max)}`. Both terms are
**deterministic and zero-LLM**, and both are computable from the `(board_hash, action)` memo P1
already maintains — `N(o_t)` is just a visit count over board hashes.

**But I am not proposing it, and the reason is honest rather than cautious:** the rule stops an
*explore phase* so an *act phase* can begin, and **we do not have that phase split.** Adopting
the stopping rule requires adopting the phase structure, the phase structure is P2's
territory, and P2's phase structure was already independently derived from AERA on 08-12. So the
correct routing is: **log the dual-convergence criterion as the concrete stopping rule P3/P2
currently lacks**, and do nothing with it until a phase-split arm is actually priced. It does not
justify a build and it must not be smuggled into the P1 successor.

## 3. Cost, honestly, against the record

Nothing here is buyable, and the audit **subtracts** rather than adds.

- **No ADOPT, no new ADAPT.** The tally for 2026-08-15 stands at **0 ADOPT / 2 ADAPT / 14 IGNORE**;
  this audit does not add an item, it corrects the characterisation of item #14.
- **The infra argument does not arise**, because there is no infra proposal. The mapping-phase
  split would cost a build slot and confound the live P1 endpoint, and on the record since 08-13
  — ~4 push slots and ~3 GPU-h for **zero measurements** — a mechanism whose artifact is
  already on the rail in another form does not clear the bar for a slot.
- **The one free action is the one already recommended in §1 of the main sweep**: the
  against-prior differencing of the P1 delivery endpoint. That remains the cheapest thing on the
  table and it is unaffected by this audit.

## 4. The four-against-one-null table, audited

Every claim I cited in the first pass, checked against its source. **Not one is
competition-comparable, and the reason is the same for all of them.**

| System | arXiv | Model(s) | Game set | Metric | Headline | What the headline ACTUALLY counts | Comp-comparable? |
|---|---|---|---|---|---|---|---|
| **MAP** | 2605.13037 (May 13) | **Claude 4.6 Opus** | public 25 (our game IDs) | "Score", labelled RHAE, **no formula given**; values 3.34–11.59 | "22 of 25" | **games where MAP beat its own ReAct baseline** — not solves. Levels: **~41** vs ReAct ~4 | **NO** — undefined metric; frontier model; **unspecified offline Stage-1 `K_g`**; no private/offline protocol |
| **OPINE-World** | 2607.01531 (Jul 1) | LLM, **not named in abstract** | public 25 | action-efficiency vs human baseline | "solves **20 of 25** games, action-efficiency **78.4**" | **a genuine solve count**, "without per-game training" | **NO** — public set, API-model, online CEGIS with code execution; our exec-sim substrate is CLOSED |
| **Tycho** | 2607.28287 (Jul 30) | **GPT-5.6 Sol, Opus 5**, Opus 4.8 | public 25 = **183 levels** | RHAE, 0–100 | "**100.00 RHAE**, all 183 levels; Opus 5 uses 61% fewer scored actions than human baselines" | **genuine saturation of the public set** | **NO** — frontier API models on the public set |
| **Verification coding agent** | 2607.15439 (Jul 16) | GPT-5.4/5.5/**5.6-sol** | public | RHAE | "**~99% RHAE**, fully solves every public game, <half human actions" | **genuine saturation of the public set** | **NO** — frontier API models on the public set |
| **NULL — LLM Proposes, Executive Disposes** | 2608.04066 (Aug 4) | unspecified | public | level completions | "**0 completions across 52 gated runs**" | **a genuine null**, pre-registered | **NO** as a score — but the *direction* is informative and it is the only pre-registered arm in the table |

**What the table is actually worth, and it is not what I implied yesterday.** I framed this as
"four systems clear >20/25 against one null," with the implication that the public set is being
solved while we sit at 17 levels. The audit reframes it:

1. **Two of the four are saturation results from frontier API models** (Tycho, 2607.15439) and
   they say the public 25 is *tractable to a frontier coding agent with internet*. They say
   nothing about a 27B in an offline kernel, and they are the reason the public set is a poor
   proxy for the private one.
2. **One is a real solve count** (OPINE-World, 20/25) but its mechanism is **online programmatic
   world-model synthesis with code execution** — the substrate this campaign CLOSED on 08-10.
3. **One (MAP) was miscounted by me** and is a baseline win-rate.
4. **The null is the only pre-registered result in the table**, which is a comment on the
   literature, not on us.

**The corrected regularity, which is weaker but true:** on the *public* ARC-AGI-3 set, published
success tracks **frontier model capability plus an externalised model of the game**, and the two
saturating systems are both **coding agents with internet access**. There is **no published
result at all** for a small local model on the private set — which is the regime we and the
entire Kaggle field actually compete in. That absence is the real finding of this table, and it
argues *against* reading the public-set literature as evidence that we are underperforming.

## 5. How the miss happened — and the fix matters more than the paper

**It was not a search-term blind spot.** A full-text `abs:"ARC-AGI-3"` arXiv query returns MAP;
it returned it today, and it returned it on 08-12 and 08-14 when the same query was run and
reported as "16 papers total, most recent 2608.04066." **The corpus was being retrieved and only
the in-window slice was being read.**

**The actual mechanism is a bad baseline that every later sweep inherited.**
`learnings/panel_research_literature.md` — "ARC-AGI-3 Literature Survey — **verified
2026-07-06**" — is 35 lines and enumerates five systems: Rodionov 2605.05138, Rudakov 2512.24156,
DreamTeam 2605.09650, AERA 2605.25931, Sensi 2603.17683. It then states:
**"No other new ARC-AGI-3 methods papers surfaced for June–July 2026."** MAP was submitted
**May 13** — inside the period that survey covered and before its verification date — and it is
absent. So the back-catalogue baseline was **incomplete at the moment it was frozen**, and from
07-06 onward every sweep was window-filtered and treated that file as the record of everything
prior. OPINE-World (Jul 1) and 2607.15439 (Jul 16) later entered the record through other routes,
which disguised the gap by making the record look like it was growing.

**Three compounding factors, named so the fix is targeted:**
1. **A one-time survey was treated as complete without an enumeration check.** Nothing in the file
   records how many ARC-AGI-3 papers existed on 07-06, so "no others surfaced" was unfalsifiable.
2. **Window filtering after a frozen baseline.** Correct for cost, fatal when the baseline is
   wrong — the error becomes permanent and invisible.
3. **Title-shape bias.** MAP's title contains neither "ARC" nor "ARC-AGI-3"; it reads as a generic
   interactive-agent paper. In a 16-row listing skimmed for in-window items, it does not
   announce itself. The two ARC-AGI-3 papers the 08-12 sweep *did* discuss from that same listing
   (AERA, Rudakov) both name ARC-AGI-3 in the title.

**Recommended fix (a panel action, not something I have done).** The ARC-AGI-3 corpus is
**16 papers**. Enumerate it **exhaustively, once**, into a single registry file — one row per
paper with model, game set, metric, what the headline counts, and a **competition-comparability
flag** — seeded with the five rows in §4 above. Thereafter sweep only the window *against the
registry*, and require the registry's row count to be reconciled against a live
`abs:"ARC-AGI-3"` query on every sweep, so that "no new papers" becomes a checkable statement
instead of an assertion. Cost: one CPU-only pass, no GPU, no slot. It closes the blind spot
permanently and it would have caught MAP on 07-06.

---
---

# Panel agenda — Sunday 2026-08-16

*Two items submitted for the panel to rule on. Neither has been implemented, and I have changed
no cadence, no schedule and no configuration. These are proposals.*

## Item A — Drop step 1c (research sweep) from daily to every 2–3 days

**Proposal.** Move the research sweep off the daily loop to a 2–3 day cadence, keeping 1a
(result deep-dive) and 1b (discussions/leaderboard) daily.

**Evidence, from today.** Date-range arXiv queries over `cs.AI ∪ cs.LG ∪ cs.CL` with
`abs:"agent"` for `submittedDate:[20260814 TO 20260818]` return **zero results**; the identical
query from Aug 13 returns 46. Aug 14 was a Friday and Aug 15 a Saturday, so **arXiv announced no
in-scope agent papers at all in the period this sweep was supposed to cover.** The genuinely-new
window available today was Aug 12–13, which the 08-14 sweep had already read: roughly **60% of
today's effort went into re-confirming yesterday's dedup list**, and I pulled full abstracts on
six memory-cohort papers only to reproduce yesterday's batched verdict without amendment. The
supporting pattern is not one day: the last three sweeps read **0 ADOPT / 3 ADAPT**,
**0 ADOPT / 2 ADAPT**, **0 ADOPT / 2 ADAPT**, and every ADAPT in that run has been a refinement
of work already in flight rather than a new lane. A daily cadence against a literature that
announces on weekdays and moves in weeks is buying re-reads at full price. **Counter-argument the
panel should weigh:** the 08-14 sweep's single most valuable item (the *qs* inequality) was a
four-month-old paper found on a Thursday, and today's MAP audit is a back-catalogue find — so
the value in this workstream is coming from **depth on the back catalogue, not freshness**, which
is an argument for changing what the slot does as much as how often it runs. A 2–3 day cadence
plus the §5 registry pass would serve both.

## Item B — Reallocate the freed slot to the 66%-no-world-model-update finding

**Proposal.** Give the freed capacity to the finding the 08-14 brief records as having **no open
lane**: 66% of turns emit no world-model update at all, replicated independently over 596
archived event files and 50,140 model responses, and estimated at **~66× larger than the
reasoning-channel bug patched on 08-13**.

**What the lane would actually test, concretely.** The brief's open question #3 asks whether the
cause is **prompt, parser, or schema**, and that is a three-way discrimination answerable
**entirely offline on archived data, CPU-only, with no GPU, no push slot and no build**:
partition the 50,140 archived responses by whether the model emitted a world-model update in any
recognisable form, then classify the 66% that did not into (i) **model never produced one** — no
candidate text anywhere in the response; (ii) **parser dropped it** — candidate text present but
not matched by the harness extractor, which is measurable by re-running the extractor offline
with a relaxed grammar and counting recoveries; (iii) **schema rejected it** — extracted but
discarded for failing validation. The primary endpoint is the **three-way split with a
denominator stated up front**, and the decision rule is pre-registrable before the count is run:
if (ii) or (iii) dominates, this is a harness defect recoverable by a runner-side change with no
model involvement and no slot; if (i) dominates, it is a capability finding and it belongs to the
~40%-real-capability half of the gap, where it changes what the remaining ~80 days should buy.
**Why it outranks the alternatives:** every currently open lane (122B brain swap, LoRA serve) is
trying to make the model *better*, and this finding says the harness discards two thirds of what
the model **already produces** — and unlike those lanes it costs nothing to answer. **Blocking
dependency the panel must handle first:** the 08-13 memory-channel rider is pre-registered on a
band of [0.5%, 3%] that the 11×-larger replication puts at 0.32%, i.e. **the rider would fail its
own gate while working correctly**; the brief already flags that the band must be corrected
before the rider ships. Item B should not be opened until that band is re-sealed, or the two will
be read against each other.
