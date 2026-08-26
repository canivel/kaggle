# RESTART RESEARCH — Does the ARChitect (TTT / fine-tuning) recipe transfer to ARC-AGI-3?

**Written 2026-08-15 for the Sunday 08-16 five-reviewer panel.**
**Research only. Zero cloud spend, zero GPU, zero pushes, zero submissions. Every number below is
either read from a public API in this session or cited to a file in this repo.**

Provenance tags: **[V]** verified by direct read/measurement in this session · **[C]** computed from a
verified source · **[INF]** inference · **[UNK]** not established.

---

## 0. VERDICT (read this if you read nothing else)

**TRANSFERABILITY: DOES-NOT-TRANSFER as published. The half that transfers is already our LoRA
lane; the half that makes ARChitect win does not exist in an interactive benchmark.**

**Q2 RULING: INFERENCE FROM AUTHORSHIP, NOT DISCLOSED METHOD.** There is zero public, citable
evidence about the method behind Daniel Franzen's 2.58. No discussion post, no notebook, no model,
no repo, no paper, no interview. [V]

**And two load-bearing premises in the order are factually wrong on the public record:** [V]

| Order says | Public leaderboard CSV says |
|---|---|
| "His arrival at 2.58 **in one submission**" | **41 submissions.** He moved **1.24 → 2.58 across 3 submissions** in the 08-13→08-15 window. He submitted **38 times before the jump and never exceeded 1.24** — which is *below our own 1.33*. |
| "That is precisely the lane we identified and **have not executed**" | The lane is designed, costed at 7–10 GPU-h inside the free allowance, **stage 1 and stage 1c are DONE**, and we have already spent **2 slots + ~1 GPU-h** on stage S. It is not un-executed; it is **stalled on two non-mechanism failures**. |

The "38 submissions at ≤1.24, then +1.34 in three" profile is itself evidence *against* the
authorship inference in its natural reading. Whatever Franzen deployed on 08-13/08-14, it was **not
a 2024 recipe he could have dropped in on day one** — he tried 38 times without it.

---

## 1. Q2 — SEPARATING THE EVIDENCE FROM THE INFERENCE

### 1.1 Direct evidence for "Franzen's 2.58 uses TTT": NONE [V]

Exhaustively searched and empty:
- Kaggle user `dfranzen`: **1 discussion post ever**, 8 months old, about ARC Prize **2025**. Zero
  posts in `arc-prize-2026-arc-agi-3`. Public kernels: **2, both historical** (2024, 2025 solutions).
  Datasets: 3, newest 2025-10-27. Models: 3, all ARC-AGI-1/2 artifacts. **Nothing dated 2026.**
- `github.com/da-fr`: 5 repos, newest `Product-of-Experts-ARC-Paper` (ICML 2025). No 2026 successor.
- `da-fr.github.io`: sections for 2024 (1st, "test-time training") and 2025 (2nd, masked diffusion).
  **No 2026 section at all.**
- arcprize.org blog / Milestone #1 writeup: names Tufa Labs, Reki, forge. **Franzen not mentioned.**
- arXiv: no 2026 Franzen/Disselhoff/Hartmann paper.
- No X/LinkedIn post found. The competition has no Writeups tab.

**Why the silence is structural, not accidental.** `arc-prize-2026-arc-agi-3` is a FEATURED CODE
COMPETITION; disclosure is only forced to claim milestone money. Franzen's 2.58 landed after
Milestone #1 (June 30). **The next forced-disclosure date is 2026-09-30** (Milestone #2: 1st $25K,
2nd $10K, 3rd $2.5K — he is currently in the $10K seat). That is a **free, dated falsification
event ~6 weeks out.**

**Of the entire top 10, nobody has disclosed a method.** Confirmed by direct API check: [V]
`nikitasorokin` 0 kernels, `abelincoln1865` 0, `gatamaz`/`tehnar` (cstl) 0 ARC kernels, `mlrush` 0,
`yutokojima` 0, `ymuroya47` 18 kernels but none on ARC-AGI-3. The highest-scoring *published* method
in the competition is Tufa Labs' duck harness at **1.62** — a full point below Franzen.

⚠️ **One hallucination to blacklist:** a web summarizer confidently attributed
`github.com/arodmor/arc-agi-3` ("JEPA world-model") to Franzen. **That repo is not his.** If a JEPA
attribution to Franzen appears in another agent's output, it is fabricated. (Our own record already
has JEPA dead on this benchmark, 3 strikes.)

### 1.2 What IS well-evidenced — and it is a better argument than the one in the order [V]

Direct diff of the public leaderboard CSV, 2026-08-13T01:46Z snapshot (`runs/harness_diff_0813/`)
against a fresh pull at 2026-08-15T14:42Z. `SubmissionCount` is a column in that CSV.

```
  1 cstl                     2.70 (was 2.52)  subs  25 (was 23)   +0.18 in 2
  2 Daniel Franzen           2.58 (was 1.24)  subs  41 (was 38)   +1.34 in 3
  3 Nikita Sorokin           2.10 (was 1.33)  subs   6 (was  5)   +0.77 in 1
  4 Yusaku Muroya            1.98 (was 1.25)  subs  71 (was 70)   +0.73 in 1
  5 AbeLincoln1865           1.90 (was 1.07)  subs   7 (was  5)   +0.83 in 2
  6 YUTO KOJIMA              1.86 (was 1.86)  subs  69 (was 66)    0.00 in 3
  7 MLRush                   1.75 (was 1.46)  subs  49 (was 45)   +0.29 in 4
 15 Tufa Labs                1.62 (was 1.62)  subs 107 (was 104)   0.00 in 3
119 Canivel (us)             1.33 (was 1.33)  subs 111 (was 108)   0.00 in 3
```

Field-wide: **344 teams submitted, 113 improved. Teams ≥1.70 went 2 → 7. Teams ≥1.60 went 12 → 20.**

**Three things this settles, and one it does not.**

1. **No rescore occurred.** Seven-plus teams' scores are byte-identical across the two snapshots
   (KOJIMA 1.86, Andy liu 1.69, Tufa 1.62, Helmut 1.61, FOYSAL 1.61, Van-Phuc 1.61, DhanaLakshmi
   1.60). A global scoring change would have moved those. Our own 08-14/08-15 draws were **0.70 and
   0.89** on a frozen fork — no rescale reached us either. **The jumps are real submissions.** [V]

2. **The top jumps are NOT draw-count order statistics, and this is the one place the order's
   instinct is right for a reason it did not state.** On our own measured per-draw process
   (`runs/ledger.json`: n=32, μ=0.9353, σ=0.1533), 1.25→1.98 is **+4.8σ** and 1.24→2.58 is
   **+8.7σ**, each in a *single* additional draw. The 08-13 harness-diff established that the entire
   1.44–1.62 band is order statistics on the stock harness (P(draw ≥1.4) ≈ 0.016). **The 1.75–2.70
   band is not.** Something real is happening above 1.70. [C]

3. **Most of the "explosion" in the movers list is routine onboarding, not re-basing.** Of the top 25
   movers, the majority are teams going 0.00–0.33 → 0.9–1.3 in one submission — the signature of
   first-time forks of the public duck harness. Reporting "113 teams improved" as a re-basing
   overstates it. The re-basing is **five teams**, at the top. [V]

4. **NOT ESTABLISHED: whether the five top movers share a common cause.** No top mover has a public
   kernel. I could not find a newly-published high-scoring notebook in the window that would explain
   1.90/1.98/2.10 landing within 24h of each other. Three independent teams arriving in the same new
   band in one draw each is suggestive of a shared artifact, but **I have no evidence for it and I am
   not going to reason my way to one.** [UNK]

**Ruling.** "The top regime is TTT/fine-tuning" is not supported. "The top regime is *something we
do not have*, and it is not draw count" **is** supported, by the σ arithmetic above — and that
statement is method-agnostic. The panel should adopt the second and discard the first.

### 1.3 Ranking note the order should absorb

We did not "stay static while the field re-based." **We fell 91 → 119** while making 3 submissions
that drew 0.78 / 0.70 / 0.89 — the low end of our own distribution. Our trailing-4 is **0.86**,
down from 0.91. The filler lottery is drawing badly *and* the bar moved. Both, not one.

---

## 2. Q1 — THE CRUX: does the recipe transfer?

### 2.1 The benchmark makes the question answerable, and the answer is mostly no [V]

ARC-AGI-3 (arXiv:2603.24621, ARC Prize Foundation): 64×64 grid frames, `RESET` + `ACTION1–6`
(ACTION6 carries an (x,y) click), system prompt literally *"Your goal is to win."* **No
instructions, no demonstrations, no train/test pairs.** Each environment is encountered **once**;
there are no repeated attempts and no carryover between environments. Scoring is RHAE:
per level `min(1.0, h/a)²` against the second-best human baseline, later levels weighted up to 5×.
**The square is an anti-brute-force term: 2× human actions scores 0.25, not 0.5.**

Component-by-component:

| ARChitect component | Its ARC-AGI-1 role | ARC-AGI-3 counterpart | Verdict |
|---|---|---|---|
| **Offline fine-tune** on Re-ARC / ConceptARC / ARC-Heavy | learn the transduction prior | **the 165 non-public `re-arc-3` families in Tufa's own bundle.** Real, present, already split 80/20 in `duck_eval/lora/splits.py` | **TRANSFERS.** This is our LoRA lane, verbatim. |
| **TTT on the test task's demo pairs** | adapt to *this* puzzle | **nothing plays this role.** There are no demonstration pairs. First contact, one episode, no repeats. | **DOES NOT TRANSFER.** |
| **Augmented inference** (D8 × colour perm × example shuffle, 16 views) | 16 independent shots at one grid | D8 applies to the *frame*, but the action semantics do not follow for free — under a rot90 the harness still calls ACTION1 "UP", and whether the game's controls are screen-relative is exactly what the agent is trying to learn. ACTION6 coordinates do transform cleanly. | **PARTIAL, UNPROVEN.** |
| **Candidate selection by product of augmented probabilities** | rank ≤20 grids, emit 2 | **you cannot generate 20 action sequences and pick one after seeing which is right — executing a candidate IS the cost, and RHAE charges it quadratically.** | **DOES NOT TRANSFER in this form.** |

### 2.2 The killer detail is in ARChitect's own ablation

Table 1 of the paper, Nemo-mix, deltas: baseline 26.0 → **TTT +14.5** → **augmented candidate
generation +17.0** → **augmented log-prob scoring +11.5** → **DFS +3.5** = 72.5. [V]

**Roughly two-thirds of the win is inference-time machinery, not the fine-tune — and the two-thirds
that dominates is precisely the part that dies when the output becomes an action sequence.** Even
granting the principal's premise entirely, **the transferable fraction is the smaller fraction.**

The paper contains **zero** occurrences of `agent | interactive | game | environment | reinforce`
across all 18 pages and both notebooks. It was written Nov–Dec 2024; ARC-AGI-2 had not launched and
ARC-AGI-3 did not exist. [V]

One further correction to a common belief: **ARChitect's TTT is not per-task.** They train **one
shared adapter over the demonstration pairs of all 100 hidden test tasks at once**, split across two
T4s. Per-task adapters were tried and *discarded* as not runtime-efficient. So even the "TTT" label
means something structurally unavailable to us: it requires the whole test set in hand, offline,
before the run. [V]

### 2.3 Nobody has published TTT on ARC-AGI-3 [V]

Full arXiv sweep, `all:"ARC-AGI-3"`, 16 papers Dec 2025 – Aug 2026. **Not one performs
inference-time weight updates.** Tycho (2607.28287, 100 RHAE), Rodionov (2607.15439, ~99%), NOOA
(2607.20709, 85.1%), OPINE (2607.01531, 78.4%), PRO-LONG (2607.20064), Rodionov EWM (2605.05138,
58.12%), DreamTeam (2605.09650, 38.4%), Rudakov (2512.24156, "training-free"), MAP (2605.13037 —
audited separately today, do not re-audit). Our own three-sweeps-old category verdict stands
(`learnings/sweep_research_2026-08-15.md` FIELD 5): *"everything in TTT requires gradient updates at
inference or a development-set profile, and neither is available."*

Two papers name the question and answer it in the negative:
- **DreamTeam (2605.09650), verbatim:** *"Fine-tuning is too slow for an online loop in which the
  agent must adapt within a single episode."*
- **Sensi (2603.17683)** is literally titled *"Curriculum-Based Test-Time Learning for LLM Game
  Agents"* and is **not gradient-based**: *"the model's parameters remain frozen."*

### 2.4 What HAS worked with gradients on ARC-AGI-3 — and we already own it

**StochasticGoose, 1st in the ARC-AGI-3 Preview, 12.58% RHAE:** a 4-layer CNN over the 64×64 frame,
trained online with BCE on `(state, action) → frame_changed` from self-generated transitions,
**reset between levels**. That is the honest interactive analogue of TTT, and nobody calls it TTT.
**Blind Squirrel (2nd, 6.71%)** retrains a ResNet18 value model on hindsight-relabelled trajectories.
[V]

**This lineage is ours already.** `ITERATION_LOG.md` line 136: our v35 is that exact architecture
("16-channel one-hot 64×64 → 4-layer CNN → action head + fully-conv coord head, BCE on
(state,action)→frame_changed, reset model + buffer on level-up"). It scored **0.18**. The one form
of test-time gradient learning demonstrated to work on this benchmark is a form we have run, and it
is 7× below the stock LLM harness.

**Shape of the gap in the literature:** everyone who does gradients on ARC-AGI-3 does it on a *small
net trained from scratch*; everyone who uses an LLM keeps it **frozen** and moves the learning into
text, code, or memory. **Nobody has published the crossbar — gradient adaptation of the LLM itself
during an ARC-AGI-3 episode.** That is either the opportunity or the reason it isn't done. [UNK]

### 2.5 The constructive finding — where ARChitect's *biggest* component actually maps

The +32 of ARChitect's +46.5 comes from *generate many candidates, then select without executing
them*. In an interactive benchmark the only way to score a candidate without paying for it is to
roll it out **against a model of the environment instead of the environment**.

That is exactly Rodionov / OPINE / Tycho — and **it is a lane already in this repo**:
`learnings/war_room/opine_world_deepread.md` §"highest-value adaptation" — plan–execute–verify over
our **12 saturated `exec_wm` sims**, harness-side BFS, one live action per step, hash-compare against
the settled frame, fail-closed. Zero LLM tokens, ~2–3 build days, expectation +0.10–0.30.

**If the panel wants the ARChitect principle, that is where it lives on this benchmark — not in
augmented decoding.**

---

## 3. Q3 — THE ACTUAL RECIPE (for costing, since the panel will ask)

Paper: *The LLM ARChitect: Solving ARC-AGI Is A Matter of Perspective*, Franzen, Disselhoff,
Hartmann. **Not on arXiv** — `https://da-fr.github.io/arc-prize-2024/the_architects.pdf`. Code:
`github.com/da-fr/arc-prize-2024`. Result: **53.5 on ARC-AGI-1 private, 1st place ARC Prize 2024**
(2nd was 40). [V]

- **Base:** `nvidia/Mistral-NeMo-Minitron-8B-Base`, chosen under Kaggle's 2×T4 16 GB limit.
- **Vocabulary surgery** (cheap and under-appreciated): tokenizer + embeddings cut from ~120,000 to
  **64 tokens** — one token per cell, no delimiters, no digit merges — plus a learned ~48-char
  "pre-prompt" scratch prefix.
- **Stage-1 offline FT:** LoRA **r=256, alpha=24** (rsLoRA), 4-bit base, unsloth; targets
  `q,k,v,o,gate,up,down_proj` **+ `embed_tokens` + `lm_head`**; LR 1e-4 adapters / 1e-5 embeddings;
  seq 8192; loss masked so it never predicts an input grid or the first output. Data: **531,318
  examples** (257,600 Re-ARC + 51,200 public-eval + 22,528 ConceptARC + 200,000 ARC-Heavy). Hardware
  single H100. **GPU-hours not stated in the 2024 paper; the 2025 successor quotes 98 H100-hours for
  a comparable stage-1** (~$200–250 spot). The public *training* set is discarded — Re-ARC supersedes it.
- **TTT (in-notebook):** one shared LoRA **r=64, alpha=16** over the demo pairs of all 100 test tasks
  at once, split across 2 T4s. **~5h20m of the 12h budget.** ⚠️ Table 4 says 32 epochs; **both
  released notebooks set 4 augmented passes** — trust the notebook, it is the artifact that scored 53.5.
- **Augmented inference:** **16 augmentations per task** = transpose × 4 rotations × 2 random
  colour/example-order draws. Un-augmentation walks the augmentation key backwards.
- **Candidate generation:** **DFS over the token tree under a probability floor** (`min_prob = 0.17`
  in the winning notebook), KV cache truncated on backtrack so only one branch is materialised.
  Guarantees *all* candidates above `p`. **Faster than greedy** (2:35 vs 3:49 per 100 tasks on H100).
- **Selection:** `argmax_k Π_i P(T_i(S_k) | T_i(C))` over 8 scoring augmentations — the generative
  model used as a classifier, ranking by **stability of likelihood under augmentation**. Worth **+25%
  over baseline for 10–17 min per 100 tasks**. Implementation detail worth stealing: a `+3` log
  offset (≈5% probability floor) so candidates surviving more augmentations aren't penalised.

---

## 4. Q4 — WHAT IT COSTS US, UNDER `feedback_arc_zero_budget`

### 4.1 The transferable half is already built and already costed

`learnings/war_room/lora_lane_2026-08-13.md`. **Weekly cost of stages S+T+2+3+4: ~7–10 GPU-h, inside
the free 30 GPU-h allowance. No cloud spend at any stage.** [V]

| stage | what | cost | status |
|---|---|---|---|
| 1 | trajectory generator + split + first dataset | CPU | ✅ **DONE** — 127 SFT examples, 33 non-public families, `selftest.py` 17/0 |
| 1c | hand-rolled LoRA / chunked CE / meta-init loader, validated vs reference on the 3080 | local GPU, free | ✅ **DONE** — 30 checks, 0 failing |
| **S** | LoRA serve canary, two random r16 adapters, `--enable-lora` the only delta | ~1 GPU-h | ⛔ **v1 ERRORed on a NameError in the agent's own guard. v2 built, NOT pushed.** |
| T | load+train smoke on jcole75's wheelhouse | ~1 GPU-h | not started |
| 2 | train v0 adapter, scope A / r16 | 2–5 GPU-h | not started |
| 3 | **H1 DEV transfer, 36 unseen non-public families** | ~1 GPU-h | not started |
| 4 | H2 EVAL A/B, adapter vs base | ~2 GPU-h | not started |

The training corpus is not synthetic hand-waving: `duck_eval/lora/teacher.py` subclasses the
harness's **own** `ToolAgent` and overrides exactly one method, so every prompt byte is produced by
the code that runs in the scored kernel. `splits.py` enforces a binding rule — **a family scored on
the leaderboard is never a training source.** That is `feedback_arc_generalization_first` compiled
into code.

**So: yes, a version of this recipe is executable for us, at zero cloud spend. It is not blocked on
money or GPU. It is blocked on our own execution.** Honest tally already in the log: **~4 slots and
~3 GPU-h since 08-13 for ZERO measurements on either model-level lane.**

### 4.2 The non-transferable half is unaffordable even if the data existed

- ARChitect's TTT costs **5h20m of a 12h budget**. Our run has a 9h wall and the analyzer is already
  the bottleneck (`max_runtime_s_per_game=7920`, `concurrency=28`).
- **No `peft`, no `accelerate`, no `bitsandbytes` in either wheelhouse** — LoRA, the loader, and
  chunked cross-entropy are all hand-rolled (that is why stage 1c exists).
- **The duck's own pinned wheelhouse cannot train this model at all** (transformers 4.57.6 vs the
  checkpoint's required 5.6.2). Training must move to jcole75's wheelhouse while serving stays
  byte-identical to the 1.62 config.
- Chunked CE is mandatory: 248,320 vocab ⇒ **48.8 GB of logits at 32k** context.

**Recommendation: do not build in-episode TTT. It has no data source, no published precedent on this
benchmark, no wall-clock budget, and two papers explicitly rejecting it for this setting.**

### 4.3 Cheapest informative fragments, ranked, all at zero GPU and zero slot

**#1 — Daily leaderboard snapshot + diff. Cost: one API call. Build: <1 hour.**
The single most decision-relevant fact in this entire investigation — Franzen has **41** submissions
and moved 1.24→2.58 in **three** — came from a CSV column (`SubmissionCount`) that is free, public,
and which we apparently only read when someone runs a census by hand. `runs/lb_daily/` **exists and
is empty.** Snapshot it nightly, diff it, and the morning check reports *movement* instead of
*levels*. This would have caught the top's move on 08-14 rather than 08-15, and it is the instrument
that turns "the top exploded" from a vibe into a table. Given that `feedback_audit_the_instrument`
has now fired three times in four days, **the cheapest informative test available to us is another
instrument, not another arm.**

**#2 — Stage 1b + H1, CPU-only.** The lane's own discriminator (H1: does the adapter transfer to 36
families it has never seen; H1a tail *and* H1b mass, both required) is the gate that kills the lane
before a single slot if the corpus teaches style rather than policy. Stage 1b — scaling from 127
examples to all 129 train + 36 dev families, plus the chat-template round-trip check — is **CPU-only
and unbuilt.** Do it before spending anything.

**#3 — Finish stage S. One slot, ~1 GPU-h.** It tests a link **no competitor has ever been observed
completing** (auxentr specified a LoRA adapter and never ran one), and its sealed decisive read
already exists: `noop ≡ base AND probe ≠ base ⇒ PASS`; `probe ≡ base ⇒ the adapter is being silently
ignored` — the failure that would otherwise read as "LoRA didn't help" *after* a full training run.

**#4 — free, dated, zero-effort: wait for 2026-09-30.** Milestone #2 forces open-sourcing to claim
prize money. Franzen is in the $10K seat and cstl in the $25K seat. **Six weeks buys us the actual
method, for nothing.** That is not a reason to idle, but it is a reason not to spend slots
reverse-engineering an inference.

---

## 5. Q5 — RECONCILING WITH THE "OUR AGENT FORGETS" ROOT CAUSE

**Position: the forgetting diagnosis is not "real but secondary." It was tested, twice, with
pre-registered controls, and it is REFUTED. The order is quoting a superseded root cause.**

The evidence is ours, not borrowed:

- **P1 mechanism C (08-12).** A deterministic memo naming confirmed-dead actions was delivered on
  **1,463/1,519 turns = 96.3%**, across all 25 games, 215 blocks naming confirmed-dead state. The
  arm's dead-reissue rate looked like a 4.4× win — and reconstruction against block-free controls
  showed it was **regression to the mean** (the arm's 7.8% sits *inside* the 5.3–23.1% control
  spread and *above* the best control). Verbatim from `ITERATION_LOG.md`:
  > **"The block lands on 96% of turns and the agent does not act on it. The agent re-explores WHILE
  > HOLDING THE GROUND TRUTH, unprompted, every single turn. This kills the memory/forgetting
  > hypothesis outright. It is a CAPABILITY limit, not an architecture limit."**
- **EFFNOTE (08-13).** Independent second test, different mechanism (quantified per-turn efficiency
  note, the scoring rule stated verbatim + live action count). **93.8% delivery, also inside the
  control spread.**
- **RedundancyBench (arXiv:2605.29893).** The best LLM-based redundancy detector scores **24.88%**,
  some below random. That is the standing rebuttal to every "let the model notice it's repeating"
  proposal, including the pull-side variant.
- The lane doc's own §0: *"INFORM was refuted twice. CONSTRAIN was refuted. There is nothing left to
  copy and nothing left to tell the model. What is left is the weights."*

**So the two are not the same problem from different ends, and they are not both live.** The
sequence is: forgetting was the hypothesis → it was instrumented → the mechanism delivered at 96%
and the behaviour did not change → the hypothesis died → the weights hypothesis replaced it. **The
weights hypothesis does not need Franzen's 2.58 to justify it and should not be re-justified by it.**

### 5.1 But the weights hypothesis is weaker than the lane doc admits, and the panel must hear this

The 08-13 field census refuted a *different* claim in the opposite direction: **`Qwen3.6-27B-FP8`
serves 99 kernels across 52 teams spanning 0.00 → 1.62. The model explains NONE of the public
variance.** That sits in direct tension with "what is left is the weights."

**Reconciliation, and it is uncomfortable:** the weights explain none of the variance *up to 1.62* —
and 1.62 is the entire published ceiling. Above 1.62, **nothing public explains anything at all.**
So "it's the weights" is a claim about a regime in which, by construction, we have no evidence
either way. **The LoRA lane is not evidence-backed. It is the last untried lane.** That is a real
justification — untried lanes are worth more than refuted ones — but it is a *weaker* one, and the
panel should adopt it under its true name rather than under a borrowed 2024 paper award.

---

## 6. PANEL PROPOSAL — accept or reject

**CLAIM.** The ARChitect recipe does not transfer to ARC-AGI-3 as published. Its offline-fine-tune
component maps cleanly onto a lane we have already designed, costed and half-built; its TTT and
candidate-selection components have no counterpart in an interactive benchmark, and those are the
components that supply two-thirds of its measured gain. Franzen's 2.58 is **not evidence for TTT**;
it is evidence that something above 1.70 exists which is not draw count. The correct response is to
**finish the lane we already own** and to **fix the instrument that made us 24 hours late to this**,
not to start a TTT build.

**EVIDENCE.** ARChitect ablation (TTT +14.5 of +46.5). ARC-AGI-3 interface: no demonstration pairs,
single contact, quadratic RHAE. 16/16 arXiv ARC-AGI-3 papers frozen-weight; two explicitly reject
gradient TTT for the online loop. Zero public disclosure from any top-10 team. Leaderboard CSV diff:
Franzen 41 subs, 1.24→2.58 in 3; no rescore; +4.8σ to +8.7σ single-draw jumps at the top; routine
onboarding below. Our own LoRA lane: stage 1 + 1c done, 7–10 GPU-h total, zero cloud spend.

**COUNTER-EVIDENCE (against my own claim, stated fairly).**
(a) The +8.7σ jump is real and unexplained; TTT remains *a* live candidate among many, and I cannot
rule it out. (b) The one crossbar nobody has published — gradient adaptation of the LLM itself
during an episode — is unpublished, which is compatible with "it doesn't work" *and* with "it's the
edge and nobody's told you." (c) Three teams landing in 1.90–2.10 within 24h smells like a shared
artifact I failed to find. (d) Our own forgetting-refutation rests on two arms that both had other
defects in the same weeks, and `feedback_audit_the_instrument` has fired three times in four days.

**TRANSFERABILITY VERDICT. DOES-NOT-TRANSFER** for TTT + augmented-inference + candidate-selection.
**TRANSFERS** for offline fine-tuning on the non-public re-arc-3 families — which is not a new lane
and carries no evidence from Franzen. **NOT ESTABLISHED** for what the top 7 actually run.

**CHEAPEST INFORMATIVE TEST.** Nightly leaderboard snapshot + diff into `runs/lb_daily/` (one API
call, zero GPU, zero slot) — then stage 1b + H1 on CPU — then stage S on one slot. Nothing else
until H1 reads.

**WHAT WOULD FALSIFY THIS.**
1. **2026-09-30 Milestone #2** forces Franzen and cstl to open-source to claim $10K/$25K. If either
   discloses TTT or in-episode fine-tuning, this document is wrong and the panel should say so. Free,
   dated, six weeks out.
2. **H1 flat on the 36 unseen dev families** ⇒ the SFT corpus teaches style, not policy ⇒ the
   transferable half dies too, on CPU, before a slot.
3. **Stage S returns `probe ≡ base`** ⇒ adapters are silently ignored on this rail ⇒ the lane is dead
   for infra reasons regardless of the science.
4. **A public kernel appears in the 1.90–2.10 band running the stock brain with prompt/harness
   changes only** ⇒ "what is left is the weights" dies, and the harness lane reopens.
5. **Any published result showing in-episode LLM weight updates beating a frozen LLM on ARC-AGI-3**
   ⇒ §2.3's negative is stale and Q1 must be re-run.

---

### Appendix — what I could not establish

| Question | Status |
|---|---|
| Does Franzen's 2.58 use TTT? | **NOT ESTABLISHED** — no direct evidence of any kind exists. |
| Do the five top movers share a common cause? | **NOT ESTABLISHED** — no top mover has a public kernel; I found no candidate artifact. |
| Franzen's per-submission score history | **NOT AVAILABLE** — the public LB exposes max score + last-submission date only. Two snapshots, 08-13 and 08-15, is all we have; **this is exactly what a daily snapshot would fix.** |
| Whether any competitor has ever served a LoRA adapter in this competition | **NO — never observed.** auxentr specified one and never ran it. That is why stage S is worth a slot. |
| Whether TTT is disallowed by competition rules | **NO, and a secondary blog claiming otherwise (labs.adaline.ai) is unreliable** — not in the ARC Prize Verified Testing Policy; TTT is how ARChitect won ARC Prize 2024 on Kaggle. |
