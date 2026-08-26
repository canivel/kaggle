# Intel Sweep — 2026-08-11 (DELTAS ONLY vs `intel_sweep_2026-08-04.md`)

**Baseline:** `learnings/war_room/intel_sweep_2026-08-04.md`. Also deduped against the internal
sweeps that ran between the two: `learnings/sweeps/discussions_sweep_2026-08-05.md`,
`discussions_2026-08-07.md`, `discussions_2026-08-09.md`, `research_2026-08-07/08/09/10.md`.
Items already in those files are marked **(known internally)** and are not re-argued.

**Method:** Kaggle CLI — full public LB CSV (2,229 teams, pulled 11:24Z), kernel list (100 rows,
`dateRun`), dataset list (`duck-harness` / `taaf` search, sorted by update), 4 notebook pulls,
2 competitor source-bundle pulls; `gh` API (repo search `pushed:>=2026-08-04`, README reads);
arXiv API (4 queries) + WebSearch/WebFetch (arcprize.org blog, llm-releases, HF).
Read-only throughout. **Zero pushes, zero submissions, zero spend.**

**Coverage gap, stated plainly:** the **Kaggle discussion feed was NOT read directly today.**
The chrome-devtools MCP browser profile was locked for the entire session
(`browser is already running for …/chrome-profile`, ~6 attempts), and every non-browser route to
the forum is blocked — `kaggle forums topics list` returns **403**, WebFetch returns only the SPA
shell, and the internal `discussions.DiscussionApiService` endpoint 404s without a session cookie.
Forum coverage therefore stands at **08-09** (`discussions_2026-08-09.md`, newest topic **733865**);
**08-10 and 08-11 forum activity is unswept.** Compensating coverage: the Kaggle **kernels and
datasets APIs** were swept exhaustively instead — and that is where both ADOPT-tier findings came
from, so the gap cost us less than it looks. **Carry to tomorrow: re-run the forum leg first.**

**Our state:** **1.33 @ #84** (was #70 on 08-04 — again pure drift, banked draw untouched).
Head KOJIMA 1.86 (unchanged, still resubmitting daily). Gold line **1.58**. Gap **0.25**.

---

## RANKED FINDINGS

### 1. THE DUCK HARNESS THROWS AWAY HALF THE OBSERVATION — VERIFIED IN OUR OWN FORK — **ADOPT (highest impact on the board)**

**Source:** `kaggle.com/datasets/jakobbrggen/taaf-kaggle-source-anim-20260807-anim` — a **public**
TAAF source bundle uploaded 2026-08-07 by **jakobbrggen**, who is a member of **Helmut AGI, #8 @ 1.61**
(LB `TeamMemberUserNames` = `antonijankowski123, jakobbrggen, nightmareblocks`). The bundle's
`git_status.txt` shows branch **`feature/animation-awareness`**, and `preamble.txt` shows
`hard_noop_guard=True, animation_awareness=True`, `passes: 4, games: 6` (a local A/B bundle).
Two new modules that do **not** exist in our frozen fork: `inference/utils/animation.py` (15,491 B)
and `inference/agent/noop_guard.py` (3,970 B). Both read in full.

**The measured claim, quoted from their module docstring:**
> `arcengine` renders a frame after every internal `step()`, so one action can come back as a short
> animation. TAAF exposes the whole list (`GameState.all_frames`), but the harness only ever consumed
> `raw.frame[-1]` — so every intermediate frame was discarded before the agent could see it.
> Measured over 24 games (12 multi-frame responses each), **13 games return multi-frame responses**,
> in two distinct shapes:
> - **type 1** (`ft09`, `sb26`): first and last frame are **identical**, all information — a rejected
>   click, a consumed attempt — **lives only in between**.
> - **type 2** (`r11l`, `sk48`): pure motion interpolation, carries nothing the final frame does not.

**I verified this against our own artifact, at zero cost, and it holds.**
`duck_eval/taaf_bundle/src/tufa-arc-agi-framework/src/taaf/game.py:170` defines
`frame → Frame(data=self.raw.frame[-1])`, and lines 172–180 define `animation_frames`
(`raw.frame[:-1]`) and `all_frames` (`raw.frame`). A grep for `all_frames` across
`duck_eval/taaf_bundle/src/ARC3-Inference/` returns **zero consumers**. **Our agent has never seen
an intermediate frame.** On type-1 games the agent is shown a board byte-identical to the one before
its action and must conclude "no effect" — the exact false-no-op / state-aliasing failure class our
own project memory names as *"state-aliasing = one root cause"*.

**Their three-stage design (all deterministic, all harness-side, all token-budgeted):**
1. **Per-action metadata** (`summarize_animation`): `frames`, `unique_frames`, `board_unchanged`,
   `transient_pixels`, `transient_bbox` — a few dozen tokens, and **`None` for the single-frame case**
   so ordinary actions cost nothing.
2. **An `animation()` retrieval tool** returning a **diff timeline**, never raw frames — their stated
   arithmetic: one 64×64 ASCII grid ≈ 1,400–2,000 tokens vs a ~1,024-token tool budget, and `sb26`
   returns **up to 42 frames for one action** (60–80k tokens raw, 24–34k deduplicated). Budgets:
   `MAX_STEPS=8`, `MAX_CELLS_PER_STEP=24`, `MAX_TOTAL_CELLS=80`, optional single-frame verbatim read
   cropped to the transient bbox. **The tool executes nothing and spends no action budget.**
3. **A proactive hint**, fired only on `turns_without_progress ≥ 6` **and** `transient_animations ≥ 2`,
   cooldown 6 — i.e. only on type-1 games where reading the final frame *cannot* work.

**Verdict: ADOPT.** This is a **perception bug in the artifact we ship**, not a strategy idea. It is
pure Python, needs no model change, no training, no new dataset, no cloud spend, and is
warpack-monkeypatchable. It is independently verifiable on the free build rail against our own
25 offline environment files (count multi-frame responses per game — no LM needed).
**Expected impact on the 0.25 gap:** the largest of any item found this window. It restores signal on
roughly half the game population, and the two named type-1 games include **`ft09`, which every public
per-game table (ours, Reki's, borro1980's) identifies as the fat head of the score distribution**.
Unquantified, but this is the first mechanism in weeks that is *cheap, verified, and aimed at a
known root cause*. **Caveat that must ride with it:** Helmut AGI's 1.61 is **not** attributable to
these two features — the bundle is a 6-game/4-pass local A/B, the team appeared at 1.61 on 08-09, and
no ablation is published. Treat as **mechanism evidence, not efficacy evidence**.
*(Route to: main, exec-reset-day1.)*

### 2. HARD NO-OP GUARD, WITH THE ANIMATION EXEMPTION — **ADOPT (same bundle, free)**

`noop_guard.py` blocks re-execution of a `(level, board_before_sig, action_sig)` triple already proven
to have no effect **before it reaches the environment**. Their own docstring is the finding:

> Experiment 1 (K1 auto-memory, reverted) only *mentioned* known no-ops in the injected context — the
> model was free to ignore that and repeat them anyway (**~12% no-op repeats remained**). This module
> instead lets the harness actively block…

Two design details we do not have:
- **State-keyed, not batch-local.** Our (c)/(d) suppression and yw8837's guard are *within-batch
  repeated-no-effect* rules; this is a persistent per-level `blake2b` board-signature index
  (bounded: 512 states/level × 16 actions/state, LRU-evicted).
- **The animation exemption, and it is the load-bearing part.** `observe(..., animated=True)` — an
  action that returned >1 frame is **never** recorded as a no-op even when the board is identical,
  and any contradicting evidence **deletes** a previously recorded no-op. Their comment: recording
  those as no-ops "made the guard hard-block actions that had clearly worked, on exactly the games
  with the most animations."

**Verdict: ADOPT, but strictly sequenced *after* finding 1** — without animation detection the guard
is actively harmful on the type-1 games, which is a pre-registered harm mechanism, not a guess.
**Impact:** ~12% of actions recovered is directly a score term, since `level_score = (human/ai)²`.
*(Route to: main, exec-reset-day1.)*

### 3. RETRODICT — NEW PUBLIC SOTA (99.86 RHAE), AND IT RE-AIMS OUR PORT — **ADAPT (high)**

`github.com/ryanbbrown/Retrodict` (created 07-06, **pushed 08-10**, 20 stars, **NO LICENSE**).
**Not in any internal file.** Official competition-mode scorecard: **99.86% mean RHAE, all 25 public
games, every level**, 7,703 actions, 0.66B tokens, **$654**, `gpt-5.6-sol` at `max`. It beats
`astroseger/arc-3-agents-baseline1` (98.97%, the previous public best) with **5.5× fewer tokens**
(0.66B vs 3.64B). Built on `ryanbbrown/thinharness` (MIT); the log-as-context + plan-queue foundation
comes from `alexisfox7/PRO-LONG` **(known internally, arXiv:2607.20064)**.

**Why this matters more than Tycho did:** re-read `port_rescope_2026-08-10.md` §1 next to it. Tycho's
entire measured lift sits in C8/C9/C12 — *an LLM authoring and repairing `world_model.py` in-loop*,
plus frontier backends — all three OUT of our scope, which is exactly R25-N2's "defanged test".
**Retrodict's lift is somewhere else, and that somewhere else is affordable:**
- **The runner, not the model, does the verification.** Each action in an `[ACTIONS]` plan carries the
  exact cells the settled board must show; the runner plays the queue one action per step, checks
  after every step, **halts on the first mismatch and re-invokes the model with the diff**. This is
  deterministic, harness-side, and costs zero LM tokens. It is the mechanism behind the 5.5× token
  reduction — and token efficiency is *our* binding constraint, since a 27B on one GPU inside a 9h
  wall is LM-call-bound, not intelligence-bound.
- **Retrodiction ≠ world-model authoring.** The model writes small python that replays *one hypothesis*
  over `log.txt`; a contradicted hypothesis is falsified for free. It never has to emit a full
  `transition()`. **This is a much smaller authoring task than the one our L0 falsifier just returned
  0/13 carriers on.**
- **The executable simulator is an ESCALATION, not the loop.** Only after 300 actions stuck on one
  level does a binding directive tell the agent to promote checked rules into `step(state, action)`
  and verify it retrodicts every recorded frame. This is the same shape as Tycho's ablation
  (actor-controlled builder 88.49 > auto-triggered 83.07) — **and it says our always-on EWM framing
  was the wrong one twice over.**
- **Context resets keep files, not summaries.** At 150k input tokens the conversation is dropped
  entirely; only `playbook.md` (curated working model + working memory) and `log.txt` survive. Note
  this is **not** the eviction-pressure mechanism that killed A22 — nothing is compacted; the channel
  is externalized and the transcript is thrown away whole.
- **Vision is ONE call.** A separate vision model reads the *opening board only*, and its answer is
  injected as a hypothesis to verify; the agent is **never shown an image again**. Everything else
  arrives through `log.txt`. (See finding 6 — this is what de-rates the GLM-4.6V screen.)
- **Deterministic perception helpers.** A bundled `arclog` library gives the python tool one-call
  parsed boards, per-step `[DIFF]` lines, and connected-component objects — object-centric priors
  computed in code, not re-derived by the model every turn.

**Verdict: ADAPT (ideas only — the repo carries no license, so nothing is copyable into an
open-source-required submission; `thinharness` and `baseline1` are MIT if a reference is needed).**
**Expected impact:** this is the strongest argument on the board for **re-scoping the port away from
Tycho's C1–C7 substrate and toward the runner-side expectation checker + plan queue.** It converts
R25-N2 from "the port is defanged and the fix is unaffordable (3.5× wall, 52× LM calls)" into
"a different architecture puts the lift in the deterministic half, which we can afford."
*(Route to: main, exec-reset-day1, review-r25-opus5.)*

### 4. NOSUMINA — AN INDEPENDENT NEGATIVE ON EXACTLY OUR LANE, AT EXACTLY OUR SCALE — **ADAPT (evidence)**

`github.com/russellmsilva/Nosumina` (MIT, pushed 08-09). **Not in any internal file.** Explicitly an
entry for "the local-LLM-only track"; thesis stated as *"how you scaffold a model matters more than
how big it is"*. A **local quantized Qwen3-Coder-Next** writes and revises a `GameModel.predict()`
class against recorded traces, certified by full sequential replay over every transition seen so far,
with counterexample-driven revision and a curriculum over trace chunks — i.e. **Schema/Tycho, ported
down to a local model. This is our lane (a), built by someone else, and reported honestly.**

**The result is negative:**
> An earlier, stateless harness scored **0.8% exact-match / 5.5% changed-cell accuracy** on held-out
> `ls20` data — **statistically indistinguishable from the ~6.25% random baseline** … **Near-0%
> exact-match on predicting the next grid is currently the state of the architecture even under that
> redesign** … the model characterizing noise rather than forming a real concept of what it's looking
> at. Whether that's a fixable harness gap or a real ceiling in the local model is still the open
> question.

He reaches for the same fix we would: an `analyze()` preprocessing pass that offloads
"same shape, moved?" onto a deterministic flood-fill component extractor rather than asking the model
to re-derive it from raw diffs.

**Verdict: ADAPT as evidence.** This is the **first external, same-constraint replication of our L0
falsifier's 0/13 carriers**. Two independent teams now report that a local ~27–80B model cannot author
a faithful executable transition model of these games. **It does not kill state-externalization — it
kills the *model-writes-the-world-model* variant of it**, and it points at the same escape hatch
Retrodict took (deterministic perception + small hypothesis checks instead of a full simulator).
**Impact on the gap:** negative-information value — it should stop us spending the remaining 83 days
re-deriving a known-negative, and it raises the prior on findings 1/3 accordingly.
*(Route to: main, exec-reset-day1, review-r25-opus5.)*

### 5. SOMEONE ELSE IS RUNNING OUR LANE-(a) EXPERIMENT, ON OUR SUBSTRATE, FOR FREE — **MONITOR (high value, zero cost)**

Two competitors have ported **PRO-LONG-style programmatic memory onto the Duck harness at
Qwen3.6-27B** and published the source:
- **`iseesmth/duck-harness-prolong-source-20260811`** — published **today 07:21Z**, notebook
  `iseesmth/duck-harness-prolong-public-eval`. New module `inference/agent/programmatic_memory.py`
  (4,907 B), which I pulled and read. Docstring: *"Lossless, append-only game memory for programmatic
  retrieval. The model never receives this file in its prompt. Instead the sandbox exposes a
  read-only `memory` object so the model can search and parse the complete trajectory with Python
  without consuming active-context tokens."* It appends the initial board, every model reasoning
  block **before context trimming can discard it**, every tool call, every tool result, and every
  action outcome. `tool_agent.py` is 90,990 B vs the anim bundle's 108,927 B — a different fork line.
  Author = **auxentr, #152 @ 1.25**.
- **`toprakg/taaf-kaggle-source-duck-prolong-memory`** + notebook `toprakg/taaf-duck-prolong-memory`
  (08-04). Author = **Toprak Gundogdu, #130 @ 1.28**.
- Adjacent, same author as the first: **`iseesmth/duck-harness-nca-{source,training,qwen36-adapter}-20260811`**
  — a **trained LoRA adapter for Qwen3.6** shipped alongside a duck fork. Training-gated, off-budget
  for us, but the first public adapter-on-duck artifact.
- Also new: **`thtennant/taaf-kaggle-source-share-fork` "(banking)"** (08-11), author
  *Beyond Good and Eval* **#127 @ 1.28** — our dead/blocked **banking** lane, tried publicly.

**Verdict: MONITOR (do not build).** Both PRO-LONG ports currently sit **below our 1.33** (1.25 / 1.28),
but both predate the published source, so that is not yet a verdict on the mechanism. **Kaggle will
surface each notebook's best score automatically as they resubmit — this is a free external read on
whether state-externalization moves the number at 27B, i.e. exactly the question our port is meant to
answer, being answered by someone else at their cost.** Add a standing check to the daily brief
(same slot as the Jason Feng notebook monitor from 08-09).
**Impact:** could save or redirect the entire port budget within days.

### 6. LB FORENSICS — the gold drift **decelerated**, and the top of the board consolidated

Full 2,229-team CSV vs the 08-04 sweep and `runs/lb_daily/*.csv` (the archive the 08-04 process note
created — it worked, and it is what made exact per-team deltas possible today):

| quantity | 07-28 | 08-04 | **08-11** |
|---|---|---|---|
| teams | — | 2,048 | **2,229** (+181) |
| gold line (top-13/14) | 1.49 | 1.56 | **1.58** |
| top-5 prize line | — | 1.61 | **1.62** |
| teams ≥ 1.49 | — | 19 | **26** |
| teams ≥ 1.40 | — | 46 | **56** |
| our rank @ 1.33 | #51 | #70 | **#84** |

- **The 08-04 extrapolation is falsified.** That sweep recorded gold drifting **+~0.01/day** and
  planned for ≥1.6 by Nov. Realized drift 08-04→08-11 is **+0.02 in 7 days ≈ +0.003/day**, and the
  per-interval series is monotonically decelerating: **0.0125 → 0.0067 → 0.0040 → 0.000/day**
  (07-28→08-01→08-04→08-09→08-11). The gold line has now been **flat at 1.58 for three days**
  (08-09, 08-10, 08-11). Naive linear extrapolation of the *recent* rate is not credible; a decaying
  fit puts the Nov-2 gold line around **1.62–1.70**, not "1.6+ and climbing fast".
  **Posture change: the target is closer to stationary than we assumed.**
- **Movers > 0.1 since 08-04 — only two, and both are informative:**
  - **Tufa Labs 1.45 → 1.62 (+0.17), into #5, on 08-10.** The authors of the harness our fork
    descends from, with 102 submissions, jumping 0.17 in one draw after sitting at 1.45 for weeks.
    This is the single strongest capability signal on the board: whatever they changed is available
    to the people who understand the duck best. **Their public bundles (`driessmit1/…`) are attached
    to every fork including the two in finding 5 — watch for a new `driessmit1` dataset version.**
  - **Helmut AGI — new entrant at 1.61 (#8), first seen 08-09.** A **merger**
    (`antonijankowski123` + `jakobbrggen` + `nightmareblocks`); Antoni Jankowski was the 124th-place
    account recruiting in thread 732706 on 08-04 **(known internally)**. The mid-board consolidation
    meta-signal flagged on 08-05 has now produced a top-10 team — and that team is the source of
    findings 1 and 2.
- **New to the top band since 08-04** (below the 0.1 mover bar, or new names): Lord Han Solo 1.65 (#3),
  DhanaLakshmiMalla 1.60, Tecnod8.AI 1.61, Biubiu / ippeiogawa 1.58, Mathurin Ache / NoOneAhead 1.56,
  Scott Le Grand 1.50 (08-08), amosokello451 1.50, **Souhardya 1.49 in 3 submissions**.
- **Head is frozen:** KOJIMA **1.86** unchanged (65 subs, resubmits daily); **Andy liu 1.69 idle since
  08-03 on 7 total submissions** — the strongest single evidence on the board that a very good draw
  can be had cheaply, and a standing rebuke to cadence-as-strategy.
- **Fossilized draws persist:** anngle 1.56 idle since 07-25 (still #16); Tshithihi 1.44 idle since
  07-03 (#41).
- **Our slide is population, not decay:** 86 teams are now ≥ 1.33; we lose ~2 ranks/day at a frozen
  score, and 181 teams joined in a week.

### 7. MODEL LANDSCAPE — nothing new in-window, and the GLM-4.6V premise is **weaker**, not stronger — **ADAPT (de-prioritize)**

- **No open-weights release between 08-01 and 08-11.** `llm-releases.com` shows nothing in the window
  (most recent catalog entries: DeepSeek-V4-Flash-0731 on 07-31, Gemini 3.5 Flash-Lite 07-21,
  proprietary). HF "recently modified" is finetune churn (`qwen3-vl-2b-instruct-lmk`, VLA robotics
  adapters), no new base VLM.
- **The A17 wall-closer premise takes a hit from finding 3.** Our standing rationale — *"the harness is
  MULTIMODAL, therefore the wall-closer must be a multimodal model"* — is contradicted by the current
  public SOTA: **Retrodict shows the agent an image exactly once (opening-board priming) and never
  again**; all 25 games are then solved through a text log. If the loop does not need vision, then
  swapping the whole reasoner for a 106B MoE VLM (GLM-4.6V-AWQ, ~55–60 GB) is buying vision we would
  use once, at the cost of every token of decode speed in a 9h wall.
- **Concrete alternative shape (new, cheap to price):** a **split stack** — a small VL model for the
  single priming call (GLM-4.6V-**Flash** 9B, or Qwen3-VL-2B, both trivially co-resident in 96 GB) +
  a fast text reasoner for the loop. The strongest reasoner candidate for that role is
  **Qwen3-Coder-Next 80B-A3B** (Apache-2.0, 256K ctx, vLLM/SGLang, **3B active**, ~48 GB at Q4_K_M,
  ~86 GB at Q8) — MoE sparsity means far more LM calls per wall-clock hour than a dense 27B, which is
  the currency finding 3 says actually buys score. **Note the honest counterweight: Nosumina's negative
  (finding 4) was obtained on Qwen3-Coder-Next.** Its failure there was at world-model *authoring*,
  not at the Retrodict-shaped task, but this must be pre-registered before any screen.
- **Verdict: ADAPT — demote GLM-4.6V-AWQ from "first in the A17 successor queue" to "screen only if a
  vision-in-the-loop design survives"; promote a priming-only VL + fast-MoE-reasoner split stack to the
  head of the queue.** gpt-oss-120b and Gemma-4-31B unchanged (WATCH).

### 8. SCORING-MECHANICS NOTEBOOKS — confirm what we know, and explain the low band — **IGNORE (for us)**

Three well-made new public notebooks, all measurement, **all by authors who are not on the leaderboard
at all** (`nekkon`, `maximolorenzoylosada`, `busyaprime` — no LB entry), so: zero efficacy evidence.
- `nekkon/the-80-action-cap-ceilings-you-at-8-7` (08-07) and
  `maximolorenzoylosada/your-agent-stops-after-81-actions` (08-09): the shipped reference `Agent` has
  `MAX_ACTIONS = 80`, undocumented in the README, and the guard is `<=` on a zero-based counter so the
  loop runs **81**. Cumulative human baseline exceeds 80 actions before the end of level 2 in **14 of
  25 games**; median levels reachable = **1**; **ceiling on the final score ≈ 8.5–8.7%** for such an
  agent even at perfect human efficiency.
- **Does not apply to us.** Our solver is the duck line:
  `duck_eval/taaf_bundle/.../taaf/kaggle_random.py` sets `DEFAULT_MAX_ACTIONS_PER_GAME = None`, and
  our shipped `preamble.txt` records `max_actions_per_game=None`. **Confirmed unbounded.**
- **It does explain the sub-1.2 band's composition**, which is useful for the LB process model: a large
  share of the ~2,100 teams below us are reference-agent forks with an 8.7% structural ceiling, i.e.
  the low band is not evidence about difficulty.
- Their scoring restatement — `level_score = min((baseline/actions)², 1.15)`,
  `game = Σ(i · level_score_i)/Σi` — is **already in `discussions_2026-07-23.md`** and unchanged.
  The one framing worth re-quoting to the war room: **1.41× human actions halves the level; 2× gives
  25%. Completing 3 of 6 levels caps the game at 28.6%.**
- `busyaprime/the-agent-never-sees-its-own-score` (08-10): proves from `FrameData` that the agent is
  never shown the efficiency-weighted score (computed by a separate `EnvironmentScoreCalculator`).
  **Known**, no action.

### 9. LOW-PRIORITY / FILED SO NO FIELD IS RE-DECLARED EMPTY

- **`jinbowang1/arc-prize-2026`** (pushed 08-11): perfect **100.00** on `ls20`, `tr87`, `ft09` and
  (08-11) `cd82`, with full per-level action tables (e.g. `ls20` 335 actions vs human 776). **IGNORE
  for lane value** — the README states the rules were **induced by a human**, and only `cd82` L1–L2
  were played by the harness unaided; a `feedback_arc_generalization_first` violation by construction.
  Its genuinely useful half is the boundary section: the author reports **his own experiments
  overturned two of his boundary claims** ("search-proof `tr87`", "click games defeat search"), and
  that **the perception layer still does not generalize** — a third independent voice on the same
  perception-is-the-bottleneck theme as findings 1 and 4.
- **Research, 08-08 → 08-11: nothing plan-changing.** (Internal research sweeps already cover through
  arXiv's 08-07 submissions.) New in the un-covered window: **2608.09888 BDH-CQ** (recurrent latent
  reasoning, 29.5% pass@2 on **ARC-AGI-1** at strong cost-efficiency — wrong benchmark, static puzzles,
  **IGNORE**); **2608.08055 SodaMem**, **2608.08236 LatticeMind**, **2608.08253 SuperLocalMemory**,
  **2608.07855 CommitKV**, **2608.07107 MemWM** — agent-memory plumbing, all **IGNORE/shelf** under
  the standing rule that context-shrinking work is sequenced behind externalization;
  **2608.09696 / 2608.09537 / 2608.09298 / 2608.08600** world-model papers, all non-agentic-harness
  (Bayesian experiment design, embodied manipulation, multi-agent rendering) — **IGNORE**.
  **arXiv has produced no ARC-AGI-3 paper since 2608.04066 (08-04).**
- **arcprize.org blog: still nothing since 2026-07-06** (fourth consecutive sweep with no host post).
- **Other new repos, all ≤1 star, none beating duck:** `sonpham-org/arc-3` (instrumented Tufa fork +
  GCP spot kit + reproduction matrix), `pinion05/arc-agi3-duck-harness-vllm` (Duck vLLM reproduction on
  Vast.ai A6000, conv1d CUDAGraph patch — **the one operational note worth a look if we ever rebuild a
  vLLM rail**), `albertvucinovic/arcagi3-physics`, `secondorderai/arc-agi-3-harness`,
  `Alexyskoutnev/TWIN-ARC-AGI-3` (unchanged since 08-02), `CalamityChasm/…JEPAstyle…` (dead lane).
  `GitMonsters/SOLVED---abc82100` claims 540/540 across ARC 1/2/3 via 514 standalone per-task solvers —
  **IGNORE** (per-task solvers, unverifiable, generalization violation).
- **Public notebook ceiling:** unchanged — boristown 1.47 remains the best public artifact; the
  top-voted *new* notebook is `jakobbrggen/taaf-anim-arc-agi-3-solver` (23 votes), which is the TAAF
  wrapper only — the value is in its attached dataset (finding 1).
- **Tara Labs (caoyupeng) #34 @ 1.46** publishes `arc3 duck v12` (62 votes); **thtennant #127 @ 1.28**
  publishes `arc3 duck v12/v18` (40 votes). The public-fork churn continues to cluster at 1.2–1.5.

---

## TOP 5 FOR THE WALL

1. **Our agent has never seen an intermediate animation frame — verified in our own bundle.**
   `taaf.game.GameState.all_frames` / `animation_frames` exist; ARC3-Inference has **zero consumers**.
   13 of 24 games return multi-frame responses, and on type-1 games (**`ft09`**, `sb26`) the entire
   signal lives between the first and last frame. A 1.61 team published the fix, in public, on 08-07.
2. **Retrodict (08-10) put the public ceiling at 99.86 RHAE with 5.5× fewer tokens — and its lift is in
   the runner, not in an LLM-authored world model.** Per-action expected-cell predictions checked by
   deterministic code; the executable simulator only appears as a 300-action escalation. That is the
   affordable half, and it is exactly the half our Tycho port left out.
3. **Nosumina independently reproduced our L0 negative at local scale** — a quantized Qwen3-Coder-Next
   writing certified `GameModel`s scores near-0% exact-match, indistinguishable from random. Two
   independent negatives now say: **stop asking a small model to author the world model.**
4. **Gold drift decelerated to a stop: 1.56 → 1.58, flat three days, realized +0.003/day vs the
   +0.01/day we planned against.** Meanwhile Tufa Labs jumped **1.45 → 1.62** in one draw and a
   three-way merger (Helmut AGI) entered at 1.61. The target is nearly stationary; the field is
   consolidating.
5. **Two competitors have already ported programmatic-memory/state-externalization onto the Duck at
   27B and published the source (08-04, 08-11).** Both currently sit below our 1.33. Their next
   submissions are a free readout on our own lane-(a) hypothesis.

## WHAT THIS CHANGES

**The port: re-aim it, do not cancel it.** `port_rescope_2026-08-10.md` concluded that all three
lift-bearing Tycho axes (C8 builder, C9 metareasoning, C12 frontier backend) are out of scope and the
re-scope R25-N2 demanded is unaffordable (3.5× wall, 52× LM calls). Findings 3 and 4 resolve that
deadlock from the other side: **the lift-bearing component of the *current* public SOTA is
deterministic and harness-side** (runner-checked per-action expectations + plan batching +
externalized log), and **the LLM-authored world model is now carrying two independent negatives at our
model scale** (our L0 0/13; Nosumina near-random). Recommendation to the panel: **respec the port
around Retrodict's expectation-checking runner rather than Tycho's C1–C7 contract**, and keep the
executable simulator only as a stuck-level escalation, which is also what Tycho's own ablation said.

**Sequencing: findings 1+2 jump the queue, ahead of the port.** They are not a lane — they are a
verified defect in the artifact we ship, fixable in pure Python with no model change, no training and
no spend, aimed at a root cause our own memory already names (state-aliasing), with a public reference
implementation and a free offline verification path (count multi-frame responses across our 25
environment files on the build rail — no LM required). Strict order: **animation awareness first, hard
no-op guard second** — the guard is harmful on type-1 games without it.

**The GLM-4.6V screen: demote.** Its whole premise was "the harness is multimodal, so the wall-closer
must be a VLM". The public SOTA looks at exactly one image per game. Screen a **priming-only small VL +
fast MoE text reasoner** split stack instead, and only revisit a full VLM swap if a vision-in-the-loop
design earns it.

**Option 3 (accept-band): the calculus improves slightly, and for a reason worth stating.** The gap is
0.25 (1.33 → 1.58), statistically the same as the 0.23–0.27 we have carried. But the *shape* changed:
the gold line has gone flat for three days and its drift has decayed by 4× since 07-28, so the band we
must eventually accept is closer to stationary than the 08-04 sweep assumed — less of the gap will be
handed back to the field by drift alone. Against that, two of the last three top-10 entrants arrived
by **merger** rather than by capability, and Andy liu holds #2 on **seven** submissions. **Nothing here
argues for accepting the band today**; it argues that the deadline pressure is slightly lower than
modelled and that the cheapest remaining capability lever (finding 1) is one that no amount of
draw-harvesting substitutes for.
