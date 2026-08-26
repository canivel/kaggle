# Daily brief — Wednesday 2026-08-12

Protocol: STEP 1 collect + deep review (1a result deep-dive, 1b discussions, 1c research, 1d this
merge), STEP 2 panel, STEP 3 develop, STEP 4 validate/submit, STEP 5 loop. **No panel today** — full
panels are Sunday cadence; R24 ran 08-09 and the next full panel is **2026-08-16**. Discussions were
swept today (the browser route was restored, so the 08-10/08-11 coverage gap is closed).

**The day's spine is a self-correction chain.** A morning finding said we could reach gold with no
new capability; our own validation refuted it, then refuted its replacement, and what actually
shipped is a much smaller arm authorised on the corrected basis. Around it: a 1.07 draw, a new #1 at
2.52, and the animation arm killed on its own sealed canary.

---

## 1a. Result deep-dive

### The draw — 1.07, and the first trailing-4 above 1.00 in the record

The overnight frozen-fork filler (`canivel/arc3-duck-repro` v3, fired 00:07:11Z, API `COMPLETE`)
returned **1.07** — **z ≈ +0.81** against the n=28 record (0.9461 / 0.1523), the **fifth consecutive
interior result** (0.87 → 0.89 → 1.05 → 1.09 → **1.07**) and a marginal step down from yesterday. Far
above the 0.80 line, so the fired-and-resolved-STATIONARY watch-rule stays resolved and does not
re-arm.

- Record → **n=29, mean 0.9503, s 0.1513**. Mean up, dispersion down again (0.1523 → 0.1513).
- **Trailing-4 mean 1.025** (was 0.975) — **the first trailing-4 above 1.00 in the record.**
- Public max unchanged at **1.33**. Under the R25-N3 `ρ̂_draw ≈ 0` finding this is a
  mean/dispersion datum, not a win.
- **Sealed mean-of-4 promotion bar moves 1.0848 → 1.0876.** It drifts with the record; any prereg
  must re-read it from `runs/ledger.json` rather than cache yesterday's number.

**Source conflict, flagged rather than resolved.** `runs/ledger.json` on disk still reads **n=28,
mean 0.9461, s 0.1523, latest 1.09 (2026-08-11), trailing4 0.975, promotion_bar 1.0848** — it has not
been re-derived since yesterday. `ITERATION_LOG.md` (08-12 morning check) and
`learnings/war_room/cstl_breakout_2026-08-12.md` §2 both carry **n=29, 0.9503, 0.1513** (CV 0.159).
Two sources against the file; the file is the one the scripts read. Do not quote the bar from the
JSON until it is re-derived.

### The board — cstl enters at 2.52, a new #1 by +0.66

Archived `runs/lb_daily/lb_2026-08-12.csv`. **cstl enters at 2.52** (submitted 08-11 18:25) — the
first score above 2.0 on this board, **+0.66** over the previous leader YUTO KOJIMA at 1.86, and the
largest single-entrant jump of the campaign.

- **Top-5 prize cutoff TIGHTENS 1.62 → 1.64** (#5 is now GeniusYY 1.64; Tufa Labs 1.62 falls out of
  the prize band one day after entering it).
- **Gold/top-13 cutoff HOLDS at 1.58 — fourth flat day.** Composition shifts: DhanaLakshmiMalla
  enters at 1.60 (#11), the 1.58 pack at #12–13 is Biubiu / ippeiogawa, and Nkosi Ndwandwe slips
  #13 → #14. Yuchen20 sits #15.
- Head otherwise static: Andy liu 1.69, Lord Han Solo 1.65, Tecnod8.AI / FOYSAL / hvp / Helmut AGI
  all 1.61. **Our 1.33 unmoved, below #49: gap to gold 0.25 unchanged, gap to the prize line widens
  0.29 → 0.31.**

**cstl investigated exhaustively** (`learnings/war_room/cstl_breakout_2026-08-12.md`). Single leap,
not a staircase: 22 submissions at the 08-11 11:24Z official snapshot (still 1.59), **23** at the
08-12 10:53Z snapshot (2.52), `LastSubmissionDate` 08-11 18:25:39 ⇒ **submission #23 is the +0.93**,
after a 2-day pause and a 1.59 held flat since 08-04. **Draw variance is falsified by 5–7 orders of
magnitude**: conditioning on their own best-of-22 = 1.59 via Blom's `E[max of n]`, 2.52 is
**z = +6.68** at our CV 0.16 and **z = +5.28** at yw8837's CV 0.26 ⇒ P(any of 23 draws) between
2.7e−10 and 1.5e−06. **Public artifacts by cstl: ZERO** — `gatamaz` (Tamaz Gadaev, SF, 1 competition)
and `tehnar` (Amsterdam SWE, one 2015 notebook) have no kernel, dataset, model, forum post or comment
anywhere; HF and GitHub checked; **nothing is liftable, and any claim about how they did it is
speculation.** The identification `gatamaz` → GitHub `tamazgadaev` rests on a name plus location
match through a secondary aggregator and **must not be repeated as fact**.

**2.52 is legal and unremarkable in absolute terms.** The metric runs 0 → ~115 (public SOTA Retrodict
is 99.86 with a frontier API at $654, not Kaggle-legal); 2.52 ≈ **71.5% of games clearing level 1 at
human action efficiency** — a first-level agent, not a deep one. Discussion 734414 confirms a
submission that skips the real Phase-B rerun finishes in ~30 s and scores 0.00, so a 2.52 required a
genuine long rerun.

### Animation-awareness — KILLED on K-A3, with the tension logged

`canivel/arc3-duck-animation-eval` v1 pulled to `runs/kernel_pulls/animation_v1/`, scored by the
pre-registered `duck_eval/warpack/animation_score.py` against
`learnings/war_room/animation_prereg_2026-08-11.md` → `runs/animation/score_2026-08-12.{json,md}`.

- **K-A0/K-A1/K-A2/K-A4 PASS** (banner `animation v1: ACTIVE (4 seams patched)`; 1,287 `ANIMATION `
  event lines on 18 distinct games; invisible engaged on 3 of 4 type-1 games; `animation_errors=0`).
- **K-A3 FAIL ⇒ KILL:** `tokens_est 57,915 / total 1,638,444 = 3.53%` against a sealed bound of
  **<1%**.
- **M0 (primary, delivery only):** `invisible/executed = 16/5,151 = 0.311%`;
  `multi_frame/executed = 1,287/5,151 = 25.0%`. The MULTI rate **reproduced the offline audit almost
  exactly** (25.0% live vs 23.2% audited) — the taxonomy is right. The INVISIBLE rate came in
  **~12× below** the audit (0.31% vs 3.6%), concentrated in the one game that mattered: **`ft09`
  produced ZERO multi-frame responses in 115 executed actions**, against an audited 80.7% MULTI /
  79.8% INVISIBLE. Verdict **DEVIATION**. This is exactly the failure Jakob Brüggen pre-warned about
  in 734369 — random-walk probe rates do not survive contact with a real agent.
- **M1:** `uninformative in both directions` — the only legal string. Family
  `duck-harness-kaggle-continuation-v1` is m=2 ⇒ **NOT SCREENABLE** (`SCREEN_PROTOCOL` §1 P2).
  Descriptively: arm **lc 17** vs family totals {16, 10}, paired mean Δlc **+0.160/game** (8W/4L,
  sign-flip p=0.339). **May not be reported as non-harm.**
- **M2:** the harm mechanism did **not** reproduce — tokens/action **318.1 arm vs 337.3 family =
  −5.70%** (Brüggen's arm was **+17%**), and the arm executed **more** actions/game (206.0 vs 189.0,
  +9.0%) at the same wall clock. **M3:** repeated-identical-no-op rate on the four type-1 games
  **0.032 vs 0.065 = 0.49×** — roughly halved, and the one real win.

**The tension, stated rather than smoothed:** K-A3 fired on a **static 45-tok × 1,287 estimate** while
the quantity it exists to proxy — measured token inflation — was **negative**. The sealed rule was
applied as written and the kill stands, but the honest reading is *"the bound was mis-specified
against a generated-token denominator"*, not *"the summary was expensive"*. A successor must (i) fix
the `canary_report()` token denominator (its `token_fraction=` field is empty by construction),
(ii) drop probe-B rates as an expectation basis, and (iii) justify itself on M3, not M0 — M0
delivered a mechanism worth 0.3% of actions. Arm killed per §5, module reverted, nothing promoted, no
submission, no queue change.

### The correction chain — two headline claims generated and refuted by our own validation, same day

This is the intellectual content of the day and the order matters.

**Claim 1 — "gold with ZERO new capability" (WRONG).** `cstl_breakout` §4.3 found that our animation
run clears **17 levels for 1.635**, and the same 17 levels re-scored at exactly the human action
baseline give **2.549 local ≈ 1.48–1.58 LB — the gold line**, with six games (bp35 8.33×, ar25 5.97×,
sp80 5.77×, m0r0 4.6×, tu93 3.68×, vc33 3.0× the human action count) burning **56% of achievable
score**, quadratically because `level_score = (baseline/actions)²`. The morning read: action
efficiency, not level depth, is the binding constraint, recoverable with no new capability.

**Refuted the same day by `learnings/war_room/efficiency_diagnosis_2026-08-12.md`** — a full
state-graph replay of all 5,151 actions (event logs reproduce `benchmark.json`'s `actions_per_level`
**exactly for all 25 games**; scorer validated at 1.776e-15 over 1,000 cross-checks). The 0.914-point
gap is real and correctly computed; the "zero new capability" reading is not supported. Bucket table
over the 17 cleared levels, 1,110 actions vs 526 human baseline = 2.11×:

| bucket | actions | share | preventable? |
|---|---|---|---|
| (a) necessary non-repeating probes | **798** | **72%** | no |
| (b) duplicate `(state,action)` | 117 | 10.5% | yes, airtight |
| (c) blind-batch tail | 195 | 17.6% | yes, airtight (runner-side) |
| (d) post-solution dithering | **0** | 0% | **structurally impossible** |
| (e) re-traversal (cross-cutting) | 180 | 16.2% | partly |

Gap decomposition: **20% redundant bookkeeping (+0.184) / 40% search-policy up to the oracle
(+0.361) / 40% irreducible capability (+0.369)**.

**The efficiency ceiling, and it is short of gold.** Two independent estimators converge — hindsight
replay-optimal oracle **2.180** and one-environment-action-per-LLM-decision **2.199** ⇒ **~2.19 local
≈ 1.26–1.36 LB**. That would turn our best-ever draw (1.33) into our expected value, but **it does not
reach the 1.48–1.58 gold line. ~40% of the gap is genuine capability, not bookkeeping.**

Three mechanisms were killed before any build: **stop-when-solved** (bucket (d) is structurally zero —
the harness ends the level on the completing action, and 16 of 17 levels completed on first arrival at
the winning state); **RESET-refund** (sp80's own final batch `RIGHT,RIGHT,RIGHT,SPACE` after its 8th
RESET is a 4-action solution that still scored 0.143); **plan-batching** (batching correlates **+0.45**
with log(redundancy) in our traces — we blind-batch *more* when flailing, one batch was 57 actions
fired blind, the opposite of GPT-5.6). The single cleanest counter-datum: **vc33**, the only game
shared with the GPT-5.6 probe — GPT-5.6 cleared L1 in **7** actions (1.17× human), we took **21**
(3.00×) with **zero** duplicates, no-ops or revisits and a replay-optimal path of exactly 21. **100%
of that waste is capability.** Aggregate redundancy: GPT-5.6 **1.10×** vs ours **4.33×**; worst
per-level 1.88× vs 8.33×; GPT-5.6 collapses to **one LLM turn per level** once the rule is known.

**Claim 2 — "+0.184 efficiency gain" (WRONG; an accounting artifact).** The diagnosis's own #1
proposal (M1+M3, runner-side, "airtight") replicated at **×1.11 / ×1.11 / ×1.09** on three
independent runs and was written up as the highest verified score-per-day item on the board — and was
briefly recorded as APPROVED at **+0.184 local (1.6352 → 1.8188), LB ≈ 0.95 → 1.05–1.06**.

**Refuted by the P1 build's own replay validation, before the arm ran.** The build reproduced every
diagnosis number exactly (as-run 1.6352194 / 1.4075 / 1.4509; dup bucket 117; re-traversal 180; all 17
per-level rows; the latent-state game set and pair counts m0r0 55, re86 19, sk48 11, ka59 10, cd82 8,
g50t 4, dc22 3, wa30 2, with zero false positives) — and then found that the mechanism **exactly as
specified deletes the level-completing action of 3 of the 17 cleared levels** (tu93 L1, sp80 L1,
ar25 L1) **and scores them as still cleared**. Straight from the traces, the winning batch *opens by
re-traversing already-visited boards*:

```
sp80 L1  step 62  batch 4/4   RIGHT(revisit) RIGHT(revisit) RIGHT(revisit) SPACE<-COMPLETES
tu93 L1  step  6  batch 16/16 DOWN(revisit)  RIGHT UP RIGHT ... DOWN<-COMPLETES
ar25 L1  step 23  batch 15/15 LEFT(revisit) x5  DOWN x9  DOWN<-COMPLETES
```

Abort-on-revisit cuts all three at action 1; `memo_mode=all` declines the same openings. **tu93 and
ar25 are precisely the two games the diagnosis names as supplying 87% of the claimed gain.** The
spec's own kill rule (`lc ≥ 16`) fails before the arm runs.

**Claim 3 — what actually ships.** With the safe defaults (below), replayed effect is
**×1.040 / ×1.003 / ×1.015, mean ×1.019 ≈ +0.02–0.07 local ≈ +0.01–0.04 LB** at c ≈ 0.58–0.62 —
inside the noise of a single draw (s = 0.1513, n = 29). **Two headline claims were generated and
refuted by our own validation within the same day.** Both refutations came from replaying our own
recorded traces at $0, before anything was spent on them.

*One documented non-reproduction inside the chain, carried rather than reconciled:* the diagnosis puts
the blind-batch tail at **195** actions; the P1 build's replay puts it at **101** under a
dup-takes-precedence partition. The prereg states the total removable count and the resulting score
are unaffected to within 0.3%, which is why the multiplier reproduces — but the two numbers are in the
record and they differ.

### P1 arm — PUSHED and RUNNING, authorised on the corrected basis

`canivel/arc3-duck-p1-eval` **v1 → RUNNING**, kernel push **slot 1 of 2**. Prereg
`learnings/war_room/p1_prereg_2026-08-12.md` **SEALED pre-push**; module
`duck_eval/warpack/_kaggle_dataset/p1_suppressor_patch.py` (v1, 4 seams); validator
`duck_eval/warpack/p1_replay_validate.py` → `runs/p1_replay/report.json`; smoke
`duck_eval/warpack/p1_smoke.py` **76 passed / 0 failed** (structural, unit, replay, real
offline-engine integration, kill-switch subprocess); builder mode `--p1`. Dataset `canivel/arc-war-kit`
versioned first, **byte-audit 10/10 MATCH**; kernel pull-back code-cell concat sha256 MATCH with the
arm flag, graft import and canary call present in cells 2/12/14. Push report
`duck_eval/warpack/p1_push_report_2026-08-12.md`.

**The stop rule was first read as fired** (the arithmetic reproduces; the *gain* does not). The
coordinator then authorised the slot **explicitly on the inflated-framing-corrected basis**: the
suppression half is worth ~+0.02 local, not +0.184, and the slot is being spent for **mechanism C —
the non-truncatable memory block, which is unmeasurable offline** and targets the day's root-cause
finding.

- **Shipped defaults are the safe arm:** `P1_MEMO_MODE=noop` (decline only pairs whose confirmed
  outcome left the board byte-identical); `P1_CONFIRM=2` **clamped in code** (at confirm=1 no pair is
  ever executed twice, the latent-state detector can never fire and the safety constraint would be
  void — a floor, not a flag); `P1_MAX_DECLINES=1` (a repeat request always executes, so nothing can
  be permanently blocked); `P1_ABORT_REVISIT=0`. Replayed with these defaults: **zero level-completing
  and zero board-changing actions declined or aborted on all three runs**; levels preserved on all 8
  latent-state games; dup rate 10.5%→4.9% / 4.9%→3.8% / 5.0%→3.6%.
- **The online latent-state detector reads no game id**, reproduces the published 8-game set exactly
  on `animation_v1`, and flags a *different* set on the other two runs (cn04/sc25 in, re86 out) —
  proving a hardcoded list would have been both illegal and wrong.
- **Primary endpoint M0 = `saved/requested`** (`saved = declined + aborted`), sealed band
  **[3%, 30%]** against replayed 5.9% / 20.0% / 17.6%. Canaries K-P0..K-P3 hard (failure ⇒
  discard-grade), K-P4..K-P6 reading gates.
- **No token-fraction canary is pre-registered** — mechanism C's cost is *input* tokens and the rail
  reports generated only (`final_uncached_input_tokens=0`), which is exactly the mis-specified
  denominator that killed the animation arm. Cost is bounded statically instead: ≤900 chars by
  construction, measured 389, asserted by smoke U13/I5c.
- **M1 is descriptive only** — family m=2, NOT SCREENABLE, advisory K3″ line −0.2977 at σ̂=0.14174
  (df 6), C(2)=2.10.
- **Nobody may read this arm against ×1.10.** The prereg says so in terms, and adds that the arm
  **cannot be evaluated on one LB draw and no LB draw may be attributed to it**.

### Root cause on record — the agent re-explores because it has FORGOTTEN

From `prompts/sp80-589a99af_p0.log` and the step-37 transcript at action 129:
`context_budget_tokens: 31744`, `history_messages: 33`, on a level that ran **225 actions over 62
analysis steps** — by action 129 it has forgotten most of what it tried, and its reasoning recalls
*"I already tried clicking on the charcoal and magenta blocks"* from compacted prose rather than
ground truth. **The harness already exposes `history`, `transitions` and `last_transition` as
preloaded Python globals; the agent simply never queries them.** Every game terminated `gave_up` after
~66–69k tokens (5,151 actions, 1.63M tokens, 2h12m) — the binding constraint is the token budget, not
the action budget. The existing batch-abort path (`stopped_early`) fired only 10 of 190 times.
**Diagnosis: re-exploration is caused by context truncation, not by a missing loop detector**, so the
fix belongs in the runner, not the prompt (the prompt already says "optimize for as few in-game
actions as possible", and `feedback_prompt_is_noise` applies).

## 1b. Discussions sweep — forum route RESTORED, swept 08-10 → 08-12

Frontier is now **734585** (was 734369). The chrome-devtools lock that blocked the 08-10 and 08-11
sweeps was an **orphaned Chrome from 08-09 18:13**, not a live session; killing only the
`chrome-devtools-mcp*chrome-profile` processes and clearing `Singleton*` fixed it instantly. **Check
process creation dates before assuming another agent owns the profile.**

- **No post anywhere mentions cstl, the 2.52, or a new #1** — no congratulation, no host statement, no
  technique thread that would explain it. The forum has not noticed.
- **734369 "Write Up: Taaf Anim Agent"** (Jakob Brüggen, Helmut AGI) now has two comments. His own
  public-set A/B was **NULL (+1.4%, p=0.92)** at **+17% tokens/action**, with only 2 of 181 tool calls
  hitting informative animation. **Xuan** reports the animation frames help `gpt-5.6-sol` but *"blow
  up the context window for Qwen"*, and surfaces **VISTA** (`vista-research.github.io`) — vision-only,
  upscales the 64×64 grid to 512×512, claims ~100% for Claude/Sol — but says it *"just does not work
  as well"* for Qwen (right intuition on ft09, wrong coordinates). **An independent same-model negative
  on vision-in-the-loop at 27B**, reinforcing the 08-11 sweep's demote-the-VLM verdict. Host **Greg
  Kamradt** replied with acknowledgement only — no rules or scoring content.
- **734414** (mina wailin): Phase-B rerun not firing ⇒ 30 s dummy parquet ⇒ 0.00. Operational, and
  used as legality evidence for cstl's 2.52.
- **734585** (Jason Feng): GPU quota blocking submission. Operational.
- **732854**: the "2.8 locally" comment is now attributed to **Son Pham** (#191 @ 1.22).
- Artifact churn confirming the animation write-up is propagating (`iseesmth/prolong-eval`,
  `iamjasonfeng/chimpanzee-1-1-anim`, `cascadematrix/arc-agi-3-causal-animation-v1`,
  `finalsunflower/arc3-anim-lb161-exact-validation`). **No cstl artifact among them.**

## 1c. Research sweep — `learnings/sweep_research_2026-08-12.md`

**15 items · 0 ADOPT · 3 ADAPT · 12 IGNORE.** Bar applied: ADOPT/ADAPT requires a named runner-side
change **and** a named endpoint on our rail; default IGNORE. Aimed first and hardest at
memory/state-externalization under a small context budget, because that is where the arm on the rail
is spending its slot. **Zero ADOPTs is called the correct outcome** — all three ADAPTs are small,
runner-side, zero-LLM-cost refinements of work already in flight.

| Item | Disposition | Substance, endpoint, constraint |
|---|---|---|
| **Prime Agent (Prime Intellect, MIT, 14.6k★, released 08-05)** | **ADAPT — one mechanism only** | Self-reports **95.5% RHAE Best@1 on ARC-AGI-3 with Opus 5** (95.0/95.2/95.5, Best@3 99.97, all 183 levels, above the 95.4 human baseline). **Discount hard: self-reported; the repo README does not mention ARC-AGI-3; the official board still shows Opus 5 at 30.2%; frontier API on public API games, not our 27B Kaggle rail. Nothing about the score transfers.** What transfers is the architecture: the *only* tool is a persistent IPython kernel and the agent reaches its own state **programmatically** rather than by reading a rendered summary. **The duck harness already has that kernel and already holds `transitions` in it.** Change: ~50 lines of zero-LLM helpers in the Python tool preamble — `untried_here()`, `tried_here()`, `dead_pairs()`, `path_to(board_hash)`, `distinct_boards()` — plus one worked call. **Endpoint: delivery, exactly as P1 M0 — fraction of analysis steps whose emitted code calls ≥1 helper, pre-registered with a band.** Risk: a 27B model may simply never call them, the same failure class as never querying `transitions`. **Sequencing: strictly AFTER the P1 arm is pulled and read** — C is the *push* side, this is the *pull* side, and they must not be confounded in one build. |
| **Addressable Recall Compaction — arXiv:2607.25066** | **ADAPT (lossless-eviction half only)** | Tool observations go to an append-only, ID-addressable log; when space is needed, older observations are replaced by **compact citations** rather than dropped. NIAH 99.40% vs 88.12%; LongBench-v2 Hard 29.97% vs 28.25%. **Evaluated on Qwen3-8B @16k and Qwen3-32B @32k — our model class and our exact context budget (31,744).** Closest published match to our defect all sweep, but take only half: the `recall(id)` half requires the agent to issue a query, and our documented root cause is that it does not query what it is handed. Change: on eviction, substitute a one-line ID-addressable stub (`[s12 CLICK(3,4)@b#a71f -> no change]`), statically bounded in characters, zero LLM calls. **Endpoint: dup-`(s,a)` re-execution rate (already instrumented by the P1 canary) and `levels_completed` ≥ 16.** **Routing: this is a compaction-lane change** — A22 is formally open/unworked with no builds bought and revival condition **R2 = a surviving mechanism claim is required**. This is a *candidate* R2 claim; it does not on its own justify buying a build and **must not be smuggled into the P1 successor as an unpriced extra**. |
| **BeliefMem — arXiv:2605.05583** | **ADAPT (narrow — the latent-state games only)** | Stores multiple candidate conclusions with probabilities updated by Noisy-OR, targeting the self-reinforcing-error failure. LoCoMo + ALFWorld, no model sizes disclosed. Relevance is specific: **8 of our 25 games carry ambiguous `(board_hash, action)` pairs** and P1 handles them by hard-disabling mechanism A; the diagnosis calls **m0r0 (19.9% of the whole 56%) essentially irreducible** for exactly this reason. Change: in block C, for flagged games report per-pair **outcome multiplicity** (`tried n=3, outcomes 2 distinct`) instead of a binary dead/alive label, and never emit "confirmed dead" for a pair with >1 observed outcome. Pure arithmetic over the memo P1 already maintains, inside the 900-char budget. **Endpoint: `levels_completed` on the 8 flagged games must not fall (already P1 kill rule #3), and dup-`(s,a)` rate on flagged games must stay above zero** — zero would mean we suppressed legitimate re-probing of latent state. **Do NOT import the Noisy-OR machinery** — no calibration data, and it would put an unvalidated numeric model in the action path. |
| **RedundancyBench — arXiv:2605.29893** | **IGNORE as a method — keep as the citation that defends P1's design** | Manually-annotated redundant steps in agent trajectories, three LLM-based detection strategies: **the best method scores 24.88%, and some perform worse than random guessing.** No deterministic detector offered. **This is the standing rebuttal to any future "let the model notice it is repeating itself" proposal** — answer it with that number. It also means **our deterministic `(board_hash, action)` memo is not a crude stand-in for a smarter LLM mechanism; it is the only version that works.** |
| **TraceCompiler — arXiv:2608.02680** | **IGNORE — and be actively suspicious of it** | Compiles clusters of noisy traces into mostly-deterministic workflows; 0.928 P / 0.943 R on 15,775 dependency edges vs 0.711 F1 baseline; 0.993 P on AppWorld; one intent went from 34 API calls to **11**. A 3× action reduction is exactly our currency and we hold 5,151 recorded actions — **so it is the item most likely to be mis-adopted.** It needs *repeated procedures* to compile; **our private-LB games are different games**; compiling workflows out of the 25 games we can see, for a private set we cannot, is the definition of what `feedback_arc_generalization_first` forbids. It also requires training on a labelled split. |
| Memory in the Loop 2607.05690 · Less Context Better Agents 2606.10209 · JAMEL 2606.01528 · OLIVIA 2605.11169 · IAL-Scan 2607.01641 · Life-Harness 2605.22166 · MemoHarness 2607.14159 · Quo Vadis World Modeling 2608.02713 | **IGNORE** | Respectively: retrieval latency is their IV and ours is already zero (`transitions` is in-process) yet we still show 4.33× redundancy — corroborative only; pruned+summarization 91.6% but the delta is bought with an **LLM summarization pass** against our binding token budget; trains memory jointly and compresses to **latent** (unauditable) memory, zero cloud budget; needs per-candidate **hidden states** we cannot get from vLLM, and its premise is falsified by our 72% non-repeating probes; static analysis of **code** loops, ours are semantic; **+88.5% average relative** across 126 settings but its gains live in tool-contract failures and **our action space is five primitives plus a click**; no retrievable number and per-case harness rewriting is unbuyable at m=2; position paper, zero adoptable content. |
| **AERA — arXiv:2605.25931** (ARC-AGI-3-specific) | **IGNORE — logged as prior art for P2** | EXPLORE → VERIFY → PLAN, and formalises RHAE's quadratic form as a second-order penalty for deviating from the efficiency frontier — an **independent derivation of today's quadratic argument**. But **RHAE 0.2116 with Qwen2.5-0.5B, 4/25 public games, 0.30 private** vs our 1.33 public max; nothing to lift. Carries two things: it raises P2's prior slightly, and it claims **24 of 25 public games are solvable by non-intelligent strategies** — an unreplicated claim from a 0.5B paper that **must be verified against our own 25 before it is ever cited as headroom.** |
| **Rudakov graph exploration — arXiv:2512.24156** | **IGNORE as new (already on the record) — published prior for P3** | Training-free; salience-prioritised actions over a directed state/transition graph; median **30/52 levels across six games, 3rd on the ARC-AGI-3 Preview private LB**. Closest published analogue of P3, and its salience ordering is the one component P3 does not specify. **Its vision front-end is out** — two independent same-model negatives on vision-in-the-loop at 27B (VISTA, and our own). |

**Declared coverage.** Searched-and-empty, recorded so the gap is visible: plan-vs-act / batching
empirics (nothing measuring actions-per-decision against redundancy in a stateful game — **our own
+0.452 correlation remains the only measurement bearing on P2, and it is ours**); test-time
training/adaptation for ≤32B (all require gradient updates at inference — category closed unless the
serving constraint changes); BALROG/Crafter/ALFWorld/TextWorld July–August (nothing new). The Kaggle
discussion feed was swept separately (§1b), not here.

## 2. Infrastructure — today's rail work

- **Submission daemon had NO queue-refill path — CONFIRMED, now FIXED.** The 06:00 check found
  `submission_queue.json` pending empty for the **second consecutive morning** (the 00:07Z fire drains
  it and nothing enqueues), which made the morning check the only thing standing between us and a
  missed fire. Fixed with an **auto-refill of the eternal fallback plus a `queue_autorefill` log
  event; 13 new tests pass.** This was a standing daemon-side gap, not an incident.
- **IN FLIGHT — preflight duck-harness family profile.** `scripts/preflight.py` returns **BLOCK on
  K2/K4/K5/K6/K8** for the P1 kernel, **and returns the identical 5 failures on
  `canivel/arc3-duck-animation-eval`** — the kernel that built COMPLETE and produced our primary
  trace. Those checks test the `arc3-baseline` agent-swarm notebook shape (`agents/__init__.py`,
  `.env`, `main.py --agent myagent`, `%%writefile my_agent.py`); the duck-harness eval family is
  taaf+vLLM, so they are **structurally inapplicable, not failing**. K1/K3 — the two that apply —
  pass, and the applicable gate is the structural diff vs the war-eval baseline (exactly cells 2/12/14
  differ, `metadata.kaggle` byte-identical). `--mode trusted-fork` is also inapplicable. **Outcome not
  yet known.**
- **IN FLIGHT — selftest for `p1_score.py`.** The 412-line scorer that will pronounce the verdict on
  the arm currently running is untested. **Outcome not yet known.**
- **CLI runbook sharpened:** `kaggle datasets version -p` with a **forward-slash** absolute path fails
  exactly like a relative one. The 08-11 note "use an absolute path" is now **"use an absolute path
  WITH BACKSLASHES"**.

## 3. Open questions for today

1. **Does mechanism C deliver live — is M0 in band?** This is the only question the running arm can
   answer, and it is the one thing that is unmeasurable offline. `saved/requested ∈ [3%, 30%]` against
   replayed 5.9% / 20.0% / 17.6%. **Below 3% the arm is a null by delivery and may not be re-read as a
   score result** (kill rule #5); above 30% it must be inspected before any reading. Note the standing
   precedent: the animation arm's mechanism-delivery endpoint came in ~12× below its offline estimate
   because probe rates did not survive a real agent — P1's expectation is at least a
   **replay-of-a-real-agent** rate, which is the strongest basis available, but it is still an
   expectation. Also unresolved by design: K-P5 requires the online detector to flag ≥1 game, i.e. the
   safety rule must be *live*, not merely present.
2. **Given the efficiency ceiling is short of gold, what is the actual route to 1.58 — and does
   anything in the current lane reach it?** Two independent estimators put the whole efficiency lane at
   **~2.19 local ≈ 1.26–1.36 LB** against a gold line at **1.48–1.58** that has now been flat for four
   days. P1 is the small, certain part of that (+0.02–0.07 local). P2 (verified-plan batch gating) is
   bounded by the same ceiling and its expected value is explicitly *not derivable from the traces*;
   P3 (frontier-first) is bounded by the oracle and the evidence says most of it will not materialise
   (sp80 visited **198 distinct states in 225 actions** — it was searching, not looping). **The
   remaining ~40% is capability**, and vc33 is the standing proof: 3.00× the human action count on a
   provably minimal, cycle-free, no-op-free path. Nothing currently on the board is a capability lever.
   This question should be put to the 08-16 panel rather than answered in a day session.
3. **What explains cstl's +0.93 single leap when nothing is liftable?** Draw variance is excluded by
   5–7 orders of magnitude, the number is legal and unremarkable in absolute terms, and they have
   published nothing. The honest state is **[UNK]**: what they changed between 08-09 and 08-11, what
   model they run, whether they use the duck/TAAF line at all, and whether 2.52 is a new mean or the
   top of a wider distribution — **one submission cannot distinguish those.** The coherence
   observation on the record (the one self-authored recent repo on the tentatively-identified GitHub
   account is an **LLM inference-optimisation benchmark**, and there is a vLLM fork) is explicitly
   *not* evidence and the identification itself must not be repeated as fact. The usable reading is
   the one that is about us: 2.52 ≈ 71.5% of games clearing L1 at human efficiency sits between "our
   current level depth played at human efficiency" and "half a level deeper everywhere" — i.e. cstl's
   jump is independently the efficiency thesis, at a depth we do not have.
4. **Is the second kernel push slot worth spending today, and on what?** Slot 1 went to P1 and is
   RUNNING. The sweep's own answer is **no, not yet**: all three ADAPTs are modifications to or
   complements of mechanism C, and **building any of them now would confound the one endpoint we are
   currently paying for**. The Prime Agent pull-side helpers are the natural successor **if and only
   if** C's delivery reads well, and are sequenced strictly after P1 is pulled. The other candidate
   uses are the compaction-lane R2 claim (2607.25066 — explicitly does not on its own justify buying a
   build) and the one W0 continuation seed that would take `continuation-v1` from m=2 to m=3 and make
   this family screenable at all — which is the standing blocker on *every* arm we have run this week,
   and is the only spend that changes what a future result can mean rather than what it measures.
5. **Which ledger is authoritative before the next prereg is written?** `runs/ledger.json` reads n=28 /
   bar 1.0848; the log and the cstl file read n=29 / bar 1.0876. The promotion bar is priced off this
   file and preregs are instructed to re-read it rather than cache. Re-derive before anything quotes it.
6. **Does the diagnosis's blind-batch-tail figure need reconciling — 195 or 101?** The prereg says the
   difference is a partition convention and moves the score by <0.3%. If bucket (c) is ever used as the
   *justification* for a mechanism (as it was in the +0.184 framing), the two numbers need to be one
   number first.
7. **Does the animation kill's mis-specified-denominator lesson generalise to the rest of our canary
   set?** K-A3 fired on a static estimate while the measured effect was negative. P1 responded by
   pre-registering **no** token-fraction canary and bounding cost statically instead. That is a
   per-arm fix; whether `SCREEN_PROTOCOL` should bar generated-token denominators outright is not
   decided.

**Pushes: 1 of 2 used (P1, slot 1). $0 cloud spend. No submission change; the queue holds the eternal
fallback and now refills itself.** The one result owed is the P1 arm's canary read, and it may be read
only against M0 — **not against ×1.10.**
