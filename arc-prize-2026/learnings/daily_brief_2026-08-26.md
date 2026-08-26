# DAILY BRIEF — 2026-08-26 (Wednesday)

Prepared by the iterate session. Morning-check instrument output is in `ITERATION_LOG.md`
under the 06:00 stub and is **not** repeated here; this brief carries the *validated
interpretation*, per the 2026-07-16 directive.

**Headline.** The exec-WM read came back NULL and told us exactly where it broke, a
blocking instrument challenge against it was raised and **discharged on the artifact**,
P2 shipped as slot 1 with its "can this trigger even fire?" question answered *before*
the build, and the research sweep found a paper that is the exec-WM thesis executed at
**97.8%** — at a token budget roughly **three orders of magnitude** beyond our rail.

---

## 1a. RESULT DEEP-DIVE — EXEC-WM v1, board draw 1.05

Full read: `learnings/war_room/execwm_seed1_read_2026-08-26.md`. KAOS exp **53**
(canonical; supersedes 50). Scorer verdict **CERTIFIED**.

### The pre-registered expectation was NOT met, and the arm was NOT killed

| stat | value | sealed band |
|---|---|---|
| **`lc_total`** | **25** | HARM ≤23 · **NULL 24–34** · SIGNAL ≥35 → **NULL** |
| `trim1` | 2.330 | — |
| `mean_score` | 3.006 | (retired as primary) |
| board draw | **1.05** | field-floor config n=5 mean **1.5760** sd 0.2713 |

lc 25 is **−1.43σ** against the re-derived comparator (29.0, pooled sd 2.80) — inside
the null, indistinguishable from the config it wraps. **The pre-stated decisive kill did
not trigger** (it required `levels_cleared_by_plan = 0` *and* lc ≤ 23; actual 1 and 25).
The v1 class is not dead by its own rule.

### What went RIGHT, and it is a first for this campaign

`ls20` level 1 was **cleared BY PLAN on the scored rail with `llm_calls: 0` and
`llm_tokens: 0`** — 4 move rules mined and verified at **precision 1.0** (n = 33/21/23/19).
The CPU proof transferred to Kaggle *exactly*. This is our first level cleared by
deterministic search with no LLM in the loop. Level 2 then mined the same rules on far
more evidence (n = 279/173/17/121) and fell back only on `plan-budget-exhausted` after
96 plans.

### What went WRONG — and it is NOT what the obvious v2 would fix

Fallback rate **96.875%** (31 of 32 in-scope levels). The taxonomy:

| stratum | levels |
|---|---|
| **zero candidate rules mined at all** | **14** |
| candidates mined, **every one refused** by the verifier | **9** |
| verified + planned, did not clear | 5 |
| verified + planned + **CLEARED** | **1** |

Probe histogram across the 32 levels: `{4:2, 8:2, 16:11, 18:1, 20:16}`. **The 14
zero-rule levels spent their FULL 16–20 probe budget and produced nothing.** So the
binding defect sits **upstream of the planner and upstream of the verifier** — it is
object identification and rule-class coverage. "Raise the probe budget" is
arithmetically **closed** as a v2, and that is pre-registered in the read so it cannot be
proposed later.

The verifier is the one component demonstrably not fooling itself: **24 rules verified,
29 rejected**, refusing every candidate on 9 levels.

### D1 failed on a GATE DEFECT — and is recorded FAILED anyway

25/25 games **ARMED**; exactly 7 self-disabled `no-keyboard-actions` (`ft09, lp85, r11l,
s5i5, su15, tn36, vc33` — click-only, outside the v1 movement class by construction).
D1 counted that **correct refusal** as a delivery failure. The gate is **not** rewritten
post-hoc to convert its own FAIL into a PASS; what is recorded is the rule for the next
prereg: *count arm-reachability, not report-presence, whenever the arm has a legitimate
self-disable path.* Companion defect: `execwm_score.py` reports `disabled_games: 0`
because it counts only over **present** reports — it undercounts the exact quantity D1
depends on.

### ★ BLOCKING CHALLENGE RAISED AND DISCHARGED

KAOS exp **49** (filed 06:00) ruled the 1.05 read **BLOCKED**: `GameState.frame` returns
`raw.frame[-1]`, and at a level-**completing** step `raw.frame` holds ≥2 layers where
`[-1]` is the **next level's opening board**. Our settled-frame reader takes `[-1]`, so
PHASE P would mispredict at every clear, be charged a BREAK, and latch fallback at
`MAX_BREAKS_PER_LEVEL = 3`. The defect is real — published in `thtennant/arc3-duck-v25`
and **independently replicated on our own archive across 6 games**.

**The check, run on the pull:** the defect can only fire at a level-completing step.
All **5** breaks in the entire run landed on games with engine `levels_completed = 0`
(`ka59` 3, `sk48` 1, `sp80` 1) — there was no level clear for it to fire on. The one
level that *did* clear by plan recorded `breaks: 0`.

**VERDICT: real defect, 0 of 5 breaks and 0 of 31 fallbacks explained by it.** The block
is discharged on the artifact, not on argument. §5 stands untouched — 23 of 32 levels
failed upstream of any prediction check. **The reader must still be fixed in v2**, because
the hazard is live for any arm that actually clears levels; the reason is the next run,
not this one.

> **Lesson (exp 49's, and it is right):** the 08-24 brief cleared this same accessor —
> "our lineage already reads `raw.frame[-1]`, audited, no defect" — on a test that
> exercised animation strips **mid-level** and never exercised a level **transition**,
> which is the one place where reading `[-1]` *is* the defect. Record which transitions
> an audit covered; treat the uncovered ones as **unaudited**, not clean.

---

## 1b. DISCUSSIONS SWEEP — 3 posts read (CLI route; the browser route stays dead)

**`737617` "Sudden increase in top 3 teams?" (Drona Bajaj, TODAY 11:51 UTC, 0 comments).**
*"the top 3-5 teams have increased by a very good amount... almost simultaneously. Some
new release, or is this entirely random?"* → **ADOPT as corroboration, nothing more.** An
outside observer, on their own instrument, saw exactly what our LB diff measured this
morning: prize line +0.29 to 3.17, and nine single-draw steps ≥ +0.99/draw. It confirms
the *observation* is not an artifact of our differ. It supplies **zero** method
information — nobody named a model, bundle or technique. Evidence class **UNKNOWN**, and
the standing scope rule still binds: our instrument sees Score/SubmissionCount/Rank/
TeamName/LastSubmissionDate and nothing else.

**★ `736578` "Public vs. Private Discrepancy" (Nick Pellegrin, 08-21, 4 votes).** The most
important post on the board for us. He reports: Tufa duck harness + qwen3.8 → **local
2.1%, public LB ~1.4%**. His *own* harness + qwen3.8 → **local 5.0–5.4%, public LB still
~1.4%**. A **~2.5× local gain bought exactly zero board movement.** → **ADOPT as a
standing warning against our own screening strategy.** This is an external, independent
report of the decoupling we are exposed to, and it lands on a day when we screened an arm
locally and shipped it. It prompted **O1**, which I then ran against our own archive — and
the answer (§5) is that **no config has ever screened above the certified floor AND
produced a board draw**, so we have no calibration point in the region every promotion
gate is defined over. Not evidence the mapping is flat; evidence it is **untested**.

**`736540` "non-official games for training" (robenten, 08-21, 4 votes).** Links
`github.com/theredbluepill/arc-interactive` → **ADOPT (highest-value community find in
weeks).** See 1c.

---

## 1c. RESEARCH SWEEP

### ★★★ arXiv 2608.14490 — *Twin: Playing an Unknown Game with a Test-Time Digital Twin*

**This is the exec-WM thesis, executed, at 97.8%.** A frontier coding agent writes an
**executable program** modelling the game's transitions; no scored action is issued until
the program **replays every logged transition exactly**; planning is search inside the
validated model.

- **179 of 183 levels (97.8%)**; **23 of 25 games** on the public subset.
- Base model playing directly: **7.8%** → with the twin: **93.3%**.
- Beats humans on action efficiency on **158/179** levels.
- Validation rule: `∀(s,a,s′)∈D, T̂(s,a)=s′` — exact, cell-by-cell on the 64×64 grid.
- *"Building a usable world model is simpler than anticipated, whereas the harder problem
  is inferring the right goal."* First proposed goal is correct on **156/179 (87.2%)**,
  via five visual-change signals: colour appearing, colour disappearing, localised
  change, scene shift, frontier novelty.
- **Click games: "supply a shortlist of candidate coordinates to keep the branching
  factor finite"** — clicks as discrete BFS positions, not sprite movement.
- The two games it failed outright: **`sc25`** (hidden countdown) and **`sp80`** ("a goal
  wall rather than a dynamics wall").

**THE COST, AND WHY THE HEADLINE IS NOT AVAILABLE TO US.** Model: OpenAI Codex on GPT-5.6
Sol. **2.60 billion processed tokens across 25 games. ~224,000 tokens per scored action.
91.4 hours of wall-clock inference** (0.7 h on sb26 to 17.8 h on ka59). Our rail: one 27B
FP8 on one GPU, a hard **7,920 s per game**, and ~1,639 actions per 25-game run at ~1,452
tokens per generation. **Twin's per-action budget is ~3 orders of magnitude beyond ours,
and its wall-clock alone exceeds our entire per-run allowance.** This is the ARChitect
pattern again — a headline produced under a compute regime we do not have — and it must
not be quoted as though the 97.8% were liftable.

**Verdict — split the paper, take the cheap half:**

| component | call | why |
|---|---|---|
| Goal proposal by 5 ranked visual-change signals | **★ ADOPT** | Deterministic, **zero LLM tokens**, and it attacks the part the paper says is *hard*. Our v1 used rare-colour goals only — a subset of one of the five. `dc22` died on `plan-targets-exhausted`, i.e. a goal failure. |
| Click-coordinate shortlists in BFS | **★ ADOPT** | Directly addresses the **7 games we auto-disabled** as `no-keyboard-actions` — 28% of the benchmark that our arm currently refuses on principle. Widens the rule class exactly as the exec-WM read pre-registered. |
| Exact-replay validation (`∀` transitions) | **ADAPT, cautiously** | Stricter than our precision ≥0.90 + break budget. But our failure was **zero candidates on 14 levels**, not lax acceptance — a stricter rule cannot fix an empty hypothesis set, and under a 1/43 world event it would be sudden death (a design ruling we already earned on the real simulator). Adopt the *idea* of exact replay on a masked interior; do not adopt unconditional `∀`. |
| Frontier coding agent authoring the program | **IGNORE** | 224k tokens/action. Not our rail. |

**Convergence worth naming:** the read written this morning pre-registered that a v2 must
widen the rule class to *"click-addressable objects, non-constant deltas, multi-object
dynamics"* — **before** this paper was read. Twin independently says the same, and names
the click shortlist as the mechanism. Two independent routes to the same next step is the
strongest signal in today's brief.

**Also noted, no action:** `sp80` is a goal wall for the 97.8% system *and* one of our two
`prediction-breaks` games — evidence its difficulty is intrinsic, not our defect.

### ★ `github.com/theredbluepill/arc-interactive` — 249+ community games, MIT

**249+ playable ARC-AGI-3-style games** on an `ARCBaseGame` framework with a
**competition mode matching official toolkit rules**, human and agent modes, MIT licence.
→ **ADOPT as a screening instrument, on a strict condition.** This addresses two standing
defects at once: the **SCREEN-SHAPE DEFECT** (our rail screens 1 clone of 25 games while
the official benchmark is 110 runs = 25 games × ~4.4 clones) and the **PRIORITY
generalisation rule** (private LB has more games; public-LB luck-chasing is overfitting).
An exec-WM v2 whose thesis is "widen the rule class" needs games it has **never** been
tuned against, and 249 of them are free and CPU-only. **Condition: it is a
*generalisation* instrument, never a promotion gate** — these are community games, not the
private set, and no sealed band may be defined on them.

### Swept, no change
`2607.00627` (AGI Maze), `2606.30639` (self-evolving world models), `2603.17683` (Sensi
curriculum TTL) — surveyed, none supplies a reproducible mechanism at our token budget.
**IGNORE.**

---

## 2. TODAY'S BUILD — P2 shipped as slot 1

`canivel/arc3-p2-retry-eval` v1 pushed **08:44 EDT**, RUNNING. KAOS exp **54** (canonical).
Full gate: `learnings/war_room/p2_trigger_fireability_2026-08-26.md`.

**The blocking question was answered before the build, not after.** The 08-25 handoff
ordered: prove the H=4 counter can fire against retained histories *first*, because
`hard_noop_guard` shipped armed and blocked **0 of 5,255** real actions. Measured on real
artifacts already on disk:

| corpus | fires at H=4 | bar |
|---|---|---|
| **`q38_field_v1`** (P2's own vehicle) | **19/25** | ≥15/25 |
| `budget_t3_v1` | 23/25 | |
| `p1_notes_v1` | 19/25 | |
| `execwm_v1` | 19/25 | |

Margin is not fragile: 15 of 25 field-floor games have `max_stuck_run ≥ 7`, so H would
have to exceed **7** before delivery fell to the bar. **Negative control:** 6/25 games
correctly REFUSE, and they are exactly the prompt clearers. The turn reconstruction was
validated against an independent instrument and reproduces it **exactly** (424/424 acting
turns, 17.0/game).

**Priced honestly:** `sb26` is one of the refusers **and carries 50.4% of the field
floor's entire `mean_score`**. P2 cannot lift our best game by construction; its upside is
capped to the 19 stuck games.

**Two silently-dead-arm bugs found and fixed pre-push** — the class this campaign keeps
paying for:
1. The counter would have keyed on `state_path.parent`, which the shipped layout can
   **share across all 25 games**; the cleared-level count would have accumulated
   benchmark-wide and **permanently disabled retry after the 4th clear anywhere**.
2. The D2 report existed only on stdout — unevaluable under the **P1 0-byte-log** class.
   Now flushed per game to a job-dir file.

**D2 is the risk, and it is instrumented, not inferred.** P1 delivered at 96.3% and got
**1.3% use** against a 30% bar, and its read was unevaluable because nothing counted
calls. P2 counts real `attempt()` **calls by AST**, split by whether the affordance was
armed.

**Gates:** `local_gate --arm p2 --full` **PASS 57/0** · episode smoke 18/18 · trigger
smoke 50/50 · scorer selftest 33/33 (healthy positive control, 9 cross-arm refusals, 6
**real** foreign artifacts refused) · **`p2_cell_smoke` 20/20 executing the REAL notebook
cell off-Kaggle** with four loud-death negative controls. **Pull-back:** metadata EXACT,
and the remote notebook **minus the inserted patch cell is byte-identical to the certified
floor's own remote copy**.

**Tonight's head rule is SEALED PRE-DATA** — `learnings/war_room/p2_head_rule_2026-08-26.md`.

---

## 3. OPEN QUESTIONS

**★★ O1 (new — and it was ANSWERED today; see §5).** *Does our local screen predict the
board above the floor?* An outside team reports a ~2.5× local gain buying zero board
movement (`736578`). Run against our own archive, the answer is worse than "unknown": we
have **never put a data point there at all.** `scripts/local_vs_board.py`, §5 below.

**O2. Can a v2 rule class be widened *cheaply*?** Twin says the world model is the easy
part and the goal is hard — but it says so with 224k tokens per action. Our v1 says the
model is the hard part at ~0 tokens per action. Both can be true: the model is easy *if
you can afford a frontier coder*. The open question is whether the two ADOPT components
(5-signal goal ranking, click shortlists) are enough to move levels on a 27B rail, and
that is testable on `arc-interactive` for free.

**O3. Standing.** `hard_noop_guard` is still armed and has still never fired. Untouched
today.

---

## 4. LEDGER + BOARD (re-derived, not cached)

`runs/ledger.json`: **n = 37, mean 0.9316, sd 0.1771, promotion bar 1.089**, `latest_date`
**2026-08-20** — it tracks the **retired frozen-fork null** and is *not* the live
statistic. Field-floor config: **n = 5, mean 1.5760, sd 0.2713**, draws
[1.59, 1.58, 1.63, 1.16, 1.92]; the banked **1.92 is a max of five**, not a level. Board:
**#159 of 2546 at 1.92**, −13 ranks on a flat score; prize 3.17, gold 2.66.

---

## 5. ★★ O1 ANSWERED — the local→board mapping is UNCONSTRAINED above the floor

Instrument: `scripts/local_vs_board.py` (read-only, writes nothing). It joins every
pulled artifact's local screen to the board draw that artifact actually produced.

**36 pulled artifacts carry a `benchmark.json`. Exactly 2 of them have a known board draw.**

| pull | lc | trim1 | mean | BOARD |
|---|---|---|---|---|
| `q38_field_v1` (certified floor) | 28 | 3.189 | 6.173 | **1.16** |
| `execwm_v1` | 25 | 2.330 | 3.006 | **1.05** |

**Two artifacts have screened ABOVE the certified floor. Neither was ever submitted.**

| pull | lc | trim1 | board | why not submitted |
|---|---|---|---|---|
| `budget_t3_v1` | **35** | 5.021 | — | 3× budget: not a legal submission config |
| `private_base_v1` | **30** | 4.732 | — | private arm, deliberately held per order |

Both omissions are individually defensible. The **consequence** is not:

> **Number of configs that have ever screened above the floor AND produced a board draw: 0.**

**What this does and does not say.**

- It does **not** show the mapping is flat. It shows the mapping is **untested** in the
  only region a promotion decision ever cares about. With n = 2 matched pairs — both *at
  or below* the floor — there is no regression to run, and the script deliberately
  **refuses to fit one** rather than quote a slope it cannot support.
- The two pairs we do have are at least directionally coherent: lc 28 → 1.16 and lc 25 →
  1.05, i.e. lower local, lower board. That is consistent with a positive mapping and is
  also consistent with almost anything else at n = 2.
- **Correction to an earlier draft of this brief:** I wrote that "every anchor in the
  R² = 0.990 `trim1` fit sits at the floor." I could not find that fit's anchor list
  anywhere on disk, so that claim is withdrawn — the fit may well be sound over a wider
  range than I assumed. What is verified is the table above, and it stands on its own.

**Why this outranks the build rail.** Our screening gates promote on `trim1`/`lc` bands.
Every band we have sealed — exec-WM's SIGNAL ≥ 35, P2's kill at lc ≤ 21, the comparator
mean 29.0 — describes a region where **we have zero board observations**. We are
calibrating a promotion decision on a curve we have never sampled. The competitor's
report is the outside version of the same hazard, and it is why the sealed rule
`project_arc_final_selection_rule.md` (pick by CONFIG MEAN, never public max) matters more
than any single screen.

**Concrete next action (cheap, no GPU, no slot):** `private_base_v1` screens lc **30** —
above the floor's 28, on a legal config, already built and pull-back-verified. **It is the
one artifact we hold that could put a point in the empty region.** Submitting it on a
future nightly window would cost one draw and buy the first calibration point above the
floor that this campaign has ever had. That is a coordinator/private-lane decision, not
this session's to take — the private arm is another lane's, and its lock is explicit that
pushes happen per its own prereg. **Recommended for the Sunday panel agenda.**
