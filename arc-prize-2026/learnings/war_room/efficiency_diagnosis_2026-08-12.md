# Action-efficiency diagnosis — where the 56% actually goes (2026-08-12)

> ## ⚠ CORRECTION NOTICE — appended 2026-08-12, later same day
>
> **The blind-batch tail figure of 195 in this file is NOT REPRODUCIBLE and must
> not be cited.** An independent re-derivation
> (`runs/p1_replay/tail_reconciliation_2026-08-12.md`) rebuilt the buckets from
> the raw event logs with its own reader — validated by exactly reproducing
> `actions_per_level` for all 25 games, the `act` column, dup = 117 and
> re-traversal = 180 — and swept ~1,000 rule variants (batch keying, trigger set,
> dead scope, visited scope, level attribution, RESET handling, precedence).
> **Nothing yields 195.** The nearest variant is **185** (|Δ| = 10), and the P1
> validator's own `baseline_stats` path independently reports **185 / 16.67%**.
>
> **The 101 vs 195 conflict was mostly definitional, but not entirely.** Both
> definitions share the dead-batch rule and the scope (17 cleared levels, 1,110
> actions); they differ only in precedence over bucket (b) — the validator gives
> duplicate `(s,a)` precedence so the buckets **partition** (⇒ **101**, the
> *marginal* saving of batch-abort over the memo), while this file lets the
> buckets **overlap** (⇒ **185** gross, not 195).
>
> **Corrected figures for this file's prose:**
>
> | claim as written | corrected |
> |---|---|
> | 312 (28%) "provably removable" | **218 (19.6%)** — the union, and the right answer to "preventable actions on cleared levels" |
> | 798 (72%) necessary | **892 (80.4%)** |
> | blind-batch tail 195 / 17.6% | **185 / 16.67%** gross (101 / 9.1% marginal) |
> | ar25 L1 117 | **69** |
> | sp80 60 | **53** |
>
> Root cause of the error: column (a) was computed as `act − b − c` over
> **overlapping** buckets, so the 84-action overlap between the duplicate and
> blind-tail buckets was subtracted twice — inflating "removable" and deflating
> "necessary".
>
> **What is NOT affected:** the ×1.10 / +0.184 score arithmetic. The scoring path
> removed each action exactly once (1.8188 published vs 1.8239 re-derived;
> double-counting would have produced 1.8414). **No sealed P1 endpoint depends on
> 195.** The headline conclusions of this file — that the efficiency ceiling
> (~2.19 local ≈ 1.26–1.36 LB) falls **short of gold**, and that the +0.184 was
> itself an accounting artifact — are unchanged and, if anything, strengthened:
> the genuinely irreducible share rises from 72% to **80.4%**.
>
> The body below is left **as originally written** so the record shows what was
> claimed and when. Read every bucket count in it against this table.

**Mandate.** `cstl_breakout_2026-08-12.md` §4.3 found that our 17 cleared levels re-scored at the
human action baseline give **2.549 local ≈ 1.48–1.58 LB**, against our as-run **1.635**, and framed
that 0.914-point gap as recoverable "with ZERO new capability". This file tests that claim against
the traces, action by action.

**Method.** Every action of the animation arm (`runs/kernel_pulls/animation_v1`, 25 games, 5,151
actions, 2026-08-11) was reconstructed from `artifacts/*_p0_events.jsonl`, which carries the full
post-action board, `board_changed`, `reward`, `level_completed`, `analysis_step` and `batch_size`
per action. Actions were replayed into an observed state graph (md5 of the 64×64 board) and
partitioned into buckets. Every score claim is produced by `scripts/phase1_gate.py`'s exact RHAE
mirror, **validated at max abs error 1.776e-15 over 1,000 cross-checks** against Tufa's 500 stored
vanilla runs (null overall 1.600204 = published 1.600204). The event logs reproduce
`benchmark.json`'s `actions_per_level` **exactly for all 25 games**, so the buckets and the score
are computed over the same numbers.

Replicated on two further independent runs (`a22_v2_seed1`, `a22_compaction_v1`). Local CPU only,
read-only, no pushes, no spend.

**Headline: the "zero new capability" framing is wrong. Only ~20% of the gap is bookkeeping.**

---

## 1. THE BUCKET TABLE — all 17 cleared levels

Buckets, as finally defined by what the data supports:

| bucket | operational definition | causally preventable? |
|---|---|---|
| **(a) necessary** | residual — actions that are neither (b) nor (c) | no |
| **(b) provably redundant** | re-executes an `(state, action)` pair already executed in this level. Under a deterministic env this returns **zero new information** | **yes, airtight** |
| **(c) blind-batch tail** | actions inside a multi-action batch that were fired *after* the batch had already gone dead (an earlier action in the same batch no-opped or returned to a visited state). The model never saw a frame for these | **yes, airtight (runner-side)** |
| **(d) post-solution dithering** | actions taken after the level was already solved | **structurally zero — see §1.2** |
| **(e) re-traversal** | action lands on a board state already visited this level. *Cross-cutting diagnostic, overlaps a/b/c* | partly |

```
game   L  act base     x|  (a)nec (b)dup (c)blind (d)dith (e)retrav RST stp amb
tn36   1   29   32  0.91|      29      0        0       0         0   0  14   0
tn36   2   24   72  0.33|      24      0        0       0         0   0  10   0
m0r0   1  138   30  4.60|      88     23       27       0        40   0  41  16
bp35   1  175   21  8.33|     147      8       20       0        12   2  28   0
tu93   1   70   19  3.68|      55      4       11       0         5   1   6   0
tu93   2   20   16  1.25|       9      8        3       0         9   2   5   0
lp85   1    8   17  0.47|       8      0        0       0         1   0   6   0
ka59   1   18   28  0.64|      17      0        1       0         1   0  11   0
vc33   1   21    7  3.00|      21      0        0       0         0   0  14   0
lf52   1   21   32  0.66|      21      0        0       0         0   0  20   0
sc25   1   22   36  0.61|      22      0        0       0         2   0  10   0
sp80   1  225   39  5.77|     165     12       48       0        28   8  62   0
ar25   1  191   32  5.97|      74     45       72       0        53   2  23   0
ar25   2   53   50  1.06|      27     13       13       0        18   0  19   0
sb26   1   13   18  0.72|      13      0        0       0         0   0   6   0
cd82   1   65   55  1.18|      61      4        0       0        11   0  37   4
su15   1   17   22  0.77|      17      0        0       0         0   0   3   0
TOTAL    1110  526  2.11|     798    117      195       0       180
```
`act` = actions spent, `base` = human baseline, `x` = ratio, `RST` = RESETs, `stp` = LLM analysis
steps, `amb` = ambiguous `(state,action)` pairs in this game (latent-state detector).

**Read of the table.** Of the 1,110 actions spent on cleared levels, **312 (28%) are provably
removable** — 117 duplicate `(s,a)` pairs and 195 blind-batch tails. The other **798 (72%) are
genuine, non-repeating, information-producing probes.** The waste is *not* mostly no-op grinding;
it is mostly the agent needing many more distinct probes than a human.

### 1.1 Concentration
Duplicate + blind waste is concentrated in exactly two levels: **ar25 L1 (117 of its 191 actions)**
and **sp80 L1 (60 of 225)**. Eight of the seventeen cleared levels have **zero** removable actions
(tn36 L1/L2, lp85 L1, vc33 L1, lf52 L1, sc25 L1, sb26 L1, su15 L1).

### 1.2 Bucket (d) is structurally empty — kill the "stop-when-solved" idea now
The harness terminates the level on the action that completes it; `actions_per_level` cannot
contain a post-completion action. Checked across all 25 games: **0 actions occur after
`level_completed` within the same level.** Separately, on 16 of the 17 cleared levels the level
completed **on the agent's very first arrival at the winning state** — the agent never "stood on
the answer and walked away". The single exception is **sp80 L1** (arrived at the pre-goal state at
action 152, pressed SPACE at action 225: 72 actions of walk-away). *Stop-when-solved detection has
no addressable surface on this rail.*

### 1.3 RESET does not refund actions — kill "reset and replay clean"
Verified on sp80: the agent's own final batch was `RIGHT, RIGHT, RIGHT, SPACE` after its 8th RESET,
i.e. **sp80 L1 is a 4-action level and the agent executed the 4-action solution**. It still scored
0.143, because `actions_per_level` = 225 counts everything before the RESET. There is no
post-hoc route from search to score.

---

## 2. RECOVERABLE SCORE — exact scorer, every scenario

| scenario | local-25 | Δ | % of the 0.914 gap | → LB @ c=.58–.62 |
|---|---|---|---|---|
| **as-run** (animation_v1) | **1.6352** | — | — | 0.95–1.01 *(= ledger mean 0.9503 ✔)* |
| M2a Brüggen no-op-streak guard | 1.7173 | +0.082 | 9.0% | 1.00–1.06 |
| M2b drop **all** no-op actions (upper bound) | 1.7579 | +0.123 | 13.4% | 1.02–1.09 |
| M3 batch-abort on first no-op/revisit | 1.7531 | +0.118 | 12.9% | 1.02–1.09 |
| M1 full `(state,action)` memo | 1.8099 | +0.175 | 19.1% | 1.05–1.12 |
| **M1+M3 combined (runner-side, airtight)** | **1.8188** | **+0.184** | **20.1%** | **1.05–1.13** |
| M5 one action per LLM decision | 2.1993 | +0.564 | 61.7% | 1.28–1.36 |
| M4 hindsight replay-optimal (**ORACLE**) | 2.1801 | +0.545 | 59.6% | 1.26–1.35 |
| GOLD human baseline on cleared levels | 2.5489 | +0.914 | 100% | 1.48–1.58 |

**Replication of the airtight arm (M1+M3) on three independent runs:**

| run | levels | as-run | M1+M3 | multiplier |
|---|---|---|---|---|
| `kernel_pulls/animation_v1` | 17 | 1.6352 | 1.8188 | ×1.11 |
| `a22_v2_seed1` | 14 | 1.4075 | 1.5627 | ×1.11 |
| `a22_compaction_v1` | 17 | 1.4509 | 1.5794 | ×1.09 |

**Stable ×1.10 (1.09–1.11).** This is the only number in this document that is not an estimate.

### 2.1 The gap decomposes into three very different thirds

| segment | Δ score | share | what it requires |
|---|---|---|---|
| provably-redundant actions | **+0.184** | **20%** | runner-side bookkeeping. No model change. Certain. |
| search-policy improvement up to the oracle | +0.361 | 40% | the agent must reach the goal along the path it *later* proves exists — needs better exploration, **not** guaranteed |
| irreducible with the information observed | +0.369 | 40% | fewer *distinct* probes per level = new capability |

Two independent estimates of the efficiency ceiling converge: the hindsight oracle (**2.180**) and
"one environment action per LLM decision" (**2.199**). **~2.19 local ≈ 1.26–1.36 LB is the realistic
ceiling of the efficiency lane.** The gold line at 1.48–1.58 is *not* reachable without capability.

### 2.2 Per-game — where the 56% lives, and how much of it is reachable

| game | actual | M1+M3 | M5 | M4 oracle | GOLD | gap | % of the 56% | honest verdict |
|---|---|---|---|---|---|---|---|---|
| sp80 | 0.143 | 0.212 | 1.884 | 4.762 | 4.762 | +4.619 | 20.2% | oracle-only. 198 distinct states visited; genuine search |
| m0r0 | 0.225 | 0.404 | 2.550 | 0.455 | 4.762 | +4.537 | 19.9% | **irreducible.** 16 ambiguous `(s,a)` pairs = latent state; oracle recovers 5% |
| tu93 | 3.008 | **5.342** | 6.667 | 6.667 | 6.667 | +3.659 | 16.0% | **best target — 64% recovered by bookkeeping alone** |
| ar25 | 5.022 | **6.562** | 8.333 | 8.333 | 8.333 | +3.311 | 14.5% | **46% recovered by bookkeeping alone** |
| vc33 | 0.397 | 0.397 | 0.893 | 0.397 | 3.571 | +3.175 | 13.9% | **fully irreducible.** Zero dup, zero blind, zero revisits, replay-optimal = 21 = actual |
| bp35 | 0.032 | 0.039 | 1.250 | 0.484 | 2.222 | +2.190 | 9.6% | 78% irreducible |
| cd82 | 3.409 | 3.871 | 4.762 | 4.762 | 4.762 | +1.352 | 5.9% | 34% recovered by bookkeeping |

**tu93 + ar25 supply 87% of the entire airtight gain (+0.160 of +0.184).** sp80, m0r0, vc33 and
bp35 — 63% of the wasted score — are essentially *not* addressable by any efficiency mechanism.

### 2.3 A scoring fact worth writing down
`min(115, (base/act)²·100)` is further capped by `max_weights/total_weights·100`, so a completed
level's contribution **caps at exactly the human baseline**. Beating the human action count is worth
**nothing**. The target is human-equal, never better.

---

## 3. WHAT THE GPT-5.6 PROBE ACTUALLY SHOWS

`runs/gpt56_probe/experiment_full` — `gpt-5.6-sol`, 5 games, `max_actions=100`, 1 pass, 2026-07-16.
Scored with the exact scorer:

| game | arm | levels | score | per-level actions ÷ baseline |
|---|---|---|---|---|
| ft09 | **gpt-5.6** | 5 | 71.43 | 0.47, 0.37, 0.93, 0.76, 0.69 |
| | ours | 0 | 0.00 | — (115 actions, never cleared L1) |
| sb26 | **gpt-5.6** | 5 | 40.19 | 0.83, 1.88, 1.00, 1.00, 0.55 |
| | ours | 1 | 2.78 | 0.72 |
| lp85 | **gpt-5.6** | 4 | 27.78 | 0.21, 0.45, 0.61, 0.83 |
| | ours | 1 | 2.78 | 0.47 |
| vc33 | **gpt-5.6** | 3 | 16.02 | 1.17, 1.46, 1.03 |
| | ours | 1 | 0.40 | **3.00** |
| su15 | **gpt-5.6** | 1 | 1.63 | 1.17 |
| | ours | 1 | 2.22 | 0.77 |
| **mean** | **gpt-5.6** | **18** | **31.41** | |
| | ours | 4 | 1.635 | **19.2× behind** |

**Aggregate trace statistics over the same 5 games:**

| | levels cleared | actions | replay-optimal | redundancy | no-op | re-visit | actions / LLM step |
|---|---|---|---|---|---|---|---|
| ours (Qwen3.6-27B-FP8) | 4 | 767 | 177 | **4.33×** | 6.0% | 10.8% | 4.0 |
| gpt-5.6-sol | **18** | **454** | 411 | **1.10×** | 1.3% | 1.3% | 5.0 |

### Concrete behavioural deltas

1. **GPT-5.6 has no tail.** Its worst per-level ratio across 18 cleared levels is **1.88×**. Ours
   reaches **8.33×**. It either understands a level and clears it at or below human cost, or it
   stops — it never grinds.
2. **Its paths are already replay-optimal (1.10× vs our 4.33×).** It does not try-and-undo.
3. **It collapses to one LLM turn per level once the rule is known** — ft09 L2/L3/L4/L6 and sb26
   L3/L4/L5/L6 were each executed as a **single** analysis step of 7–17 actions. Our agent re-plans
   every ~4 actions forever, including after the rule is known.
4. **On level 1 it is not much better than us.** vc33 L1: 7 actions in 6 steps. ft09 L1: 8 actions,
   50% of them no-ops. Its efficiency is *rule transfer across levels*, which only pays once you
   clear L1 — and we almost never do.

### The one head-to-head on a penalty game, and it is bad news
**vc33 is the only game in both sets.** GPT-5.6 cleared L1 in **7** actions (1.17× baseline); we
took **21** (3.00×). Our 21 actions contain **zero** duplicates, **zero** no-ops, **zero**
re-visits, and the replay-optimal path through our own observed graph is **21** — we walked a
strictly simple path to the goal. **100% of our vc33 waste is "we needed three times as many
distinct probes." No bookkeeping mechanism can touch it.** This is the single cleanest datum in the
file and it argues *against* the efficiency thesis.

Caveat, stated plainly: the probe never played sp80/bp35/ar25/m0r0/tu93, so it cannot test the
efficiency thesis on the five games where our waste actually lives. On three of the four shared
games where we *did* clear a level (sb26 0.72×, su15 0.77×, lp85 0.47×) we were **already better
than human** — and better than GPT-5.6 on two of them. vc33 is the only shared game where we are in
the penalty band.

---

## 4. WHY THE AGENT RE-EXPLORES — the mechanism, from the transcripts

From `prompts/sp80-589a99af_p0.log` and the step-37 transcript at action 129:

- `context_budget_tokens: 31744`, `history_messages: 33`. On a level that ran **225 actions over 62
  analysis steps**, the agent's observation window holds ~33 messages. By action 129 it has
  **forgotten most of what it tried.**
- Its reasoning is full of *"But I already tried clicking on the charcoal and magenta blocks, and
  nothing happened"* — recalled from compacted prose, not from ground truth.
- The harness already exposes **`history`, `transitions`, `last_transition`** as preloaded Python
  globals — the full transition record exists and is queryable. **The agent simply never queries
  it.** It re-derives from a truncated chat window instead.
- Every game terminated `gave_up` after burning **~66–69k tokens** (5,151 actions, 1.63M tokens,
  2h12m). **The binding constraint is the token budget, not the action budget.**
- The harness already has a batch abort path (`stopped_early`), which fired **10 times out of 190**
  batches. The trigger is too narrow.

**Diagnosis: our re-exploration is caused by context truncation, not by a missing loop detector.**
The fix therefore belongs in the runner (a non-truncatable derived state table injected every turn),
not in the prompt. Note our prompt *already* says "Optimize for as few in-game actions as possible"
— and `feedback_prompt_is_noise` says not to iterate there.

### 4.1 Naive batching is a trap — measured, not assumed
Across 40 levels, correlation between **actions-per-analysis-step and log(redundancy) is +0.452**
(and +0.136 for mean `batch_size`). **Our agent batches *more* when it is flailing**, not less:
re86 fires mean batch 21.5 at redundancy 6.05; ar25 L1 mean batch 19.0 at redundancy 12.7; one
batch was **57 actions** fired blind. GPT-5.6's batching is the opposite — it batches only after the
rule is verified. **"Plan-then-execute batching" must be gated on a verified hypothesis or it will
make things worse.** Proposed as-is, it is the wrong mechanism.

---

## 5. RANKED PROPOSALS

Ranked by **verified** recoverable score per build-day. Nothing here is pre-registered; the
diagnosis comes first.

### P1 — Zero-information action suppressor (runner-side) — **#1**
**Mechanism.** In the runner, maintain per level: (i) a `(board_hash, action)` → outcome memo;
(ii) the set of visited board hashes; (iii) a per-game ambiguity counter. Then:
- **A.** If the model requests an `(s,a)` already in the memo **and** the game's ambiguity counter
  is 0, do not spend the action — return the memoised outcome and a one-line note.
- **B.** Abort the remainder of a multi-action batch the moment an action no-ops or lands on a
  visited state (generalise the existing `stopped_early` trigger, which fires on only 10/190
  batches).
- **C.** Inject a compact, non-truncatable "untried primitives at the current state / states visited
  / dead `(s,a)` pairs" block into every turn, computed from `transitions` — the data the agent
  already has but never queries.

**Buckets drained:** (b) 117 actions and (c) 195 actions on cleared levels.
**Expected Δ:** **+0.184 local (1.6352 → 1.8188), ×1.10**, replicated at ×1.09–1.11 on three runs.
LB ≈ **0.95 → 1.05–1.06 mean**. Concentrated in tu93 (+2.33) and ar25 (+1.54).
**Build:** ~1 day, harness-only, no model or prompt change.
**Canary:** duplicate-`(s,a)` rate on cleared levels must fall 10.5% → 0%; blind-batch tail 17.6% →
0%; `levels_completed` on the local 25 must stay ≥ 16 (currently 17).
**Kill rule:** revert if `levels_completed` ≤ 15 on any local 25-game run, or if any game whose
ambiguity counter > 0 loses a level. Suppression A must stay disabled for the **8 of 25 games that
show latent state** (game-level ambiguous `(s,a)` pairs): m0r0 55, re86 19, sk48 11, ka59 10,
cd82 8, g50t 4, dc22 3, wa30 2. The other 17 games are fully deterministic in the observed data.

### P2 — Verified-plan gating of batch size — **#2**
**Mechanism.** Two regimes. **Unverified** (no confirmed rule for this level): hard cap of 1
environment action per LLM decision — every action must be an observation. **Verified** (the model
declares and the runner confirms a plan whose first k steps match predicted transitions): allow the
full batch. This is the GPT-5.6 pattern (1 LLM turn per already-understood level) and it is the
*opposite* of what we do now.
**Bucket drained:** the search-policy third — the gap between M1+M3 (1.819) and the M5/M4 ceiling
(~2.19).
**Expected Δ:** ceiling **+0.38 on top of P1**; realistic expectation is well below that and it is
**not** derivable from the traces — the counterfactual (would the agent find the rule in the same
number of hypotheses?) cannot be read off a log. **Genuine downside risk:** at a fixed ~68k
token/game budget, de-batching cuts actions-per-token ~3.6×, which may cost levels.
**Build:** ~3 days.
**Canary:** actions-per-analysis-step on unverified levels → 1.0; correlation between batch size and
redundancy (currently **+0.45**) must go to ≤ 0.
**Kill rule:** revert if `levels_completed` drops at all, or if mean RHAE falls below the P1 arm.

### P3 — Frontier-first exploration with known-path replay — **#3**
**Mechanism.** Go-Explore on the observed transition graph: keep an archive of states with untried
primitives; when the agent wants to explore, route it to the nearest frontier state along the
*known shortest* path rather than letting it wander; try all 5 primitives at each newly reached
state before moving on.
**Bucket drained:** (e) re-traversal — **180 of 1,110 actions (16.2%)** on cleared levels; 821 of
5,151 (15.9%) run-wide; 77 RESETs.
**Expected Δ:** bounded above by the oracle (+0.545 total, so ≤ +0.36 on top of P1). But the
evidence says most of it will **not** materialise: sp80 visited **198 distinct states** in 225
actions — it was searching, not looping, and "try all primitives at each new state" only pays on
sp80 (~+0.07 local). The exhaustive-primitive variant is cheap to bolt onto P1.
**Build:** ~2 days for the primitive sweep; ~5 days for full frontier routing.
**Canary:** re-traversal rate 16.2% → < 5% without a fall in distinct states visited per level.
**Kill rule:** revert if distinct-states-per-level falls (that would mean we suppressed search, not
waste).

**Not proposed, and why.** *Stop-when-solved detection* — bucket (d) is structurally zero (§1.2);
there is nothing to detect. *Reset-and-replay-clean* — RESET does not refund actions (§1.3).
*Prompt-level "be efficient" instructions* — already present in the system prompt, and
`feedback_prompt_is_noise` applies.

---

## 6. HONEST BOTTOM LINE

1. **The thesis is 20% right.** `cstl_breakout` §4.3's 0.914-point gap is real and correctly
   computed, but its "with ZERO new capability" reading is not supported. Only **+0.184 (20%)** is
   bookkeeping. **+0.361 (40%)** requires a genuinely better search policy with no guarantee it
   works. **+0.369 (40%)** requires needing fewer distinct probes — that *is* capability.
2. **The efficiency lane's realistic ceiling is ~2.19 local ≈ 1.26–1.36 LB**, from two independent
   estimators. That is worth having — it would turn our best-ever draw (1.33) into our expected
   value — but **it does not reach the 1.48–1.58 gold line.**
3. **72% of the actions on our cleared levels are non-repeating, information-producing probes.** We
   are not mostly grinding no-ops; we mostly need too many looks.
4. **vc33 is the counter-example that should be kept in view:** 3.00× the human action count with a
   provably minimal, cycle-free, no-op-free path. Some of our inefficiency is simply a 27B model
   needing more evidence than a human, and no runner patch will fix it.
5. **Do P1 anyway.** It is one build-day, harness-only, replicated ×1.10 across three runs, has an
   airtight canary, and carries no model risk. It is the highest verified score-per-day item on the
   board. Everything after it is a bet.

---

### Provenance
- Scorer: `scripts/phase1_gate.py` `rhae_score`, validated 1.776e-15 / 1,000 cross-checks vs
  `runs/tufa_example_run/{benchmark,score}.json`.
- Primary trace: `runs/kernel_pulls/animation_v1/` — `benchmark.json`, `artifacts/*_p0_events.jsonl`,
  `prompts/*.log`, `summary.txt`.
- Replication: `runs/a22_v2_seed1/`, `runs/a22_compaction_v1/`.
- Comparison arm: `runs/gpt56_probe/experiment_full/` — `benchmark.json`, `run_config.json`
  (`model: gpt-5.6-sol`, `max_actions: 100`, `n_passes: 1`), `artifacts/*_p0_events.jsonl`.
- Local→LB calibration c ≈ 0.58–0.62 from `runs/lb_process_model/report.md`, as used in
  `cstl_breakout_2026-08-12.md` §4.4.
- All analysis local CPU, read-only. No pushes, no submissions, no spend.
