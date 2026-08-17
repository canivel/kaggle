# TASK — DAILY SWEEP 2026-08-17 (discussions + research), ARC-AGI-3 campaign

Repo: `F:\kaggle\arc-prize-2026`. **Read-only / research only. DO NOT push a Kaggle kernel,
DO NOT submit, DO NOT spend money.** Output is a written report.

## CONTEXT YOU MUST READ FIRST (in this order)

1. `learnings/war_room/conversion_trace_2026-08-17.md` — yesterday's decisive finding: the
   field's 2.5+ recipe is a public, CC0, default-OFF score-mechanics graft stack
   (`taaf_grafts`, in dataset `thtennant/taaf-kaggle-source-share-fork`) that attacks the
   quadratic action denominator of `min(115, (baseline/actions)**2 * 100)` — NOT the new
   Qwen3.8 engine (engine alone = LB ~1.71, and our own sealed eval REFUTED it 2x).
2. The `### 2026-08-17` section of `ITERATION_LOG.md` — today's board state and two named
   instrument defects.
3. `learnings/daily_brief_2026-08-16.md` — yesterday's brief, for the "what changed" delta.

Board state for framing: we are **#175 of 2365 at 1.33** (unchanged since 07-18), gold/top-13
line **2.00**, top-5 prize line **2.33**, new #1 Lord Han Solo 2.76. We fell 45 ranks in one
night without changing a byte. Ledger n=34, mean 0.9382, s 0.1559, promotion bar 1.0777.

## PART A — DISCUSSIONS SWEEP (protocol step 1b)

Find every NEW post in the competition discussion feed since **2026-08-16** (the 08-17
conversion trace already swept topics 735590, 735479, 735381, 735243, 735147 and found NO
disclosure of the graft channel — your job is what is new SINCE that sweep, plus anything it
missed).

Route options: `uvx --from kaggle==2.2.2 kaggle competitions topics arc-prize-2026-arc-agi-3`
(2.2.2 is the version that has the forums subcommand; 2.0.0 does not); WebSearch as a
cross-check. Note that a browser route has previously been a dead end for this competition.

For EACH new post: one line of `ADOPT` / `ADAPT` / `IGNORE` **with a reason**. Specifically
flag, as a priority: any post that discloses, hints at, or complains about banking / replay /
reset-after-win / action-efficiency / score-formula mechanics, and any host statement about
whether such mechanics are within the rules. Quote host posts verbatim if any exist.

## PART B — RESEARCH SWEEP (protocol step 1c)

Check for genuinely new (past ~7 days) papers/results in our active fields: LLM agents on
interactive benchmarks, ARC-AGI-3 specifically, test-time learning/TTT, agentic harnesses,
banking/replay/trajectory-reuse strategies, and action-efficiency or reward-hacking-adjacent
work on agent benchmarks. Use WebSearch + arXiv listings (cs.AI / cs.LG new submissions).

Same discipline: `ADOPT` / `ADAPT` / `IGNORE` + one-line reason each. Be ruthless — this
campaign's standing finding is that most sweep items are IGNORE (a recent sweep was
0 ADOPT / 3 ADAPT / 12 IGNORE) and inflating relevance wastes the coordinator's slots. Do not
pad. If nothing is adoptable, say so plainly; "nothing new that changes the plan" is a
perfectly good and useful answer.

One specific research question worth a targeted look, because it is now our own open problem:
**trajectory pruning + replay for agent action-efficiency** — is there published work on
pruning a successful agent trajectory to its state-changing actions and replaying it? If yes,
it tells us how well the technique is understood and whether known failure modes exist.

## OUTPUT

Write to `F:\kaggle\arc-prize-2026\duck_eval\graft\sweep_2026-08-17.md`, structured:
`## A. DISCUSSIONS` (table: post / date / author / one-line content / verdict + reason),
`## B. RESEARCH` (same shape), `## C. DOES ANYTHING CHANGE THE PLAN?` (2-4 sentences,
explicit: yes/no and what), `## D. OPEN QUESTIONS FOR THE BRIEF` (bullets).

Tag load-bearing claims **[V]** verified by direct read this session · **[INF]** inference ·
**[UNK]** unknown. Do not fabricate a post, a date, an author, or a paper — if a search
returns nothing, report that it returned nothing. An empty sweep honestly reported is far
more valuable than a plausible-looking invented one.
