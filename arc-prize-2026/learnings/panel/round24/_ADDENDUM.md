# R24 PACKET ADDENDUM — circulated 2026-08-09, same day as the panel

This addendum is **not** part of the sha256-hashed proposal body. It carries evidence that
landed on the morning of the panel, after the proposal was written (2026-08-08). Treat it as
first-class panel input: where it contradicts the proposal, say so explicitly in your review.

---

## A1. Control ledger — one more draw

The 2026-08-09 overnight draw of the frozen fork scored **0.89** (verified COMPLETE via the
Kaggle API). Control ledger is now **n=26, mean 0.9365, s 0.1540** (the proposal §6.5 quotes
n=25, 0.9384, 0.1569 — it is one draw stale). The draw is interior (z ≈ −0.31); the
watch-rule fired-and-resolved on 2026-08-07 stays resolved-STATIONARY and re-arms only on a
fresh consecutive sub-0.80 pair.

Recomputing §6.5's illustrative promotion arithmetic at n=26:
`0.9365 + t(0.95, df=28) × 0.1540 × sqrt(1/4 + 1/26)` = `0.9365 + 1.7011 × 0.08276` = **1.0773**
(vs 1.0823 at n=25). Still illustrative, still not a sealed gate — no scored draw is requested.

## A2. Leaderboard — the gold cutoff moved for the first time in five days

Top-13 (gold) line rose **1.56 → 1.58**, ending four flat days. Helmut AGI entered at 1.61
(#7); the four-name 1.58 pack slid to #10–13, evicting three 1.56 entries to #14–16. Top-5
prize cutoff holds 1.61. Head static (KOJIMA 1.86 resubmitted unchanged). Our 1.33 remains
below #49; **gap to gold is now 0.25**. Archived: `runs/lb_daily/lb_2026-08-09.csv`.

## A3. Discussions sweep (2026-08-09) — one directly on-substrate report

Full sweep: `learnings/sweeps/discussions_2026-08-09.md`. Two new topics since 08-07.

**733865 — "RPS ARC-AGI 3 Solutions Technical Report", Jason Feng, 2026-08-08.** Three
solutions built **on the Tufa Duck harness with Qwen3.6-27B — our exact substrate**, installed
via notebook-level runtime hooks. His "Tiger" solution combines within-level working memory,
**persistent cross-level memory**, and a surprise-driven proposer. His "Sandwich" solution uses
an intercepted-consultation channel.

Why this matters to the decision in front of you:
- It is the **only** ARC-AGI-3 harness work found in three sweeps that runs at **27B rather
  than a frontier model**. It therefore retires the "is this even implementable at 27B?"
  question for the proposal's **P3 durable cross-level memory arm** — on design grounds only.
- It suggests a **refinement to P3**: two-timescale memory (consolidate a cross-level policy at
  first level-clear, then *refine rather than rewrite*) instead of the proposal's flat
  `_summarized_knowledge` un-wipe.
- His memory stays compact by **prompting, not symbolic truncation** — weak corroboration of
  the campaign's generation-over-selection reading of the A22 death (§2.2).
- Sandwich's intercepted consultation is a working prior for the proposal's L4 consult gate.
- He discloses LLM (Codex) co-authorship of his solution — a **live external precedent** for
  the §5.3.1 governance ruling on workstation LLM authoring.

**Severe de-rating, applied per the proposal's own §2.1 rule:** Feng ranks **177th — below our
own 1.33**. His §6/§8 state plainly that there is **no quantitative comparison, no ablations**,
and Tiger confounds four mechanisms at once. This is **design evidence with zero efficacy
evidence.** It must not be cited as reason to believe the mechanism *works*; only as reason to
believe it is *buildable on our substrate*. Reviewers should push back if the packet elsewhere
overreads it.

**733697 — fresh-kernel-slug fix, Antoine Matemane Mahirwe, 2026-08-07.** Third-party
confirmation of our existing `feedback_fresh_kernel_slug` rule: seven generic `system error`
submissions on an iterated slug, including a faithful rebuild of known-good code; a brand-new
slug worked first try. Operational, ADOPT. Two free harness checks it surfaces for before any
P1 push: Swarm hardcodes `record=True` (~1 GB JSONL per game) and the base `Agent` never prunes
`self.frames`. Unconfirmed n=1 claim (flag, do **not** act on): ERROR submissions may not count
against the 1/day limit.

Monitors unchanged: borro1980's merge solicitation has had zero uptake from all five 1.47–1.58
targets since 08-05.

## A4. Weekly maintenance (Sunday items, run before the panel)

Full record: `learnings/weekly/weekly_2026-08-09.md`.

- **KAOS dream (run_id=8, dry_run):** 3 episodes, 221 memories scored, 0 skills scored.
  Consolidation proposals, verbatim: *"No structural changes proposed this cycle. Library is
  stable."* / *"Nothing obviously wrong. Library is warming up."* **Nothing for the R24 agenda.**
  Hot memory is pure recency (all 0 hits). Digest: `Dreams/2026-08-09-122435.md`.
- **Failure fingerprints:** 16 incidents, 8 recurring families. **Newest incident is
  2026-07-08 — no new failures in ~4.5 weeks under the preflight regime.**

```
family                         n  first       last
--------------------------------------------------------------
class:ERROR:none               7  2026-05-26  2026-06-28
provenance:scratch-built       5  2026-05-26  2026-06-28
slug:canivel/arc3-final        4  2026-05-26  2026-06-10
class:COMPLETE:0.00            3  2026-03-29  2026-06-10
slug:canivel/arc3-forge35      3  2026-04-24  2026-06-22
slug:canivel/arc3-pilot-eval   3  2026-07-07  2026-07-08
t1:07d0f5248c48401d            3  2026-07-07  2026-07-08
class:COMPLETE:null-band       2  2026-06-01  2026-06-08
```

Bearing on the decision: all three top families are **build/provenance-infra** modes, not
agent-algorithm modes. They say nothing about any lane's mechanism — they bound **execution
risk at S2** (the one push), where the proposal's kernel-side artifact code would sit on the
same `provenance:scratch-built → ERROR` path that produced five incidents. Note also that the
banking lane's claimed root cause (N5 `prune_trace` phase desync → step-0 frame divergence)
has **no fingerprint in the store at all** — neither corroborated nor refuted by this evidence,
and it carries the largest diff surface of the three candidates.

## A5. Panel-mechanics disclosure (methodology reviewer: this is for you)

The R24 panel took **two rail failures** before reaching you, and both are disclosed because the
panel's own independence and comparability are methodological premises.

1. **KAOS rail failed.** The normal `fable-panel` rail was attempted first and **all five
   reviewers failed at iteration 0** — the KAOS `agent_sdk` provider spawns the Claude Code
   CLI, which refuses to launch nested inside the running campaign session.
2. **Reviewer model changed.** The retry ran as subagents of the campaign session on the
   *same* reviewer model (`claude-fable-5`) — and **that model is out of usage credits**, so
   all reviewers failed again. This round is therefore being reviewed by **Opus** rather than
   the `claude-fable-5` used in R10–R23.

**What this means for your review:** the circulated prompt files are **byte-identical** to the
KAOS attempt and no proposal content was altered by either retry — but **R24's reviewer model
is not the same as prior rounds'.** Round-over-round score comparisons (e.g. "scores rose from
R23") are therefore **not valid** for this round, and no conclusion in these minutes may rest
on such a comparison. Verdicts and objections stand on their own merits; the *trend* does not.
If you believe a model change materially compromises the panel's continuity, file it as an
objection — it is a legitimate one.

## A6. Still outstanding at circulation time

The 2026-08-09 research sweep and the S1/S1b runnability check were still executing when this
addendum was sealed. If either lands before your review completes it will be appended as
`_ADDENDUM2.md`; if you see no such file, treat the research field as **unswept for 08-09** and
treat S1/S1b runnability as **unverified** — in particular, do not assume the 24 exec-wm
simulators and 25 recorded traces are present and replayable just because the proposal
sequences them.

## END OF ADDENDUM ##
