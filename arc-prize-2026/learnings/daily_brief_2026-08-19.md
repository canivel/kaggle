# ARC-AGI-3 daily brief — 2026-08-19 (Wednesday)

**Session role:** iterate/orchestrator. **Weekday ⇒ no panel round, no weekly fingerprint pass.**
**Push budget: 0 available to this session — both slots were already spent by the graft-lane
operator before this session opened (v3, v4), and `runs/lane_locks.json` holds an active lock on
`canivel/arc3-graft-floor-eval`.** Under the 08-18 lane-ownership ruling this session pushes
nothing. Its job today is monitor → certify → record → keep the queue armed.

---

## 1a. RESULT DEEP-DIVE — the 08-19 00:07 draw

**Score 1.15** (frozen-fork filler, `canivel/arc3-duck-repro v3`, auto-armed by the daemon).
Ledger after the draw: **n=36, mean 0.9461, s 0.1558, max 1.33, trailing-4 1.0325, z=+1.36,
promotion bar 1.0848.**

**Interpretation, not the number:** this is the second-best draw of the campaign and the best
since the 1.33 was banked on 2026-07-18 — and it changes *nothing*, because it is **interior to
the record and below the promotion bar**. It was produced by a submission whose bytes have not
changed in a month, so it is a **draw from the frozen fork's own noise distribution, not
evidence about any treatment**. The correct reading is the one the campaign has already paid to
learn: at s=0.1558 on this rail, a single +1.36σ draw is an ordinary event (~9% of draws), and
the standing rule — *never gate on public-LB single draws* — applies to good news exactly as it
applies to bad. **Pre-registered expectation: none — no arm was riding this draw.** The public
max remains **1.33, unmoved since 2026-07-18**.

**Mechanism evidence:** none available from a filler draw (no per-game eval output is produced
by the submission rail). The mechanism question of the week — *forgetting REFUTED or
DELIVERY-WITHOUT-USE?* — is untouched by this draw and remains open.

## 1a-bis. THE LIVE EVAL — graft-floor v4

`canivel/arc3-graft-floor-eval` **v4 is RUNNING** as of session start. This is the material
result of the day so far: v1, v2 and v3 all died at t≈5s, and **v4 has survived past that
point**, which provisionally confirms the root cause diagnosed yesterday — the competition input
**moved** from `/kaggle/input/competitions/<comp>/` to `/kaggle/input/<comp>/` in the batch
environment; it was never absent. The two-environment divergence is therefore a standing fact:
**build/eval rail = NEW layout, submission-rerun rail = OLD layout** (the 1.15 filler scored
hours after v1/v2 died, so the eternal fallback was never at risk).

**Reading rule (binding, sealed pre-push):** the scorer certifies the graft banner **before any
levels number is read**; an uncertifiable install is **INFRA DEATH, never NULL**.
**Instrument audited before the data landed** (this session, 08:26 EDT):
`graft_score.py --selftest` → **22/22, 0 failures**, sealed thresholds printed:

| verdict | condition |
|---|---|
| HARM (decisive) | lc_total ≤ 12 (mean Δlc ≤ −0.286315) |
| NULL (decisive) | lc_total 13..26 |
| SIGNAL (decisive) | lc_total ≥ 27 (mean Δlc ≥ +0.286315) |
| INFRA DEATH (not decisive) | banner uncertifiable / no benchmark.json / ≠25 games / window drift >5% |

Baseline is m=3, per-game mean lc 0.773333, σ 0.141740, C(3)=2.02. NULL must be written as
**"not a BIG effect," not "no effect"** — the SIGNAL bar is +28% over the best lc ever recorded
on this rail (22, war_eval_v1).

## 1a-ter. UNREAD ARTIFACT — the 08-17 engine misfire

`canivel/arc3-q38-engine-eval` is **COMPLETE**. Its output has been **pulled to
`runs/kernel_pulls/engine_misfire_0817/`** (benchmark.json, per-game cards, transcripts, vLLM
log) so it is on disk and no longer at retention risk. **It has NOT been scored and MUST NOT be
converted into a verdict:** it came from a misfire push, no sealed prereg covers it, and the
Q38 engine-generation hypothesis was already settled REFUTE-2× on 08-16. Reading data and then
deciding what it tests is the one move that voids a screen. **Disposition remains the
coordinator's call** — the honest options are (i) discard as an unregistered run, or (ii) treat
it strictly as an exploratory, non-inferential mechanism sample explicitly barred from any
promotion decision.

## 1a-quater. ★ INSTRUMENT DEFECT — the KAOS spawn path itself

Both analysis agents spawned on the standing directive's **default model `fable-panel` failed
producing zero bytes**: `agent_sdk provider cannot forward 8 OpenAI-style tool schema(s)` →
`Reached maximum number of turns (1)`. Root cause is **configuration, not transient**: `kaos.yaml`
defines `fable-panel` and `opus5-panel` as `provider: agent_sdk`, adopted deliberately as the
"sanctioned text-only path" because claude_code's XML tool protocol tripped dual-use gating on
those models. **`agent_sdk` is text-only and single-turn, so the mandated default cannot execute
any task that holds a tool** — i.e. almost every task in this protocol. The default is correct
for its origin use (`scripts/panel_round.py` reviewers, which only read a proposal and reason)
and was generalized to a job it cannot do. Re-spawned on `claude-sonnet` (`provider:
claude_code`). **Fifth instance of this campaign's signature failure mode — the instrument, not
the hypothesis — and the first time it has been the orchestration substrate.**
**Recommendation to the coordinator:** amend the KAOS-native mandate to *"spawn via KAOS;
`fable-panel` for text-only review, a `claude_code`-provider model for anything holding a tool."*

`kaos bench rejections` was run at session start per the CONSUME directive: **registry empty** —
nothing to consume yet, the loop is live but has no cross-workspace content.

---

## 1b. DISCUSSIONS SWEEP

**No new topics since the 08-18 sweep.** Newest post on the board is still **735662** (08-17
13:03), already swept yesterday. Nothing at all from 08-18 or 08-19.

**One topic yesterday's sweep MISSED and that is worth the correction — 735590, "ARC-AGI-3 run
went backwards on the leaderboard. What are we missing?"** (Pengyi Peng1, 08-17 03:31, 3
replies). It was posted before yesterday's cutoff and the 08-18 brief recorded "one new topic";
it should have been two. The content is directly on our symptom:

| their date | submission | public |
|---|---|---|
| 08-12 | naive graph-explorer MVP | 0.25 |
| 08-15 | PAGI-001 (offline eval **+121%** vs baseline) | 0.74 |
| 08-16 | PAGI-002 (offline eval **+21.6%** over 001) | **0.28** |

**ADOPT as corroboration — with an inversion of their reading.** They interpret 0.74→0.28 as
their "most improved" agent being worse. **On our own measured rail that inference is unsound:
s=0.1558 and their own 3-draw span is entirely consistent with draw noise around a flat mean.**
They are doing on n=3 precisely what our standing rule forbids — reading a single public draw as
a treatment effect — and they are doing it in the *pessimistic* direction, which is the same
error we avoided in the *optimistic* direction with today's 1.15. **The independent value is
that a second team, with a different agent and a different offline harness, reports offline
gains that do not appear on the public rail.** That is a third-party datapoint for the
local↔public decoupling we already treat as standing. It changes no plan.

The most substantive reply (Akagha Chimgozirim) claims teams may be deep-copying **game source
code** for offline exploration that is not available for the LB test set. **IGNORE for us** —
our eval rail scores levels-completed on the 25 public games with the same harness the
submission uses; we do not have or use private-set source. Recorded because it is a plausible
explanation of *other* teams' local↔public gaps and therefore weakens any inference we might
draw from their reported offline numbers.

**Re-verified: the forum still discloses NOTHING about banking, transfer, grafts, or any
mechanism behind the step jumps.** Zero disclosed, unchanged.

## 1c. RESEARCH SWEEP

**★ The one real find — MemHarness: *Memory Is Reconstructed, Not Replayed* (arXiv:2607.28272).**
**Zero prior hits in our record.** Recasts memory-guided decision-making as five stages —
observation → retrieval → **critique → reconstruction** → action — on the explicit claim that
agents which **replay retrieved experience verbatim suffer negative transfer**, because stored
experience is abstract while the decision-time state is concrete and changing. Reports 85.2%
ALFWorld / 75.6% WebShop at 7B, +8.8/+9.5 over pure GRPO.

**Verdict: PARK (not ADOPT, not a plan change) — but it is the closest thing in weeks to our own
open question, and it converges with the 08-16 find.**
- **Why it matters here:** our mech-C measured **96.3% delivery with no behaviour change**. The
  open question is *forgetting REFUTED or DELIVERY-WITHOUT-USE?* MemHarness is a direct claim
  that **delivery without reconstruction is not merely insufficient but actively harmful**, which
  is a mechanism for our null rather than a restatement of it.
- **Why PARK and not ADOPT — and this is the disciplined part:** MemHarness's reconstructive
  ability *emerges through end-to-end GRPO training*. The inference-only version — "insert a
  critique-and-reconstruct step in the prompt" — is **exactly the class of intervention that
  arXiv:2608.12321 (swept 08-16) predicts will null**: probes decode the constraint >88%, no
  prompted intervention reaches the repair corner, *"a routing problem, not a knowledge
  problem."* Two independent papers now point at the same place, **and the one that tested
  prompt-side fixes found they don't work.** Adopting the cheap version would be running the
  experiment our own sweep already predicts fails.
- **Blocked by:** `feedback_arc_zero_budget` — GRPO training needs a rail we do not have.
- **Standing recommendation to the coordinator (unchanged from 08-16, now with a second
  citation): retire "did transitions arrive?" as settled at 96.3% and pose the successor
  question as one about USE/routing, not delivery.**

Other items, all **IGNORE**:
- **OSU-Mem / "When Does Overlap Help?" (2606.28376)** — cell-conditional analysis of trajectory
  memory. Analysis of when overlap helps, not a liftable mechanism.
- **Evo-Memory, AgentOdyssey, EvoAgentBench** — re-surfaced; **IGNORE, unchanged** (benchmarks,
  not mechanisms). Same disposition as 08-18.
- **AERA (2605.25931)** — re-surfaced by the ARC-AGI-3 query; **already swept 3× (07-06, 07-27,
  08-12). IGNORE, no new version.**
- **Sensi (2603.17683), Workspace Optimization (2605.09650)** — pre-date our sweep window and
  are training-rail methods. IGNORE under zero-budget.
- **No new published work on prune-then-replay action-count banking in a scored environment.**
  Unchanged for the Nth week; the banking arm remains falsified on reachability regardless.

---

## 1b-bis. BOARD (full CSV, 2026-08-19T12:30Z, 2410 teams)

| metric | 08-18 | **08-19** | Δ |
|---|---|---|---|
| teams | 2408 | **2410** | +2 |
| **our rank / score** | #261 / 1.33 | **#266 / 1.33** | **−5 ranks, score unmoved** |
| gold cutoff (rank 13) | 2.24 | **2.33** | +0.09 |
| **gap to gold** | 0.91 | **1.00** | **+0.09** |
| median | ~0.25 | **0.26** | ~flat |
| p99 | 2.10 | **2.13** | +0.03 |
| ≥2.00 | 33 | **36** | +3 |
| ≥1.65 | 112 | **119** | +7 |
| ≥1.33 (ties incl.) | — | 267 | — |

**The pattern holds for a fifth straight night and its shape is unchanged: the tail inflates,
the median does not.** Gold is now a full **1.00 above us** — the gap has roughly doubled in
four days while our bytes sat still. We lost only 5 ranks (vs 44 yesterday), which is *not*
good news; it reflects that the teams passing 1.33 have mostly already passed it. Medal rule
used: Kaggle's standard for >1000 teams, gold = top 10 + 1 per 500 teams ⇒ **rank 13**.

**Controls (the load-bearing check).** For a **third** consecutive day, the two teams whose
methods we can actually name spent draws and gained nothing:

| team | rank | score | subs | last submission |
|---|---|---|---|---|
| cstl | 1 | 3.57 | 29 | 08-18 17:18 |
| Daniel Franzen | 4 | 2.58 | 45 | 08-18 19:59 |
| **Tufa Labs** (authors of the harness we fork) | **127** | **1.62** | 111 | **08-18 16:52 — drew, flat** |
| **@Abstraction Lab & MindsAI** (Jack Cole, TTT originator) | **30** | **2.05** | 123 | 08-18 16:50 |

**★ One control moved and it deserves care: MindsAI is at 2.05, against the 1.59/#22 in our
memory file.** That is a **+0.46 change of unknown date and unknown mechanism** — our record has
no dated series for them, so this is **not** a measured single-draw step and must not be added
to the step-signature list. What it does do is **weaken a specific comfort we have been taking**:
"pedigree doesn't predict this board" was anchored partly on Cole sitting at 1.59. He is no
longer there. Tufa Labs, the other pedigree anchor, **is** still flat at 1.62 — so the claim
survives in weakened form, on one anchor instead of two. **Flagged for Sunday's panel; not
actioned today.**

**★ INSTRUMENT GAP FOUND AND CLOSED.** Per-team single-draw deltas — the step-signature
analysis this campaign has run for days — **cannot be computed from our artifacts**: no prior
full-leaderboard CSV was ever retained, so every prior "+2.09, +1.37 on ONE draw" claim was
derived from same-day aggregates and recollection, not from a diffable snapshot. **Today's CSV
is retained at `runs/lb/arc-prize-2026-arc-agi-3-publicleaderboard-2026-08-19T12_30_37.csv`**,
which makes 08-20 the first day a genuine per-team draw-level delta is computable. Until then
the step-signature list stands as previously recorded and **no new steps are claimed today** —
the honest statement is that we lacked the instrument to measure them. Sixth instrument defect
this week; same family as the others.

**Unchanged and still binding:** every step remains **UNKNOWN — zero disclosed** (today's
targeted search for a disclosed August technique returned nothing; the forum discloses nothing).
**Do not infer method from movement.** And `LastSubmissionDate` is LATEST while `Score` is BEST
— proven on our own row, which reads "2026-08-19 00:07" against a 1.33 banked 07-18 — so no
"submitted after T" inference is sound.
---

## 2. OPEN QUESTIONS (weekday — no panel; these go to Sunday's round)

1. **★ PENDING TODAY: the graft-floor verdict.** v4 is the first eval build to survive the mount
   incident. Certification order is binding — **banner first, number second**; uncertifiable
   install is INFRA DEATH, never NULL. Sealed bars: SIGNAL ≥27 lc, HARM ≤12, NULL 13–26.
2. **Forgetting REFUTED, or DELIVERY-WITHOUT-USE?** — now with a **second independent citation**
   (MemHarness 2607.28272 joins 2608.12321). Both locate the failure at **use/routing**, not
   delivery, and the one that tested prompt-side fixes found they do not reach the repair corner.
   **Recommendation stands: retire "did transitions arrive?" (settled, 96.3%) and pose the
   successor question about USE.** Note the trap this creates: the cheap in-context version of
   MemHarness is precisely the intervention our own sweep predicts will null.
3. **Disposition of the 08-17 engine misfire artifact** (now safely on disk, unscored).
   Coordinator's call: discard as unregistered, or admit as exploratory-only and barred from
   promotion.
4. **Does "pedigree doesn't predict this board" still hold?** It was anchored on Cole at 1.59 and
   Tufa at 1.62. **Cole's team now reads 2.05.** One anchor gone, one intact — the claim needs
   restating in weakened form or retiring.
5. **The KAOS-native mandate is currently unexecutable as written** for tool-using tasks (60s
   time-to-first-byte watchdog, no config knob; `fable-panel` additionally text-only/single-turn).
   Needs an amendment, and the defect should be filed upstream per the AI-feedback policy.
6. **Unchanged:** every step jump remains UNKNOWN/undisclosed; the >1.70 band is not explained by
   draw count; never gate on single public draws — in either direction.

## 3. WHAT WAS DONE TODAY (no pushes available to this session)

- Certified the sealed scorer **before** the data landed (22/22).
- Pulled and preserved the 08-17 engine artifact **without scoring it**.
- Root-caused the KAOS spawn failure to a hardcoded 60s idle watchdog; verified providers healthy.
- Closed the leaderboard-snapshot instrument gap (`scripts/lb_snapshot.py`, diff path
  exercised end-to-end against a synthetic prior, synthetic file removed).
- Corrected yesterday's missed discussion topic (735590) and inverted its unsound reading.
