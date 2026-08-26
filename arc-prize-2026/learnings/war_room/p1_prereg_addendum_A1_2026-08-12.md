# P1 PREREG — ADDENDUM A1: the K-P6 denominator

**Status:** PRE-DATA. Written 2026-08-12 **12:45Z**, with
`canivel/arc3-duck-p1-eval` v1 confirmed `KernelWorkerStatus.RUNNING` at that
timestamp (checked immediately before writing). No arm output has been pulled,
and none exists.

**The seal is not touched.** `learnings/war_room/p1_prereg_2026-08-12.md` is
unmodified. This addendum resolves an *ambiguity* in a sealed clause; it does
not loosen, retune, or reinterpret any threshold, band, or kill rule.

---

## The ambiguity

Sealed canary **K-P6** reads:

> `dup_rate` (M3) is **below** the family's as-run duplicate rate

"The family's as-run duplicate rate" is not defined anywhere in the seal, and
the family admits **two** defensible values:

| Scope | Family rate | Provenance |
|---|---|---|
| **All executed actions** | **12.65%** | like-for-like with what the canary itself computes |
| Cleared levels only | 2.70% | the scope used by `efficiency_diagnosis_2026-08-12.md` |

This was surfaced by `duck_eval/warpack/p1_score_selftest.py` (flagged tension
T1), not by looking at arm data — which does not exist yet.

**Why it is decisive:** the three sealed M3 replay expectations are 6.9% / 3.4%
/ 4.1%. All three fall **between** 2.70% and 12.65%. So the choice of
denominator, and nothing else, determines whether K-P6 reads PASS or FAIL. Left
unresolved it would have resolved **DISPUTED**, and would then have been settled
*after* seeing the arm's number — which is the one thing that must not happen.

## Ruling: the denominator is **all executed actions — 12.65%**

The canary's numerator is `dup_exec / executed`, computed over **all executed
actions**. Comparing that numerator against a cleared-levels-only denominator
pairs an all-actions numerator with a cleared-levels denominator. That is not a
conservative choice or an aggressive one, it is an incoherent one: the two
quantities are not measured over the same set of actions, so their ordering
carries no information about the mechanism.

**This is the identical error that killed the previous arm.** Canary K-A3 fired
`KILL` on animation-awareness because a token bound was evaluated against a
*generated-token* denominator while the mechanism's cost was in *input* tokens;
the post-mortem recorded that the bound was mis-specified, not that the summary
was expensive. The sealed P1 prereg already cites that failure as its reason for
pre-registering **no** token-fraction canary. Repeating the same scope mismatch
one arm later, in a different unit, would be a governance failure with a written
precedent sitting directly above it.

Like-for-like scope is therefore the only coherent reading, and it is the one
that was in force when the canary was written.

## Disclosure (stated before the data, deliberately)

This ruling makes K-P6 **more likely to PASS**: all three sealed M3 expectations
(6.9 / 3.4 / 4.1%) sit below 12.65%, whereas all three sit above 2.70%. We are
recording that plainly rather than leaving it to be noticed later. The argument
above stands on scope coherence and would be made identically had the inequality
pointed the other way — but the reader is entitled to check that claim against
our incentives, so the incentive is disclosed.

Two further constraints keep this honest:

1. **K-P6 is a READING gate, not a hard gate.** Per the seal, K-P0..K-P3 are
   hard (failure ⇒ discard-grade, nothing may be read). K-P4..K-P6 failing means
   *the mechanism did not deliver* and M1/M2 may not be read as evidence. So
   this ruling cannot rescue a dead arm; it can only decide whether a
   secondary endpoint is legible.
2. **M0 is untouched and remains the primary.** `saved/requested` ∈ [3%, 30%]
   decides mechanism delivery. Nothing in this addendum moves it.

## Tensions noted but NOT acted on

Recorded so they are on the record before the data, and explicitly left for the
coordinator. None of these change any sealed threshold.

- **T2 — K-P5 can fail while safety is intact.** The shipped latent-state
  detector is *level*-scoped; the 8-game certification was *game*-scoped. A
  K-P5 failure therefore does not by itself imply the safety rule was inert.
  Do not read it as one.
- **T3 — Kill rule 4 is not post-hoc falsifiable.** The scorer verifies the
  sealed settings were in force rather than reconstructing them from behaviour.
  This is a limit of the instrument, stated rather than hidden.
- **T4 — K-P1 is weaker than it looks.** It is satisfied by per-game
  `kind=game_end` lines, so it tests that the seams fired, not that the
  mechanism engaged. Engagement is M0's job. Do not cite K-P1 as evidence of
  engagement.

## Scorer changes made in the same pre-data window

`duck_eval/warpack/p1_score.py` was repaired (and `p1_score_selftest.py`, 98
assertions, added) **before** any arm output existed. The repairs are logged in
ITERATION_LOG.md for 2026-08-12. The load-bearing one: the scorer read the
Kaggle log raw, but real logs are a JSON array of `{stream_name,time,data}`
records, so the canary lines never matched and `errors` resolved to `None` —
**the arm would have been KILLED by a log-parsing bug, with no defect in the
mechanism at all.** Validation: on the real `animation_v1` output the repaired
scorer reproduces the sealed 1.6352 RHAE and the 10.54% cleared-level duplicate
rate exactly.
