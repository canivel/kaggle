# EXEC-WM AUDIT — THE BREAK-CLUSTERING HYPOTHESIS IS REFUTED, AND THE FALLBACK LABEL WAS LYING

**Arm:** `canivel/arc3-execwm-eval` v1 · pushed 08-25 · board draw 1.05 (08-26)
**Prereg:** `learnings/war_room/execwm_prereg_2026-08-25.md`
**Artifact:** `runs/kernel_pulls/execwm_v1/`
**Status of this item:** #1 handoff on 08-26 and again on 08-27. **Closed today.**

## THE HEADLINE: WE SPENT TWO DAYS CHASING A COUNTER THAT NEVER FIRED

The standing hypothesis was that the agent's `MAX_BREAKS_PER_LEVEL = 3` latch was punishing
a **frame-layer defect** — that BREAKs would **cluster on level-completing steps**, so the world
model was being blamed for a frame-selection bug. The brief's proposed repair (08-27 handoff #2b)
was to convert the latching BREAK threshold into a Polyphony-style retrodiction ratio.

**Both the hypothesis and the proposed repair are wrong, and the artifact says so plainly.**

| fallback reason | occurrences | share of 32 level-instances |
|---|---|---|
| **`no-verified-model`** | **26** | **81 %** |
| `sprite-lost` | 3 | 9 % |
| `prediction-breaks` | 2 | 6 % |

*(Verified mechanically: `grep` over the pulled kernel log + `stdout.log`. Raw concatenated counts
were 52 / 6 / 4 because both logs carry the same lines; halved above. Independently hand-tallied
by the audit agent to the same numbers.)*

- `MAX_BREAKS_PER_LEVEL = 3` was reached **exactly once** in the entire run (ka59) — and via the
  **`sprite-lost`** path, not the prediction path.
- **The prediction-BREAK latch fired ZERO times.**
- Both level-instances *labelled* `prediction-breaks` actually fell back at **`breaks=1`**. They
  tripped the **second disjunct** of the same condition (`exec_wm.py:1066-1067`,
  `len(verified_moves()) < MIN_VERIFIED_MOVES`) — **not** the break counter.

**⇒ Raising or re-shaping the BREAK threshold would have changed 0 of 31 fallbacks.**

### The defect that cost the two days is a MISLABELLED REASON STRING
One code path, two disjuncts, one shared label. A level that fell back for *having too few verified
moves* was reported as having *broken its predictions*. Every downstream reader — including two
consecutive daily briefs and this campaign's own handoff queue — then reasoned about a break-latch
that had never engaged. This is `feedback_audit_the_instrument` in a new costume: not a gate that
could not fire, but **a gate that reported someone else's firing as its own.**

## WHAT THE NULL ACTUALLY IS: DATA STARVATION, NOT A BAD WORLD MODEL

Re-read as retrodiction (each transition scored from the **observed** prior frame, Polyphony-style):

| | value |
|---|---|
| Transition accuracy, pooled | **810 / 818 = 0.990** |
| Excluding ls20 (which alone supplies 686 of 818) | **124 / 132 = 0.939** |
| Games with **any** transition evidence | **9 / 18** |
| Level-instances planned / cleared / fell back | 6/32 · 1/32 · 31/32 |

Per game: ls20 686/686, ka59 45/45, dc22 32/32, g50t 16/16, re86 2/2, wa30 1/1 (all 1.000) ·
sp80 13/14 · sk48 9/10 · m0r0 6/12 · **nine games undefined (n = 0)**.

**The world model is accurate wherever it has data. It usually has none.** Nine of eighteen games
produced **zero** usable transitions *despite 16–20 probes executed per level*. 26 of 32
level-instances died at `no-verified-model` — a deterministic upstream extraction failure, several
of them while holding candidate rules with real deltas and nothing to test them against.

Two further shape problems, both independent of the above:
- **Gate shape.** g50t retrodicted **16/16 = 1.000** and was still discarded, because only 1 of its
  4 verified rules was a `MOVE` and the gate demands ≥2 verified *moves*. A perfect model, refused.
- **21 EVENTs vs 3 BREAKs** — unexplained mispredicts outnumber charged model errors **7:1**, so the
  charging rule is not measuring what it claims to measure.

### The frame-layer defect is probably still real — it just doesn't look like BREAKs
It manifests as **n = 0**, not as clustered breaks. **A retrodiction instrument would have caught
this on day one, because a 0/0 denominator is loud, whereas "fell back, floor ran, lc = 25" is
silent.** That is the transferable lesson and it is worth more than the arm: *instrument your
mechanism so that having no evidence is itself a visible result* — which is, word for word, the
principle Polyphony states for its own `metrics.py` (*"so that a score of zero is still a result"*).

## RULING

**The prereg's decisive-kill clause DOES NOT FIRE** — the hypothesis it was conditioned on is refuted.
**But exec-WM must NOT be naively re-seeded either.** The failure is **not stochastic**: 26 of 32
level-instances died at a deterministic upstream step, and a second seed of the same build reproduces
it. Note also the board pressure has evaporated — the certified floor itself drew **1.14** the night
after exec-WM drew **1.05** (floor config now n=6, mean 1.5033, sd 0.3010), so the 1.05 is not
distinguishable from its own comparator and argues nothing either way.

What the artifact **does** establish, and it is not nothing: the mechanism works where it gets data
(810/818), one level was cleared **by pure BFS with zero LLM tokens**, and the per-level floor
fallback did **no damage** (lc 25 vs floor 28 at n=1, MDE 11.1 — indistinguishable).

**RE-SCOPE: "repair the observation layer, then re-seed."** Ordered, CPU-only, slot-free:
1. Replay the retained `artifacts/*_events.jsonl` to determine why transition extraction returns
   empty in 9/18 games. **This is the whole ballgame** — nothing else matters until n > 0.
2. Replace the binary `verified_moves() >= 2` gate with a retrodiction-ratio gate (this repair is
   still right, but for the g50t reason, **not** the BREAK reason the brief gave).
3. Split the mislabelled `prediction-breaks` reason string into its two disjuncts, and add `game_id`
   to the log lines.
4. **Only then** spend a slot.

## CREDIT AND CORRECTION
This audit was produced by a KAOS agent (`execwm-audit`, `opus5-code`) reading the artifact directly.
Its numbers were independently machine-verified above before being recorded here. It corrected a
diagnosis that two consecutive daily briefs had asserted — including the specific repair the 08-27
brief recommended — which is exactly what an adversarial read is for.
