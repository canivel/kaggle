# R17 Portfolio GO/KILL decision — contract-v1.1 / budget-regime (OBJ-H)

**Status: PENDING-ORCHESTRATOR.** Filed 2026-07-23 concurrent with the A14 look, per R17 OBJ-H
(rl-planning MAJOR, `learnings/panel/round17/rl-planning.md` line 27) and directives D6. The
development agent files the decision *line* with a recommendation; the GO/KILL *call* is a
strategy decision above the dev agent's pay grade and is left for the orchestrator to ratify or
override. It must not be tabled — rl-planning is right that the plan-to-1.44+ currently rests on
A17″ alone, and that concentration must be stated and priced, not implied.

## The finding (rl-planning MAJOR, accepted)

After the held-out resolver collapse (10/11 DROP), W1's null-to-negative sentinel, and the
verified depth-weighted (level-number-weighted) scorer, the portfolio prices almost entirely at
≈0:

| Component | Priced value | Depth channel? | Registration date |
|---|---|---|---|
| B− branch | sealed near-certain FAIL (P_pass 0.037) | no | sealed |
| EWM | +0.04 central, no new-clear channel | no (efficiency-denominated) | sealed |
| Banking (best carrier sc25) | 36–43 live actions post-replay | no (residual budget, depth-scorer discounts) | sealed |
| **A17″ 72B screen** | the ONLY depth-targeting line | **yes** | pre-Aug-1 screen |
| **Schema-class revise loop** | closed g50t/m0r0/sk48 first-line; 283–412 actions/game (3–8× B=150) | **yes (demonstrated)** | **NONE — deferred, no date** |

The one mechanism with running-code evidence of closing hard games (the Schema revise loop) is
deferred to "the next registration cycle" with no date, while the LB wall moved 1.44 → dense
1.44–1.60 and we sit at 1.33 and sliding (#44 → #45). Under the completion / level-number-weighted
scorer, sc25's 36–43 residual actions cannot plausibly buy a new level (rl-planning Q4).

## The tension to resolve (not table): B=150 vs the revise loop

The sealed sentinel/EWM regime is **B=150 actions/game**. The Schema revise loop that actually
closed the UNRESOLVED trio runs at **283–412 actions/game — 3–8× the sealed budget.** These are
in direct conflict: the sealed efficiency discipline (B=150, sentinel warns at 90% = 135 actions)
structurally *forbids* the one demonstrated depth mechanism. You cannot seal B=150 as the budget
regime and also run the revise loop; one of them has to give.

## Recommendation (dev agent — for orchestrator ratification)

**Recommend: GO on a contract-v1.1 budget-regime registration that carves out a depth-budget
lane, with a dated line concurrent with the A14 look.** Concretely:

1. **KILL nothing yet in the sealed B=150 lane** — sentinel/EWM stays as certified telemetry +
   efficiency rail (it costs ~0 and the seal is done). But stop pricing it as a path to 1.44+.
2. **GO on a second, depth-budget lane (contract-v1.1):** register the Schema-class revise loop at
   its native 283–412 actions/game as a **separate budget regime** targeted only at the hard-game
   frontier (g50t/m0r0/sk48 and their class), NOT the whole 25-game bench. Resolve the B=150-vs-
   revise-loop tension by making them *different lanes with different budgets*, not one global B.
3. **Date it:** file the contract-v1.1 registration to be adjudicated at the **A17 pre-Aug-1
   screen review** (the next scheduled decision point), so the plan does not rest on A17″ alone.
4. **Gate GO vs KILL of the revise loop on:** whether the depth-weighted scorer credits the trio
   closures at a rate that beats null10 on held-out games (the sealed gate discipline, 2607.12227
   charter) — i.e. verify the Schema fixed-resolver hypothesis first (OBJ-I, the 2nd free-build
   push slot). If the fixed-resolver verification passes on held-out streams, GO the depth lane;
   if it collapses like the general resolver did, KILL and fall back to A17″ + frozen-fork.

**One-line rationale:** the only two depth-targeting lines are A17″ and the Schema revise loop;
resting the whole Nov-2 plan on A17″ alone is the concentration rl-planning correctly flags, and
the revise loop is the only component with running-code evidence of closing the UNRESOLVED trio —
so it deserves a dated GO/KILL, and the cheapest gate for it (OBJ-I fixed-resolver verification)
is already the recommended 2nd push.

**PENDING-ORCHESTRATOR:** ratify GO on contract-v1.1 depth-lane (recommended) OR explicit KILL
(accept A17″-only concentration and state it). Either way the decision is now on a dated line, not
tabled.

---

## ORCHESTRATOR RATIFICATION — 2026-07-23 (dated line, concurrent with the A14 look)

**Decision: GO on the contract-v1.1 depth-budget lane**, with evidence-limited scope. The
OBJ-I gate returned a split verdict, ratified as follows:

- **Gate evidence (runs/schema_fixed_resolver/report.md, C6-legal fixed hypotheses, no
  fitting):** tr87 CERTIFIES at the sealed bar (det 1.0000, Wilson LB 0.9738, 143 pooled
  visits — the in-sample DROP was split support-starvation, not wrong physics); wa30 and
  ka59 FAIL under every legitimate reading and STAY UNRESOLVED. This is neither the full
  pass nor the general-resolver-style collapse the draft gate anticipated: the law class
  demonstrably transfers when the law is right, and fails detectably when it is not —
  which is exactly the certify-or-reject behavior the lane needs.
- **Depth pricing (runs/r17_sealing/d4_provisional_reprice.md):** a single frontier depth
  event prices +0.19 to +0.29 rail vs ~0 for every efficiency channel (4–15×). The revise
  loop and A17″ are the only depth-targeting lines; A17″-only concentration is the risk
  rl-planning flagged.

**Ratified terms:**
1. B=150 lane: sealed as certified telemetry + efficiency rail; no longer priced as a
   path to 1.44+ (per recommendation §1, unchanged).
2. Depth lane GO: register the Schema-class revise loop at native budget (283–412
   actions/game) scoped to the hard-game frontier (g50t/m0r0/sk48 class). tr87 enters
   via the fixed-hypothesis sealed re-entry path (filed, not silently un-struck);
   wa30/ka59 are explicitly NOT re-admitted.
3. Adjudication date: the pre-Aug-1 A17 screen review, as recommended.
4. Kill-switch (sealed with the registration): the lane's first certified eval must beat
   null10 on HELD-OUT games (arXiv 2607.12227 charter); failure = KILL, fall back to
   A17″ + frozen-fork.
5. Push discipline note: the OBJ-I question was answered locally at $0; today's second
   kernel push is HELD (not spent) — a fresh-stream tr87 confirmation build rides a
   later quota slot after the panel sees this ratification.
