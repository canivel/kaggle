# TASK — DESIGN S1's SUPERVISOR TRIGGERS FROM OUR OWN TRANSCRIPTS (ZERO GPU, zero pushes)

Repo: `F:/kaggle/arc-prize-2026` (ARC-AGI-3 campaign). Read-only except the ONE output file in section 6.
NO kernel pushes, NO submissions, NO queue edits, NO writes to `notebooks/` or another lane's artifacts.

## 1. THE ORDER

Coordinator ruling, ITERATION_LOG 2026-08-24, item 3:

> **ZERO GPU: design S1's supervisor triggers from OUR OWN transcripts** (AVO discloses nothing
> quantitative — waiting is waiting for nothing): thresholds from the measured distributions
> (repeat-action rate 45.2%, interior no-op 8.8%, post-clear burn 88%) with the interior-keyed
> signature; draft as a prereg for review.

## 2. WHAT S1 IS, AND WHY THERE IS NO CRIB

S1 is a proposed **supervisor**: a cheap, non-LLM (or minimally-LLM) monitor that watches the agent's
own action/observation stream during a game and INTERVENES when the agent is measurably stuck —
rather than letting it burn the 7920 s game clock repeating itself.

The inspiration is AVO (arXiv 2603.24517 + NVIDIA dev blog), read and summarized in
`learnings/community/brief_2026-08-24.md` find 7. **AVO discloses its trigger conditions only
qualitatively** — "stalls when it exhausts its current line of exploration", "unproductive cycles:
edits that repeatedly fail to improve scores", dead-end revisits. No thresholds, no windows, no
cadence. So there is nothing to copy and waiting for a v2 of the paper is waiting for nothing.
**Every threshold in your design must be derived from OUR measured distributions and defended.**

## 3. THE EVIDENCE BASE (all local, all free)

Primary forensics already done — READ THESE FIRST, do not redo them:

- `learnings/war_room/perturn_program_2026-08-22.md` — the wasted-turn taxonomy on the certified field
  floor (424 acting turns / 449 analyzer invocations / 1181 generations / 1639 actions). Key rows:
  **C2 = 45.2% of actions immediately repeat the identical previous action (same id + same `data`)**;
  **C3 = 44.8% of 524 ACTION6 clicks re-use a coordinate already clicked in that game**;
  54.0% of acting turns sit on a level never cleared; 62.1% of generations execute no action.
  Also the LONG-RUN DEGENERATION section and the epsilon=0.17 retry arithmetic
  (`r < sqrt((k+2)/k)`), which bounds what an intervention can be worth.
- `learnings/war_room/original_program_2026-08-22.md` — the CAP THEOREM
  (`per-game score <= 100*k(k+1)/(N(N+1))`) and the 675/675 gave-up-on-the-clock finding.
- `duck_eval/p0/BP35_DIAGNOSTIC_2026-08-22.md` and anything else under `duck_eval/p0/`.
- **The noop guard**: `hard_noop_guard` is ARMED and has NEVER FIRED — 0 blocks across 1639 field-floor
  actions — because `board_signature()` is blake2b of the FULL 64x64 grid while the games render a
  ticking HUD/timer strip, so the (level, board, action) key can essentially never recur. Find it
  (`noop_guard.py:16-21` per the record) and read it. **This is the single most important design
  lesson available to you**: the campaign has already shipped one supervisor-shaped guard that could
  not fire. The "interior-keyed signature" the coordinator names is the fix — key on the INTERIOR of
  the grid, excluding the HUD/timer strip.

Raw artifacts (read-only): `runs/kernel_pulls/q38_field_v1/` (certified floor, lc 28), plus other pulls
in `runs/kernel_pulls/`. `runs/tufa_example_run/benchmark.json` is 20 clone passes x 25 games
(Qwen3.6 — a PREDECESSOR-MODEL property, per `reference_config_provenance_2026-08-22.md`; use it for
consistency questions only and label it).

**Two of the coordinator's three numbers need sourcing.** 45.2% is `perturn_program` row C2 (verified).
**"interior no-op 8.8%" and "post-clear burn 88%" — FIND THE SOURCE OR MEASURE THEM.**
`perturn_program:290` records the interior no-op rate as **[UNK]** and says the frames sit in
`intermediate_states.pkl`, reachable with a small unpickling shim, ~one CPU hour. If those numbers
are not already on record, either (a) measure them yourself from the artifacts (preferred — you have
the CPU budget) or (b) mark them **UNSOURCED** and design triggers that do not depend on them.
**Do not silently inherit a number whose provenance you could not establish.** Say which you did.

## 4. WHAT TO DESIGN

A trigger set, each trigger specified as: **signal, window, threshold, intervention, expected firing
rate on the measured distribution, and what would prove it fired in the kernel log.**

Minimum coverage:

1. **Repeat-action trigger** (from C2/C3). Immediate repeats and coordinate re-clicks.
2. **Interior no-op trigger** — the state-didn't-change signal, keyed on the grid INTERIOR so that the
   ticking HUD cannot defeat it. Specify the crop/mask precisely enough to implement.
3. **Level-flat / dead-game trigger** — the agent is on a level it will never clear. 54.0% of acting
   turns are already in this state; the cap theorem says those turns are worth 0.
4. **Post-clear burn** — behaviour right after a level clears.

For each threshold, **state the firing rate it implies on the field floor's measured distribution**.
A trigger that fires on 60% of turns is a rewrite of the agent, not a supervisor; one that fires on
0.1% is the noop guard again. Say where you think the usable band is and why.

## 5. ADVERSARIAL DUTIES (non-negotiable — these are the campaign's own scar tissue)

- **PROVE EACH TRIGGER CAN FIRE.** For every trigger, run it (or simulate it) against REAL logged
  trajectories from `runs/kernel_pulls/` and report the count. A trigger with a computed firing rate of
  zero on real data is dead on arrival and must be reported as such, not shipped. This is the exact
  failure that produced `hard_noop_guard`.
- **PROVE EACH TRIGGER CAN REFUSE.** Also report a negative control: a synthetic or real HEALTHY
  trajectory segment on which the trigger must NOT fire.
- **DELIVERY vs USE.** If the intervention is a message to the model, the campaign has FIVE recorded
  instances of a mechanism that was delivered and never used (most recently P1: an affordance advertised
  only in a tool-call JSON schema got a ~1.3% use rate against a 30% bar —
  `learnings/war_room/p1_seed1_read_2026-08-23.md`). Put the affordance where the model demonstrably
  reads, and pre-register a USE-RATE bar that is checked BEFORE any effect is read.
- **BUDGET HONESTY.** The decision budget is binding: ~17 acting turns/game against a designed 132, and
  675/675 games die on the 7920 s clock. Any supervisor that costs LLM tokens competes directly with the
  agent's own deliberation. State the token cost per firing and defend it. Prefer interventions that
  cost ZERO generation (e.g. filtering the valid-action list) over ones that cost a turn.
- State the **strongest argument against S1 as a family** in its own section.

## 6. OUTPUT — write exactly one file

`learnings/war_room/s1_supervisor_prereg_2026-08-24.md` — a DRAFT prereg for coordinator review
(explicitly marked DRAFT / NOT SEALED; you are not authorized to seal it).

Structure: (1) one-paragraph claim and what would falsify it; (2) the trigger table with columns
signal / window / threshold / intervention / measured firing rate / log marker; (3) the firing-rate
evidence from real trajectories INCLUDING the negative controls; (4) sourcing status of every number
used — V / INF / UNK / UNSOURCED, with file:line or artifact path; (5) token cost and the budget
argument; (6) the case against; (7) what you could not establish and the artifact that would settle it;
(8) a pre-registered read: the exact bar, the comparator, and the decision rule — the standing
comparator for arms on this vehicle is field-floor lc 28 plus Arm A lc 30, giving **mean 29.0, pooled
seed sd 2.80**, and single-seed n=1 has MDE ~11 lc, so do NOT propose a single-seed effect read.

Cite everything. An uncited threshold is a guess wearing a number.
