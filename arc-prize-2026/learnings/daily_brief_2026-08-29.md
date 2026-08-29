# Daily Brief — 2026-08-29 (Saturday; build day, no panel)

Collection (STEP 1) ran automated before this session: `ARCMorningCheck` 06:00 and
`ARCCommunityBrief` 06:11 → **`learnings/community/brief_2026-08-29.md`** (2,603-row
board archive, 193 topics paged to exhaustion, tennant v28 diffed cell-by-cell,
external radar). This brief does not restate it; it records the **build** and the
**verdict**, plus two instrument failures found today.

## 1. RESULT DEEP-DIVE — yesterday's draw, and the window

**There was no 08-28 night draw.** Root cause is settled and closed: the
disarm/re-arm of `ARCDailySubmit` straddled the window, `LastRunTime` was stuck at
08-26 20:07, and the Mac rail (since **deleted**) never fired either. The 08-29
window was empty until submitted manually at **2026-08-29T11:41:14Z** —
`canivel/arc3-q38-field-eval` v1, the field-floor filler, `ok: true`, queue
auto-refilled to 1. Verified in `runs/submission_log.jsonl`.

**Interpretation, not the number.** The filler is insurance, not strategy. The
certified field-floor config stands unchanged at **n=7, mean 1.4686, sd 0.2897**;
a floor draw cannot reach the top 10 (P(≥2.80 on one draw) ≈ 2.2e-06) and today's
adds no information to the estimate we select on. Board: **#203/2,603 at 1.92**,
fifth straight day of pure rank drift on an unchanged score. The field-wide number
that matters: **278 teams submitted, 53 (19.1%) gained anything** — our own
"buying draws does not buy score" arithmetic reflected off 2,603 rows.

## 2. ★ TODAY'S BUILD — A30, THE UNTRIED-SET FIREABILITY GATE

Full sealed verdict: **`runs/untried_gate_0829/RESULT.md`**. KAOS `exp_id 66`,
**admitted to the public bench registry** (`tb1:e4196f46…`).

Two new instruments, both CPU-only, zero GPU / zero slot / zero model call:
`scripts/untried_probe.py` (662 archived passes, 27 arms, 33,820 turns) and
`scripts/action_profile_probe.py` (24 official games on the real offline engine).

**VERDICT: FIRES — and the same instruments discount it three ways.**

- **The gate fires.** 72.7% of 510 K=25 stagnation windows have a non-empty
  untried discrete set; 60.8% excluding ACTION7; 65.8% at K=10, 75.7% at K=50.
  `feedback_verify_treatment_can_fire` is satisfied — unlike banking, this
  treatment *can* fire on our rail.
- **★ The sharpest sub-finding: 53.1% of stagnation windows (271/510) occur with
  EVERY declared arrow key still unpressed**, while MOUSE was pressed **28,270**
  times pooled. More than half of all long stalls are the agent clicking MOUSE
  having never once tried moving.
- **★ ACTION7 is declared in 137 passes and pressed 0 times in 33,820 turns** —
  tennant's named case, on our rail, stronger than his (his: 0 in 12). The
  inversion beside it: **RESET is pressed 229 times while never being declared.**
  The agent reaches for the undocumented control and ignores the advertised one.
- **We do NOT replicate the field's 91%.** Our one-press level-1 liveness is
  **~53%**; the honest comparator is tennant's one-press figure of **69%**
  (411/600), not his 91%. ACTION7 is live in **2/6** games.
- **★ No outcome advantage.** Passes pressing the full discrete set reached mean
  max level **1.66**; passes leaving ≥1 untried reached **1.69**. Confounded and
  weak either way — but it is the only outcome-linked evidence we have and it does
  not support the graft.

**What this licenses:** the stagnation supervisor stays the right build, and now
has a concrete non-empty target at the moment it fires. **The rule the evidence
supports is narrower than `untried`:** *on a stagnation window, if no declared
arrow key has ever been pressed this game, press the unpressed arrows* — 53.1% of
windows, ≤4 actions against a ~600-action budget, ~53% measured one-press liveness.
**Not licensed:** adopting `untried` wholesale. **FALSIFIER 5 stands** — no
individual graft has board validation and this supplies none.

**Scope, stated so it is never over-quoted:** level-1 opening only (`GameAPI` has
`number_of_levels` but no setter); ACTION7 liveness n=6; 1 of 25 games failed to
start (`cn04-65d47d14`, `base_actions_per_level has 6 entries; number_of_levels is
5`) and is excluded, not silently dropped; 92 passes carried no declared line and
are reported UNMEASURED.

## 3. ★ TWO INSTRUMENT FAILURES FOUND TODAY (`feedback_audit_the_instrument`)

**(a) KAOS agents cannot do this campaign's work.** Two hard blockers, both new:
`kaos run` did **not expand `@file`** (the agent received the literal path string
as its task), and KAOS agents run in an **isolated virtual filesystem** — they
cannot read `runs/kernel_pulls/` or write real artifacts. A stored result from an
earlier agent confirms it verbatim: *"Write was also not approved, so I'll report
inline."* Separately `kaos doctor proposer` reports **`fable-panel` and
`opus5-panel` both wall-timeout at 30s**; only `opus5-code` passes. **Consequence:
the KAOS-native mandate's spawning mechanism cannot be used for any task that
reads or writes local files**, which is nearly all of them. The mandate's *intent*
(queryable journal, portable learnings) was met by other means today — `experiment
log` + bench push both worked, and the verdict is in the public registry. Owed:
file these upstream per the KAOS AI-feedback policy.

**(b) My own probe was wrong twice before it was right.** The first run reported
plausible tables built on two parser defects: `MOUSE(row=23, col=60)` was split on
its internal comma into junk tokens, and the declared/executed regexes matched the
**model's prose quoting the prompt back** rather than the harness-authored
`[USER PROMPT]` block — harvesting the agent's speculation as ground truth. Both
are fixed and commented in `scripts/untried_probe.py`. Had the first output been
written up it would have read as a finding. The rule held: audit the instrument
before the verdict, not after.

## 4. OPEN QUESTIONS (for tomorrow's Sunday panel)

1. **Does the arrow-key sub-case survive at depth?** Our 53% liveness is a level-1
   number. Driving deeper levels by real play is free CPU and would either promote
   or kill the licensed rule. **This is the natural next build.**
2. **Is the 1.66-vs-1.69 null real or confounded?** Stratifying by declared-set
   size and by game would settle it cheaply on data we already hold.
3. **Compaction arm** (community brief handoff #2, unstarted): re-scope the 08-28
   "cheapen the decision" closure to **truncation only**, and screen a *within-run*
   per-game manual on a **delivery/use metric before any score metric**
   (mech-C delivered at 96.3% with no behaviour change; P1 died at 1.3% use).
   Public-game manuals are forbidden by `feedback_arc_generalization_first`.
4. **Post-fire assertion for the submit rail** — the daemon verifies its own
   submissions well (row-diff + retries), but nothing asserts that a UTC window
   closed *covered*. Silence still reads as success at the window level.
