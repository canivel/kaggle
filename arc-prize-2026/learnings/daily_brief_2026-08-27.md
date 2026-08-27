# DAILY BRIEF — 2026-08-27

**Session type:** iterate (weekday — no panel per the 07-27 cadence ruling).
**Slots spent:** 0 of 2 Kaggle pushes. **GPU spend: $0.** **Submissions made by this session: 0.**
**Registry:** 3 verdicts logged and **admitted** to the public bench (exp **57**, **58**, **59**).
**Sweeps 1b/1c** were executed at 06:00–06:11 by `ARCMorningCheck` → `learnings/community/brief_2026-08-27.md`
(61 topics paged, LB archive 2564 rows, Polyphony source read in full). **Not duplicated here**; this
brief carries 1a (result deep-dive) and the day's rulings.

---

## 1a. RESULT DEEP-DIVE — P2 IS CERTIFIED AND IT DIED ON DELIVERY

Full read: `learnings/war_room/p2_read_2026-08-27.md`. **Verdict: CERTIFIED / DELIVERY FAILURE.**

The artifact had been sitting **COMPLETE and unpulled** since 08-26 — the day-late read the 08-26
handoff flagged as today's first action. Pulled, certified, scored.

| gate | bar | measured | result |
|---|---|---|---|
| D1 trigger fires | ≥15/25 games | **24/25** | PASS |
| **D2 `attempt()` use-rate** | **≥25 %** of armed turns | **10.73 %** (75/699) | **FAIL** |

Ceiling on the tightest defensible denominator (75/408 acting turns) is **18.4 %** — **the gate fails
on every reading.** Per prereg §5 this licenses **no lc verdict**, doesn't count against the kill
budget, and triggers **re-scope-never-re-read: seed 2 is REFUSED.**

**Was the pre-registered expectation met? Half of it, and the half that was pre-measured is the half
that worked.** D1's trigger-fireability was measured on retained artifacts *before* the build
(predicted 19/25, observed 24/25 — excellent). D2 was never pre-measurable, and D2 is what killed it.
**That asymmetry is the finding**, and it is now the standing rule for this arm family.

**Why this null is worth something.** Delivery is *not* in doubt: the patch certified, the boot
invariant held, the trigger armed on 24/25 games. So the null is about the **agent**, not the pipe.
And **24 of 106 `attempt()` calls came on UNARMED turns** — the model knew the affordance existed and
reached for it unprompted. **This is a preference failure, not a discovery failure**, so making the
tool more prominent is the wrong repair.

**Third paid confirmation of `feedback_advertise_where_model_reads`** (96.3 % delivery/no behaviour
change → 1.3 % vs a 30 % bar → **10.7 % vs a 25 % bar**). The memory predicted this before the build.

**Comparator table — read it with the MDE in front of you.** All n=1 against an lc MDE of **11.1**:

| artifact | lc | mean | **trim1** |
|---|---|---|---|
| certified field floor | 28 | 6.173 | **3.189** |
| P1 notes | 27 | 4.762 | **4.057** |
| exec-WM | 25 | 3.006 | **2.330** |
| **P2 retry** | **24** | 3.232 | **2.672** |

**Not one of these is distinguishable from the floor or from each other.** The floor's headline mean
is **50.4 % one game** (sb26 = 77.78), and on `trim1` the floor ranks **third of four**. The ranking
flips with the statistic — `feedback_screen_calibration_range`, again.

---

## 2. RULINGS ISSUED TODAY

### (i) EXEC-WM — the two-day-outstanding item, CLOSED, and the standing diagnosis OVERTURNED
`learnings/war_room/execwm_break_audit_2026-08-27.md` · exp 58

The BREAK-clustering hypothesis is **refuted by the arm's own artifact.** The prediction-BREAK latch
fired **zero** times; the threshold was reached **once**, via the unrelated `sprite-lost` path; both
level-instances *labelled* `prediction-breaks` fell back at `breaks=1` through the **second disjunct**
of a shared condition. **Changing the threshold would have altered 0 of 31 fallbacks** — so the repair
this brief's own handoff #2b recommended would have done nothing.

**The defect that cost two days is a mislabelled reason string**: one code path, two disjuncts, one
shared label. `feedback_audit_the_instrument` in a new costume — not a gate that couldn't fire, but a
gate **reporting someone else's firing as its own.**

Real cause: **data starvation.** 26/32 level-instances fell back at `no-verified-model`; **9/18 games
yielded zero usable transitions despite 16–20 probes per level.** Where data existed the model was
accurate — **retrodiction 810/818 = 0.990** (124/132 = 0.939 excluding the dominant game), and one
level was cleared **by pure BFS at zero LLM tokens** with no floor damage.

**Ruling: kill clause does NOT fire; arm is NOT naively re-seeded** (failure is deterministic, not
stochastic). **Re-scope to "repair the observation layer, then re-seed"** — CPU-only, slot-free.

### (ii) STICKY POLICY DEADLINE — refuted pre-build, for ten minutes of CPU
`learnings/war_room/sticky_deadline_fireability_2026-08-27.md` · exp 59

The morning brief's handoff #3 ("if only one point is built today, build this"). **Killed by its own
fireability check before a line was written.** On the certified floor, **39.3 % of level completions
and 47.1 % of last-level completions land after the 0.55 mark**. The back half of our clock is **not
slack — it is where 30–40 % of our score is made.**

And the failure the deadline prevents isn't ours: Polyphony guards against *"an elegant policy and
zero actions played"*; our agent plays **2081 actions**. We have a throughput problem, not a paralysis
problem — **opposite interventions.** (Structural mismatch also recorded: their deadline switches
between two *phases*; our vehicle has no phase boundary to cut.)

**This is today's P2 lesson applied the same day.** One arm cost a slot and 2h13m of GPU to learn it;
the next cost ten minutes.

**Reusable prior banked:** level completions per-run-normalised — p50 **0.310**, p90 **0.850**. Any
future arm proposing to truncate or re-budget the back half of the clock must clear that table first.

---

## 3. INSTRUMENT FINDINGS — three, all new, all operational

1. **`kaggle kernels output` can return EXIT 0 with a PARTIAL file set.** Pull 1 of the P2 artifact
   silently omitted **both** large logs (256 KB kernel log, 360 KB vLLM log); an identical re-pull
   minutes later got them. Scored against the partial pull, `p2_score.py` returned **INFRA DEATH /
   "served model absent" / `cert_facts: {}`** on a completely healthy 2h13m run. **A download race
   nearly killed a certified arm.** Tell: *an instrument reporting a sick subject while reporting zero
   facts about it is describing itself.* **Fix not yet built** — see open items.
2. **P1's 0-byte kernel log is REAL and is a different defect.** Third independent pull: 0 bytes,
   while its sibling logs arrived intact **in the same pull**. The 08-23 finding **stands, re-confirmed**.
   Two failure modes, separated today.
3. **KAOS rail: partially unblocked, and I over-claimed mid-session.** Correcting: the `opus5-code`
   entry **does now forward tools** (superseding the 08-24 "text-only" finding), and the sandbox root
   is the **process cwd** — so spawning from the campaign repo lets agents **read** repo files, which
   they previously could not. Verified. **But writes and Python execution remain denied**, so the rail
   is **read/analysis-only, not build-capable**. Working invocation and this caveat are recorded in
   `runs/lane_locks.json`. Also: `kaos bench push` needs `--config-file /f/kaggle/kaos/kaos.yaml` when
   run from the campaign repo, or it silently reports *"local-only mode"*, `pushed: 0`, **exit 0**.

**Both KAOS agents refused to invent numbers when blocked** and reported the blocker instead. That is
the behaviour we want and it is why their partial output was still usable — the exec-WM agent's
analysis, machine-verified here, overturned a diagnosis two briefs had asserted.

---

## 4. BOARD (from the 06:00 archive — no action, recorded)

Us **1.92, #182** (−23 ranks on field drift, 123 subs, unchanged for a 4th pull). Floor config now
**n=6, mean 1.5033, sd 0.3010** (draws 1.59/1.58/1.63/1.16/1.92/1.14). Top 30 flat for a 5th day
(26/30 gained exactly 0.00); the two real steps were **rfbr +1.18 → 3.37 on 13 lifetime subs** and
**MindsAI +0.89 → 2.94, #115→#7** (verified same TeamId + roster, so a capability step, not a merger).
Prize line **3.37**, gold **2.70**, gap to gold **0.78 and widening**. Method **UNKNOWN** for every mover.

**The 08-26 exec-WM board scare is formally dissolved**: the floor itself drew **1.14** the next night,
within 0.09 of the arm it was supposed to indict. Third time this month a board number nearly bought a
wrong conclusion.

---

## 5. TONIGHT'S HEAD — deliberate, not a default

**Head: certified field floor** (`canivel/arc3-q38-field-eval` v1, trusted-fork). This is **not** the
auto-refill accident it was on 08-25 and 08-26 — the queue message has been corrected to say so.

Reasoning: P2 is dead on delivery and **not** head-eligible; exec-WM is re-scoped to a CPU repair and
**not** head-eligible; the sticky arm was refuted before build. **No arm cleared a promotion gate
today, so no arm heads.** The floor is our best config (best-ever draw 1.92) and
`project_arc_final_selection_rule` selects the final two by **config mean**, so an extra floor draw
tightens the estimate we will actually select on. That is a real, if modest, return — and it is the
honest reason, not "the queue was empty."

---

## 6. OPEN QUESTIONS / TOMORROW'S FIRST ACTIONS

1. **Repair the exec-WM observation layer** (ruled, slot-free, CPU-only): replay the retained
   `artifacts/*_events.jsonl` and find why transition extraction returns **empty in 9/18 games**.
   **Nothing else about this arm matters until n > 0.** Then the retrodiction-ratio gate (justified by
   the g50t case — 16/16 = 1.000 discarded for having only 1 MOVE among 4 verified rules — **not** by
   the refuted BREAK story), then split the mislabelled reason string.
2. **Build the pull-completeness assertion.** Verify the kernel `.log` is present and non-empty before
   any scorer runs; re-pull if not. Widen `certify()` beyond a top-level `*.log` glob —
   `taaf_setup_env.json` and `prompts/*.log` both survived the partial pull and both carry the served
   model. This defect is now **campaign-wide**, not P2-specific.
3. **Level census still unbuilt** (morning handoff #1, free, verified premise). The KAOS agent produced
   a good metric design with positive/negative controls but couldn't execute. `set_level` is confirmed
   working on our `.venv` and 25 game dirs load. **Carry forward — build inline, not on the KAOS rail.**
4. **Standing, unanswered:** what separates Polyphony's 19.8 % from Retrodict's 99.9 % on the same
   mechanism class? Both have a verifier. **The idea is not the hard part; the implementation is** —
   and that is the correct prior for costing every mechanism-adoption proposal we have queued.
