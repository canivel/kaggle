# DAILY BRIEF — 2026-08-23 (Sunday; weekly-consolidation day)

**Session type:** Sunday. Full panel is IN SCOPE (weekday panels are suspended per the 07-27 restructure).
**GPU spent today:** 0. **Kernel pushes today:** 0. **Submission slots used today:** 0 (tonight's fire is the 00:07 queue head).

---

## 1a. RESULT DEEP-DIVE — no new score landed since the last read

The overnight draw of **1.63 COMPLETE** (submitted 2026-08-23 00:07:10) was already pulled, interpreted and logged under the
08-22 ITERATION_LOG entry, which was written after midnight. **There is no unread score today.** What changed today is only the
*status* of that number once it is read as a config rather than a draw:

**The certified Q38 field-floor config is now 3/3: 1.59 / 1.58 / 1.63.**
- config mean **1.6000**, sample sd **0.0265**, **sem 0.0153**.
- This is the campaign's first config with a *replicated level* rather than a lucky maximum, and it is the exact statistic the
  **final-selection invariant** reads (`project_arc_final_selection_rule`: pick the two private twins by CONFIG MEAN, never by public max).
- Pre-registered expectation MET and then some: the sealed prereg called this a "typical draw of ~N(1.6, 0.2)". The realised spread is
  **0.0265, i.e. ~7.5x tighter than the assumed sd**. That is worth flagging as its own finding — *we have been pricing draw variance
  on this config far too pessimistically.* Caveat before anyone spends it: n=3 gives the sd itself only 2 df; the 95% CI on a 3-sample sd
  runs roughly 0.014–0.17, so the honest claim is "materially tighter than 0.2", not "0.027".

**The consequence that matters, stated plainly:** with sem 0.0153, a **4th redraw of this config buys essentially nothing** — it would
move the mean estimate by ~0.01 and cannot change any decision we face. The floor is measured. Further redraws are not evidence-gathering,
they are lottery tickets on a public max that the final-selection rule explicitly ignores.

**And the floor is not a path to gold.** +0.04 on the last draw bought **-11 ranks** (#239 to #250 of 2489); the gold line is **2.50**.
Redrawing this config asymptotes near ~2.0 as a *max over draws*, which is a display number with no private-board meaning.

## 1b. DISCUSSIONS SWEEP — **no new posts since the last sweep**

Feed pulled via `kaggle competitions topics list -c arc-prize-2026-arc-agi-3 --sort-by recent` (CLI 2.2.2). Newest two topics are
**736578** (Public vs. Private Discrepancy, 08-21 15:56Z) and **736540** (non-official games for training, 08-21 12:05Z). **Both were
already evaluated and dispositioned in `daily_brief_2026-08-22.md`.** Nothing has been posted in the ~44 h since.

Restated for the record because it remains the most load-bearing external datapoint we have:
- **736578 — ADAPT (unchanged).** Pellegrin reports duck+Q3.8 local 2.1 -> LB ~1.4, own harness local 5.0–5.4 -> LB **still ~1.4**.
  Our own answer is on file (`war_room/local_lb_transfer_2026-08-22.md`): the failure mode exists in *our* record too but resolves as a
  **single-seed artifact, not a transfer failure** (war-v1 read +3.16 sigma on one seed, 22/15/13 across three, and its family mean 16.67
  predicted the null LB correctly). Our one large replicated local effect **did** transfer near-proportionally: local x1.84 -> LB x1.70,
  agreeing to 8%. The untested cell is his: a **from-scratch** harness has far more freedom to overfit 25 public games than our
  duck-lineage fork does. No change to our gates.
- **736540 — IGNORE (unchanged).** Third-party non-official games; no bearing on a 25-game public / hidden-private scored rail.

## 1c. RESEARCH SWEEP — **no new results**

`arXiv:2607.03441` (Agentic Test-Time Training) and `arXiv:2511.04847` (Test-Time Adaptation via Environment Interaction; WebArena
multi-site 2% -> 23% via deployment-time dynamics grounding) both re-surfaced. **Both are already on file** and dispositioned
(`artifacts/research_sweep_2026-07-27.md`, briefs of 07-19/07-25/07-28, and 08-22). Disposition unchanged: **ADAPT-not-ADOPT** — the
transferable idea is *search over observed dynamics*, which is precisely the gap the per-turn program measured independently
(agent queries `transitions` in 16.3% of generations but shows search idioms in 5.8% and explicit candidate scoring in **0.2%**).
It is already the surviving program; nothing in these papers is a drop-in for a 27B served model under a 7920 s/game clock.

## 1d. WEEKLY MECHANICS (Sunday)

**FAILURE FINGERPRINTS — writer run first, then reader (protocol order).**
`fingerprint_backfill.py` reported **2 NEW incidents** the store on disk did not have — so the reader would again have described a stale
store had the order been reversed. Post-write the reader asserts **`store FRESH: 51 retained logs all scanned`**.

| family | n | first | last |
|---|---|---|---|
| class:ERROR:none | 7 | 2026-05-26 | 2026-06-28 |
| provenance:scratch-built | 5 | 2026-05-26 | 2026-06-28 |
| slug:canivel/arc3-final | 4 | 2026-05-26 | 2026-06-10 |
| class:COMPLETE:0.00 | 3 | 2026-03-29 | 2026-06-10 |
| slug:canivel/arc3-forge35 | 3 | 2026-04-24 | 2026-06-22 |
| slug:canivel/arc3-pilot-eval | 3 | 2026-07-07 | 2026-07-08 |
| t1:07d0f5248c48401d | 3 | 2026-07-07 | 2026-07-08 |
| class:COMPLETE:null-band | 2 | 2026-06-01 | 2026-06-08 |
| slug:canivel/arc3-a17-72b-canary | 2 | 2026-07-25 | 2026-07-25 |
| t1:fb1e96c3815797ad | 2 | 2026-07-25 | 2026-07-25 |

23 incidents / 11 recurring families / 5 deaths flagged-in-advance (strict). New: `inc-t1-010` (08-17, q38low) and `inc-t1-011`
(08-18, graft-floor v2) — both already-known events, now indexed. **`provenance:scratch-built` (n=5) remains the single most
expensive family in the campaign** and is exactly what `feedback_arc_kernel_structural_drift` and `preflight.py` now block.

**KAOS consolidation.** `kaos_ingest.py` -> inserted 7 / unchanged 218 / total 291. `kaos dream run` -> digest
`Dreams/2026-08-23-122518.md` (17 episodes, 4 ok / 12 failed, dry_run, 0 skills promoted — the expected shape; skills never auto-promote).
Hot memory is dominated by the six war-room documents written on 08-22, i.e. the consolidation is tracking the live program.

**Bench.** `kaos bench rejections` -> `{"rejections": []}`. Still no cross-workspace pull upstream; the gap remains filed.

## 1e. STATE OF THE RAIL

- **Kernels:** all five recent slugs (`q38-field`, `q38-graft`, `duck-repro-pathsafe`, `q38-private`, `graft-floor`) report COMPLETE.
  **No build is open.** No pull is pending.
- **Queue:** non-empty. Head = `canivel/arc3-duck-repro-pathsafe` v1 (pathsafe insurance fork).
- **Ledger** (`runs/ledger.json`): **n=37, mean 0.9316, sd 0.1771, latest_date 2026-08-20** — this is the NULL/filler family and it is
  correctly dated; the 1.59/1.58/1.63 draws are treatment and do not enter it. Promotion bar **1.089**.
- **GPU week boundary:** the weekend-prep lane's standing finding is that the accounting window appears to open **Monday**
  (31.4 GPU-h reconstructed for 08-17..08-22, reproducing two independent coordinator checkpoints only under a Monday open).
  **It has never been observed** — no Kaggle-side quota banner or refusal exists anywhere in our logs. Treat as inferred.

---

## 2. TONIGHT'S HEAD — the recommendation put to the panel

**Recommend: leave the pathsafe filler as the head. Do NOT redraw the field floor a 4th time.**

1. The floor is already measured to **sem 0.0153**; a 4th draw changes no decision (section 1a).
2. Public max is **not** what the final-selection rule reads, so a lucky redraw is worth exactly zero private information.
3. The pathsafe fork is **insurance that has never been scored end-to-end on a competition rerun**, and `arc3-duck-repro` — the entry it
   replaces — is retiring. Submitting it converts an untested fallback into a certified one.
4. **It cannot cost anything.** The public score is a max over submissions; a null-band pathsafe draw cannot displace the banked 1.63.

## 3. MONDAY — three sealed arms, two slots. The panel's job is to rank them.

All three are sealed, pre-data, with instruments validated **before** their data lands (`feedback_audit_the_instrument` discharged):

| arm | prereg | claim under test | self-registered prior |
|---|---|---|---|
| **C1 `cadence-effort`** | `cadence_prereg_2026-08-22.md` | bounding deliberation (`reasoning_effort=medium`) converts into levels | **P(SIGNAL) ~ 20%** (author's own) |
| **P1 seed 2** | `p1_notes_prereg_2026-08-22.md` | owed replicate; a single seed is never read | — |
| **P2 reset-retry** | `p2_reset_retry_prereg_2026-08-22.md` | retry allowance where 20/25 games sit at k<=1 | — |

The cadence prereg orders **C1 = first slot of the new GPU week, unconditional**, and states its own strongest counter-evidence:
`reasoning_effort=medium` has run twice on Q3.8 and delivered the mechanism (-48% tokens/turn, +72% acting turns) while scoring
**lc 21 and 17 against the floor's 28** — on the *June-30 vehicle*, never on the current floor, n=1 per point. **KILL-F** is designed so
that a delivered-but-null C1 retires the uniform-effort branch and a delivered-but-null C2 retires the whole family.

**Open questions for the panel (ranked):**

1. **Is the cadence family worth the first slot of the week at a 20% self-declared prior?** Its *positive* value is mostly the KILL — it
   closes the last "more of the same currency" lever. Is buying a kill the best use of the week's first, cheapest-to-verify slot,
   given the Monday quota boundary has never actually been observed and slot 1 is also the natural place to test that boundary?
2. **The SCREEN-SHAPE defect may dominate all three arms.** taaf's own submission benchmark is **110 runs = 25 games x ~4.4 clones**
   (`make_benchmark_kaggle_official_110`); our rail screens **1 clone**. Our sealed lc bands, our pooled sd 2.80, and every kill
   criterion above are computed on a screen whose *shape* differs from the thing being scored. Does this invalidate reading C1/P1/P2 on
   the current screen, or is it orthogonal? Note the cost is **not** free in GPU (~4.4x runtime, ~10 GPU-h vs 2.3), only free in code.
3. **The consistency lever vs the capability lever.** Per-turn arithmetic says recovering *all* wasted turns caps out at **lc 32 / LB ~1.78** —
   it cannot reach 2.50. Clone-consistency on near-certain level-1 games (bp35 / r11l / sp80) is worth **~+3 lc of pure consistency**,
   but that estimate is a **Qwen3.6 property** (weekend-prep's correction) and may not survive on Q3.8. Which lever gets the week?
4. **Do we still believe the 2.50 gold line is reachable on this program at all**, and if not, what is the honest objective for the
   remaining window (entry deadline 2026-10-26, Milestone 2 on 2026-09-30)? A defensible answer of "maximise a certified, replicated
   config and bank the milestone" is a legitimate output of this panel.

---

# ADDENDUM (same session, written BEFORE the panel verdicts were read) — I overstated the variance finding in 1a

**What is solid.** The three draws are three competition reruns of the **identical artifact**: `canivel/arc3-q38-field-eval` **v1**,
submitted 08-21 / 08-22 / 08-23 (daemon log, `runs/daily_submit_stdout.log`). So the 0.05 spread is *pure rerun noise on one fixed
build* — not a comparison across builds. That part strengthens the config-mean reading.

**What I overstated.** I wrote that the realised sd is "~7.5x tighter than the assumed sd" and let that stand as a finding.
Two problems:

1. **n=3 gives the sd 2 df, and that is a very wide sampling distribution.** Testing the observed sd against the null family's
   0.1771: chi-square stat 0.0446 on 2 df, **P(sd <= 0.0265 | true sd = 0.1771) = 0.022**. Suggestive on a single comparison,
   nowhere near sealed, and we have not been counting comparisons.
2. **The 0.1771 is not a clean same-artifact control.** The ledger's n=37 mixes forks, slugs and months, and the frozen duck fork's
   own draws run **0.41 to 1.33** — a ~0.9-wide range on what is largely one artifact, including the 0.41 tail draw this campaign
   already de-escalated as unexplained. **Same-artifact reruns demonstrably CAN swing hard on this platform.**

**Corrected claim:** the field floor's rerun spread *looks* tighter than the null family's (p ~ 0.02, single comparison, 2 df), and
the config mean is **1.600**. Whether the true rerun sd is 0.03 or 0.18 is **not settled by n=3**.

**Does this change tonight's recommendation? No — but for a different reason than I gave.** My argument was "sem is 0.0153, so a 4th
draw is worthless". Under the pessimistic sd the sem is **0.1022**, and a 4th draw would tighten it to 0.089 — a real reduction.
The recommendation survives because **no decision we currently face turns on +/-0.1 of this config mean**: the field floor is our only
certified config, the gold line is 2.50 and the prize line ~1.90, and the final-selection rule only needs to *rank* configs against
each other. Replication precision starts to matter the moment we have **two** certified configs within ~0.1 of each other — at which
point redraws become genuinely valuable and should be budgeted. Flagging that now so it is not rediscovered late.

**Consequence for the panel's question 1:** the pessimistic sd also means our LB instrument is coarser than section 1a implied, which
*raises* the relative value of the local rail (sealed lc bands) over LB draws for reading arms — and therefore raises the stakes on
question 2 (the screen-shape defect), since the local rail is then carrying more inferential weight, not less.

## Rail verification run this session (zero GPU)

- `scripts/local_gate.py --self-test` -> **PASS 13/13, 0 fail** (40.9 s). Includes **S13 `cadence_instrument_can_refuse`** — Monday's
  C1 instrument is proven able to report failure against a poisoned expectation, and **S10 cross-arm refusal** still fires.
- Daemon: healthy, only `already-submitted-today` skips; queue head correctly armed for tonight's 00:07 fire.
- **Monday prerequisite discovered:** `local_gate.py --arm` currently registers 10 arms and **none of them is a cadence arm**
  (`budget-t05, budget-t3, graft-confirm, graft-floor, private-base, private-edge1, private-edge12, private-edge2, q38-field, q38-graft`).
  C1's per-arm certification suite therefore does not exist yet. The cadence instrument (P9) and its negative control (S13) DO exist.
  **The Monday build session must register the C1 arm before pushing** — that is a cadence-lane edit, not this session's to make
  (one-lane-one-operator, 08-18 ruling).
- **GPU-week boundary: NOT observed this session.** The Chrome profile is locked by a running browser, so the web-console quota page
  could not be read without killing the user's session. The boundary remains **inferred**. Weekend-prep's advice stands: Monday,
  cheapest build first.

---

# ADDENDUM 2 — the day's actual result, found AFTER this brief and its panel questions were written

**P1 seed 1 was completed on Kaggle on 08-22 and had never been pulled.** It was pulled and read today at zero GPU.
Full read: `learnings/war_room/p1_seed1_read_2026-08-23.md`. KAOS **exp 44**, admitted to the public registry (`tb1:e52945cc…`).

**VERDICT: `DELIVERY FAILURE` (proxy-based) under an INFRA-DEATH certification caveat.** The patch *applied* — the kernel
wrote out its patched source tree carrying all three `[notes]` markers and the swapped tool description verbatim — and the
run was healthy (25 games, 1,502 actions). But the model referenced the injected namespace **5 times across all 25
transcripts** against ~400 acting turns (validated cadence instrument) = **≈1.3% write-rate versus a sealed 30% bar**.
Descriptive only, not read: lc 27 / mean_score 4.762 / trim1 4.057. Certification is *unevaluable* because the Kaggle API
returned a **0-byte kernel log on two pulls** (every other retained pull carries 250 KB+) ⇒ strictly INFRA DEATH #1.

**This changes section 3 of this brief.** The Monday table above lists "P1 seed 2 — owed replicate". **It is no longer owed.**
A DELIVERY FAILURE seals to *re-scope, never re-read*; the INFRA reading seals to *re-run, not advance*. Neither licenses
seed 2. Panel question ordering is unaffected for C1, but the slot-2 line item is withdrawn.

**It also partly answers panel question 3.** The mechanism finding — an affordance advertised **only** inside a tool-call
JSON schema description is not advertised; the model was reading the *stock* guidance block instead (`Cross-level notes:`
159 hits) — is direct evidence for **DELIVERY-WITHOUT-USE** over "the agent forgot". The per-turn program's surviving
lever (search over observed dynamics) has the same delivery hazard and should be designed to advertise itself where the
model demonstrably reads, not in a schema field.

**And one defect that touches every arm:** `benchmark.json.label` does not identify the arm — P1's artifact and the
certified field floor's carry the byte-identical `anim-20260807-anim-25g-p1`, where `-p1` is **pass 1**, not the P1 arm.

**Panel status:** rounds 27 and 28 both died on KAOS infra (sandbox refusal, then 0-char failures) — see the ITERATION_LOG
entry. Rounds 16–26 are verified uncontaminated, so the historical "0 accepts" record stands. The strategic questions in
section 3 above are carried unanswered to the coordinator.
