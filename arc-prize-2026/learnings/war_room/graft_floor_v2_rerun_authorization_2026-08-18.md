# GRAFT-FLOOR v2 — DELIBERATE IDENTICAL RE-RUN: authorization record
**Date:** 2026-08-18 (iterate session, ~08:30 EDT) · **Slot:** 08-18 slot **2 of 2** (the last one)
**Arm:** `canivel/arc3-graft-floor-eval` v2 · **Prereg (unchanged, sealed 08-17):** `learnings/war_room/graft_floor_prereg_2026-08-17.md`

## Why an identical re-push is authorized
v1 (pushed 06:47 EDT, slot 1) died at **t=7s** in the **STOCK** wheels-install cell — papermill
`In[2]`, code ordinal 1, **not** one of our modified cells `[2,6,12]` — because the competition
data mount `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels` was absent at
runtime. Per prereg §4 that is **INFRA DEATH, never a NULL/HARM**: zero GPU-minutes of evidence,
the arm's own code unexercised and unimplicated (its graft banner printed cleanly before death).
The mechanism is untested **in either direction**, so the arm still owes the campaign a reading.

Platform-recovery evidence re-checked this session, all green: `competitions files` lists
`arc_agi_3_wheels/arc_agi-0.9.8-py3-none-any.whl` live; `kernels status`, `kernels pull`,
`kernels list` all answer cleanly; the SaveKernel 500 that co-occurred with the push has not
recurred. **Honest limit: this is API-plane health, not a positive observation of a successful
post-incident *runtime* mount.** No cheap probe for the latter exists — a probe would itself cost
the slot. Accepted risk, stated in advance: if v2 dies the same way, that is a *second* INFRA
DEATH and it escalates from "transient" to a rail defect requiring diagnosis, not a third re-run.

**Code is byte-identical to v1** (ASCII-identical; `code_sha256=3c047dff2e6c02fd`, 17 cells,
differing cells vs frozen fork `[2,6,12]`). Nothing about the arm is being changed — that is the
whole point: a re-run is only interpretable if the only thing that moved is the infrastructure.

## The guard this trips, and why it had to be fixed first
Gate **1b (idempotence)** exists to stop a scarce slot being spent on a no-op re-push. It compared
an **exact** code sha — and on this rail it was **BROKEN OPEN and had never once been able to
fire**: Kaggle's push path re-reads our UTF-8 as cp1252, so the frozen fork's own em-dashes come
back mojibake'd (U+2014 → `â€"`, verified this session: remote is *exactly* local misdecoded, all
8 code cells) and the remote can never equal the local. Step 3's pull-back verify already knew
this (**VERIFIER FIX 1**, inherited from the q38 arm 2026-08-16, which is why this morning's
"code sha match" was honest and not a contradiction); **step 1b did not.**

1b now applies the same normalisation — ASCII-visible drift = a genuinely different notebook,
non-ASCII-only drift = the SAME code — and **it was verified to actually REFUSE on this exact
push before the override was set** (exit 3). The bypass is a new explicit
`GRAFT_ALLOW_DUPLICATE=1`, which is what this file records the reason for.

*Standing lesson, fifth instance this week: **silence from an automation is not success.** A gate
that has never fired is indistinguishable from a gate that cannot.*

## Authorization
- `GRAFT_ALLOW_V2=1` — the slug now exists (v1); a v2 needs a fresh slot + fresh authorization. Slot 2 is it.
- `GRAFT_ALLOW_DUPLICATE=1` — identical code is *intended*; see above.
- Slot 2 was **held** by the coordinator for exactly this lane ("hold it for your lane; q38-low v2 yields").
- Reading rules **unchanged** from the sealed prereg: LEVELS-primary, SIGNAL `lc_total >= 27` ·
  NULL `13..26` · HARM `<= 12`; uncertifiable install = INFRA DEATH, never NULL; `shortcircuit`'s
  score effect recorded as NON-INFERENTIAL. **Do not read a levels number before the scorer
  certifies the graft banner.**
