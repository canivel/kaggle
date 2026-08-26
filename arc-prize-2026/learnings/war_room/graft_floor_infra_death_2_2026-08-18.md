# GRAFT-FLOOR v2 — **SECOND INFRA DEATH**, and the cause is NOT our arm
**Date:** 2026-08-18 · **Kernel:** `canivel/arc3-graft-floor-eval` v2 · **Slot:** 08-18 slot 2 of 2 (both now spent)
**Prereg:** `learnings/war_room/graft_floor_prereg_2026-08-17.md` (sealed 08-17, UNCHANGED)
**Authorization for the re-run:** `learnings/war_room/graft_floor_v2_rerun_authorization_2026-08-18.md`
**Log:** `runs/kernel_pulls/graft_floor_v2/arc3-graft-floor-eval.log`

## VERDICT: **INFRA DEATH (2nd consecutive). NOT a NULL, NOT a HARM. The mechanism remains untested in either direction.**
Pushed 12:28Z, QUEUED 12:28–12:34Z, **ERROR 12:35Z**. Death at **t = 6.1 s**, papermill `In [2]` —
the **STOCK** wheels-install cell, code ordinal 1, **not** one of our modified cells `[2,6,12]`:

```
WARNING: Location '/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels' is ignored:
         it is either a non-existing path or lacks a specific scheme.
ERROR: Could not find a version that satisfies the requirement arc-agi (from versions: none)
CalledProcessError: ... pip install --no-index --find-links .../arc_agi_3_wheels arc-agi  -> exit 1
```
Byte-for-byte the same death as v1 (10:47Z, t=7 s). **The arm's own code never executed.** Its banner
printed cleanly at t=4.98 s with the correct flag set and `FORBIDDEN=banking+transfer`, so the arm is
unexercised and unimplicated.

## The 08-18 morning diagnosis ("transient platform instability") DOES NOT SURVIVE
That read rested on the SaveKernel 500 that co-occurred with the v1 push. A transient does not
reproduce identically 1h41m later on a clean API. **Escalation was pre-committed in the v2
authorization file, before the outcome was known: *"if v2 dies the same way, that is a second INFRA
DEATH and it escalates from transient to a rail defect requiring diagnosis, not a third re-run."***
That is binding. **No third push.**

## Hypotheses, and what actually discriminated them
| # | Hypothesis | Status |
|---|---|---|
| H1 | Transient platform blip | **DEAD** — reproduced exactly, 1h41m apart |
| H2 | Fresh-slug defect (new kernel can't mount competition data) | **DEAD** — `arc3-q38-low-eval`, also a fresh slug, died at `In [4]`, i.e. **its competition mount worked** |
| H3 | Total input size / too many sources | **DEAD** — the working `arc3-duck-repro` attaches the *same* 35.9 GB engine + wheelhouse + a bundle; graft-floor's only delta is a small source tree |
| H4 | Our metadata is wrong / competition source dropped | **DEAD** — remote metadata pulled and compared against two kernels whose mounts work: `competition_sources`, `enable_gpu`, `enable_internet`, `machine_shape` all **identical**; the sole difference is the bundle dataset ref |
| H5 | The `thtennant` fork dataset specifically breaks the mount | **WEAKENED, not excluded** — it is the only remaining local variable, but it is a small additive source tree and there is no mechanism by which attaching it would unmount a *competition* input |
| **H6** | **Kaggle-side input-mounting defect, live today** | **SUPPORTED — and it is the leading explanation** |

## ★ The evidence for H6 is independent of us
`matthewblakeward/notebook1d22107bd4` — **different author, different dataset, same morning
(last run 09:23Z)** — is in `ERROR` with, verbatim from its log at t=540.1 s:
```
FileNotFoundError: Attached dataset kehhill/gemma3-llm-cli was not mounted;
  checked [PosixPath('/kaggle/input/datasets/kehhill/gemma3-llm-cli'),
           PosixPath('/kaggle/input/gemma3-llm-cli')]
```
**An attached input, declared in metadata, absent from `/kaggle/input` at runtime — the same failure
class as ours.** Public ARC kernels at 07:15Z and 08:33Z are COMPLETE; the two most recent, 09:12Z
and 09:23Z, are both ERROR (the 09:12Z one is an unrelated CUDA OOM). Our last *confirmed* good
competition mount is `arc3-q38-engine-eval` v3, started 08-17 10:41Z.
⇒ **Best estimate: the regression begins between 08-18 08:33Z and 09:12Z**, i.e. **before** both of
our pushes. We did not cause it and could not have avoided it by waiting the 1h41m we waited.

**Honest limits, stated:** theirs is a *dataset* mount and ours a *competition* mount — same
`/kaggle/input` subsystem, **not proven the same root cause**; external corroboration is **n=1**; and
public notebooks ERROR for many reasons. H5 is weakened, **not eliminated**, and the clean way to
separate H5 from H6 is a re-push once public kernels are observed mounting inputs again — if it then
runs, H6; if it dies again while others succeed, H5 and the fork dataset is the suspect.

## HARDENING SHIPPED TODAY (no slot, no treatment change)
The stock cell reports this failure through **pip**, as `Location ... is ignored: non-existing path`
— which reads like a packaging problem and cost real diagnosis time **twice**. Added a
**GRAFT-EVAL MOUNTCHECK** to our own cell 2 (already a modified cell, runs first at ~t=5 s, *before*
the stock cell): it prints the full `/kaggle/input` tree and raises a labelled
`GRAFT-EVAL INFRA DEATH: competition data mount ABSENT at ...` naming the prereg §4 rule inline.
It changes **no** flag and **no** treatment and cannot move a levels number — it only converts a
silent infra death into a labelled one, and makes the next post-mortem free.
Rebuilt and re-gated: **code sha `3c047dff2e6c02fd` → `79aa21fdbecbccf7`**, cells 17, differing cells
still **[2,6,12]**, **smoke 36/36**, **sealed scorer selftest 22/22**. *Note the consequence: the arm
is no longer byte-identical to v1/v2, so gate 1b will correctly classify tomorrow's push as new.*
Also fixed today, separately: **gate 1b (idempotence) had been BROKEN OPEN since inception** — it
compared an exact code sha, and Kaggle's cp1252 round-trip guarantees the remote never matches, so
the one guard protecting a scarce slot from a no-op re-push **had never been able to fire**. It now
normalises like step 3's VERIFIER FIX 1, and was verified to actually REFUSE before being overridden.

## STATE / NEXT
- **08-18 slots: 2 of 2 SPENT. Nothing further may be pushed today.**
- The graft-floor mechanism has **cost two slots and produced zero GPU-minutes of evidence.** It is
  still the reachable public floor and still worth one clean read.
- `graft_push.sh` remains **date-guarded to 2026-08-18**, so it is **fail-closed** for tomorrow by
  design: bumping `PUSH_DATE` to `2026-08-19` must be a deliberate, recorded act.
- **Pre-check before spending another slot (free):** confirm public ARC kernels are completing again
  (`kernels list --sort-by dateRun` + `kernels status` on the newest few). Do not re-push into an
  unmounted rail a third time.
