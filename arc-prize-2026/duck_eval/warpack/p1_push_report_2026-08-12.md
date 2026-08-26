# P1 suppressor — push report, 2026-08-12

Arm: **P1 zero-information action suppressor v1**, single flag on the duck
baseline (`P1_SUPPRESS=1`), (f) continuation default riding.
Prereg: `learnings/war_room/p1_prereg_2026-08-12.md` — **SEALED BEFORE THIS PUSH**.
Slot: **1 of 2** for 2026-08-12 (the animation eval spent 08-11's).
Authorised by the coordinator on the record: the suppression half is worth
**~+0.02 local, NOT +0.184**; the slot is spent for **mechanism C** (the
non-truncatable memory block), which is unmeasurable offline and targets the
day's root-cause finding (the agent re-explores because it has forgotten —
31,744-token context / 33 history messages on 225-action levels, while the
harness exposes a full `transitions` object it never queries).

---

## 0. Pre-push gates

| gate | requirement | result | verdict |
|---|---|---|---|
| smoke | all green | `p1_smoke.py` **76 passed, 0 failed** (10 structural, 15 unit, 21 replay, 22 real-offline-engine integration, 8 flag-gate/kill-switch subprocess) | OK |
| replay validation | ×1.09–1.11 reproduction with the exact scorer | reproduced (×1.1154 / ×1.1110 / ×1.0943) **and shown to delete 3/17 level-completing actions** → shipped defaults changed, see §5 | OK |
| module compiles | required | `py_compile` clean | OK |
| module is ASCII-only | required | 0 non-ASCII bytes | OK |
| stray `__pycache__` removed pre-push | required | removed | OK |
| other dataset modules untouched | required | 9/9 sha256 unchanged | OK |
| builder regression | byte-identical | `--animation` and `--compaction` rebuilt **byte-identical** | OK |
| notebook structural diff | arm-defining cells only | differs from `duckwar-eval` in **exactly cells 2, 12, 14**; `metadata.kaggle` **byte-identical**; nbformat 4.4 | OK |

## 1. Dataset version push — `canivel/arc-war-kit`

Pushed `duck_eval/warpack/_kaggle_dataset/` (10 `.py` files). Note:
`p1_suppressor_patch v1 (P1 zero-information action suppressor; safe defaults
memo_mode=noop confirm=2 abort_revisit=OFF; prereg p1_prereg_2026-08-12.md;
smoke 76/76)`.

*Operational note — the known CLI bug, re-confirmed.* `kaggle datasets version -p`
folds the path into the temp upload filename. A **forward-slash** absolute path
(`F:/kaggle/...`) fails exactly like a relative one
(`… Temp\.kaggle/uploads\F_/kaggle/…_animation_patch.py.json`). **Use a
backslash Windows absolute path** (`F:\kaggle\arc-prize-2026\duck_eval\warpack\_kaggle_dataset`).
The 08-11 runbook note ("use an absolute path") is hereby sharpened to
"use an absolute path **with backslashes**".

## 2. Dataset byte-audit (`feedback_kaggle_dataset_code_sync`)

Downloaded the new version back (`datasets download --unzip`) and re-hashed all
10 modules against staged: **10/10 MATCH.**

| module | sha256 (16) |
|---|---|
| **p1_suppressor_patch.py** (new, 27,969 B) | **`6b3addd587d6e378`** |
| animation_patch.py | `0c44e12121c55ec3` |
| budget_sentinel_patch.py | `6a28592c4a0ff637` |
| compaction_patch.py | `a7db3743470d6689` |
| continuation_patch.py | `0ecb33692e436e21` |
| fenced_recovery_patch.py | `90c575417130cf58` |
| ledger_core.py | `671e883ce0542262` |
| ledger_hook_cell.py | `1cfc0975862a4a3b` |
| ledger_patch.py | `73d1950efc7643f0` |
| warpack_patch.py | `17aa912b4e888cba` |

**DATASET BYTE-AUDIT VERDICT: PASS.**

## 3. Kernel push — `canivel/arc3-duck-p1-eval`

`kernels push` → **"Kernel version 1 successfully pushed."**
Status immediately after: **`KernelWorkerStatus.RUNNING`**.

## 4. Kernel pull-back verify

- **Code cells:** remote code-cell concat sha256
  `c7b59bc763e432b244c1352fba35e4079e269f6412e1c41b00d1b3b5e8e223fb`
  = local. **CODE MATCH: True.**
- Remote cell 2 carries `os.environ["P1_SUPPRESS"] = "1"` — **yes**.
- Remote cell 12 carries `import p1_suppressor_patch` — **yes**.
- Remote cell 14 carries `_p1.canary_report()` — **yes**.

## 5. `scripts/preflight.py` — BLOCK, and why it is not a finding

`preflight.py --kernel canivel/arc3-duck-p1-eval` returns **BLOCK** on
K2/K4/K5/K6/K8. **The same command returns the identical 5 failures on
`canivel/arc3-duck-animation-eval`** — the kernel that built COMPLETE and
produced our primary trace on 08-11.

Those checks test the **`arc3-baseline` agent-swarm notebook shape**
(`agents/__init__.py` imports, `.env` keys, `main.py --agent myagent`,
`%%writefile my_agent.py`). The duck-harness eval family is a different artifact
entirely (taaf + vLLM), so the checks are **structurally inapplicable**, not
failing. K1 (pulls cleanly) and K3 (nbformat 4.4) — the two that do apply —
both pass.

The applicable structural gate for this family is the **diff against the
war-eval baseline**, run pre-push: **exactly cells 2/12/14 differ,
`metadata.kaggle` byte-identical** (§0). `--mode trusted-fork` is also
inapplicable: the war-eval baseline itself differs from the Cottaar upstream in
7 code cells, so every member of this family is a graft, not a trusted fork.

*Runbook item (not fixed here, to keep this push a single diff):* `preflight.py`
should gain a duck-harness family profile, or the daemon should route this
family to the structural-diff gate. Filed for the next infra pass.

## 6. What ships (and what deliberately does not)

Shipped defaults, all verified in the offline replay of three recorded runs to
decline **zero** level-completing actions and **zero** board-changing actions:

```
P1_MEMO_MODE=noop      decline only pairs whose CONFIRMED outcome left the
                       board byte-identical -> board-equivalent to executing
P1_CONFIRM=2           CLAMPED IN CODE. At confirm=1 no pair is ever executed
                       twice, so the online latent-state detector can never
                       fire and the hard safety constraint would be void.
P1_MAX_DECLINES=1      a repeat request always executes -> no path can ever be
                       permanently blocked
P1_ABORT=1             batch abort on no-op
P1_ABORT_CYCLE=1       ...or on a loop closed INSIDE the same batch
P1_ABORT_REVISIT=0     OFF. Cuts the level-completing batch of tu93 L1,
                       sp80 L1 and ar25 L1 in the recorded traces.
P1_BLOCK=1             mechanism C, <=900 chars (measured 389)
```

`P1_MEMO_MODE=all` and `P1_ABORT_REVISIT=1` remain in the module as **ablation
handles only**. Turning either on for a scored run requires a new prereg that
addresses prereg §4 head-on.

## 7. Reading rules (from the sealed prereg)

- **M0 PRIMARY = mechanism delivery**, `saved/requested`, band **[3%, 30%]**
  (replayed 5.9% / 20.0% / 17.6%). The family
  `duck-harness-kaggle-continuation-v1` is still **m = 2 ⇒ NOT SCREENABLE**
  (`SCREEN_PROTOCOL` §1 P2), so per §4.6 the primary endpoint may not be Δlc.
- **M1 Δlc is DESCRIPTIVE ONLY** and **may not be reported as non-harm.**
- **M2 score**: sealed replay expectation ×1.040 / ×1.003 / ×1.015. **Not ×1.10.**
  Any post-hoc reading against ×1.10 is out of order.
- **No token-fraction canary** — mechanism C's cost is *input* tokens and the
  rail reports generated only. That mis-specified denominator is what killed the
  animation arm on a rule that fired while the measured token effect was
  negative. Cost is bounded statically instead (≤900 chars, smoke U13/I5c).
- **Build ERROR or missing banner ⇒ infra death, not a mechanism result.**

No submissions. No queue changes. No cloud spend.

---

# RESULT — build COMPLETE, scored 2026-08-12 → **NO-PROMOTE**

Pulled to `runs/kernel_pulls/p1_v1/` (~144 min build). Scored by
`duck_eval/warpack/p1_score.py` against the sealed prereg →
`runs/kernel_pulls/p1_v1/p1_score.json`.

## Infra: ALIVE (not an infra death)

| canary | verdict | evidence |
|---|---|---|
| K-P0 banner + arm + applied + no PATCH FAILED | **PASS** | `p1 v1: ACTIVE (4 seams patched)`, `applied=True`, graft from `/kaggle/input/datasets/canivel/arc-war-kit` |
| K-P1 ≥1 event on ≥5 games | **PASS** | 251 `P1 v=1` lines, 25 games (75 batch_abort, 9 decline, 142 latent_state, 25 game_end) |
| K-P2 banner states safe defaults | **PASS** | `mode=noop confirm=2`, `revisit is DEFAULT OFF` |
| K-P3 errors=0 | **PASS** | `errors=0`, 0 tracebacks |
| K-P5 detector fired LIVE | **PASS** | 6 games: cd82, g50t, ka59, m0r0, sc25, wa30 — no game id read |
| K-P6 dup below family | **DISPUTED** | 10.11% vs 12.65% all-actions (below) **but** 5.25% vs 2.70% cleared-levels (above) |

## M0 PRIMARY — 3.68%, PASS on the bottom edge

`saved/requested = 181/4,920 = 3.68%`, band [3%, 30%]. **Far under the replayed
5.9% / 20.0% / 17.6%.** Composition: **9 declines + 172 aborted** vs 4,739
executed. The decline arm barely fired: `confirm=2` — the floor that keeps the
latent-state detector alive — requires two consistent executions before any pair
may be declined, and the no-op batch-abort usually fires first. **Delivered, at
about half the low end of its own replay.**

## MECHANISM C — DELIVERED, BEHAVIOUR NOT DEMONSTRATED (the null)

**Delivery is unambiguous.** 1,463 of 1,519 turns carried the block = **96.3%
coverage on all 25 games**; mean **339** chars, max **599** (sealed bound 900);
1,463 blocks named untried primitives, 215 named confirmed-dead, 147 flagged
latent state.

**Behaviour is not.** Dead-reissue rate, arm: first half **176/519 = 33.9%** →
second half **46/592 = 7.8%** (ratio 0.229, z = 10.9). In isolation that reads
as a large win. The identical statistic on **block-free controls**
(reconstructing what the block *would* have said) says otherwise:

| run | first half | second half | ratio |
|---|---|---|---|
| **ARM p1_v1 (block delivered)** | **33.91%** | **7.77%** | **0.229** |
| CONTROL animation_v1 | 11.35% | **5.33%** | 0.470 |
| CONTROL a22_v2_seed1 | 16.39% | 13.90% | 0.848 |
| CONTROL a22_compaction_v1 | 15.32% | 23.05% | **1.505 (reverses)** |
| ARM minus m0r0 | 9.82% | **4.31%** | 0.439 |
| CONTROL animation_v1 minus m0r0 | 10.90% | **1.86%** | 0.171 |

**The arm's second-half rate sits inside the control spread (5.3–23.1%) and
above the best control.** `m0r0` alone supplies **190 of the arm's 222
re-issues (86%)** and is a latent-state game where suppression is off by design.
The within-run first→second fall happens without any block, and on one control
it runs the other way. **The block lands; the agent does not act on it.
Mechanism C = DEAD on this evidence.**

## M1 / M2 — descriptive only

- **M1:** arm **lc 17** over 25 games vs family {16, 10}. Family
  `duck-harness-kaggle-continuation-v1` is m = 2 ⇒ **NOT SCREENABLE**; this
  **may not be reported as non-harm**.
- **M2:** arm local-25 RHAE **1.9273**. **Not attributable to P1.** The sealed
  expectation was a ×1.019 multiplier *on the same run*, not computable without
  a paired baseline; this is one draw against an m=2 family with a disputed
  reading gate. Reading 1.9273 as a P1 win is exactly the error this build
  exists to prevent.
- Blind-tail 11.58%; actions/analysis-step 3.96.

## Kill rules — none fired; verdict is NO-PROMOTE on the reading gate

All five sealed kill rules pass (patch installed, `errors=0`, lc 17 > 15, no
flagged game below the family mean, M0 ≥ 3%). **Zero board-changing and zero
level-completing actions were declined or aborted**, by construction
(`mode=noop` + `abort_revisit=0` + `confirm≥2`) and replay-verified 0/0/0
pre-push. The arm is **NO-PROMOTE** because reading gate **K-P6 is disputed** —
per prereg §5, M1/M2 may not then be read as evidence of anything.

## Hygiene

**No submission fired. Queue untouched** — the frozen-fork filler is still the
only pending entry, and `runs/submission_log.jsonl` shows only the 00:07Z filler
submit and the 22:37Z `already-submitted-today` skip. **Push usage 2026-08-12:
1 of 2** (one dataset version + one kernel push). No cloud spend.
