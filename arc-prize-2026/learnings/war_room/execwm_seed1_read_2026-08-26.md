# EXEC-WM v1 — SEALED READ (seed 1), 2026-08-26

Arm: `canivel/arc3-execwm-eval` v1 · KAOS exp_id 48 · prereg `learnings/war_room/execwm_prereg_2026-08-25.md` (SEALED pre-push 08-25)
Artifact: `runs/kernel_pulls/execwm_v1/` (COMPLETE, pulled 08-25 20:32-20:34) · Board draw: **1.05** (submitted 08-26 00:35, COMPLETE)
Scorer: `duck_eval/execwm/execwm_score.py` (selftest 15/15, cross-arm A2 all-refused) → `"verdict": "CERTIFIED"`, `"reason": "certified"`

## 1. CERTIFICATION

**CERTIFIED.** Not INFRA DEATH. `armed_count 50` (25 games × 2 arming records), `source: "reports"` — the scorer read the job-dir report files, so the P1 0-byte-log failure class did not recur. The 308,634-byte kernel log is present and parseable.

## 2. PRIMARY + CO-PRIMARY (bands sealed pre-push)

| stat | value | band |
|---|---|---|
| **`lc_total`** (primary) | **25** | HARM ≤23 · **NULL 24–34** · SIGNAL ≥35 → **NULL** |
| `trim1` (co-primary) | 2.330 | — |
| `mean_score` | 3.006 | (retired as primary: 50.4% one game) |
| `total_actions` | 2,972 | |
| `won` | 0 | |

Comparator re-derived at read time per Rule 0: local-rail lc series = field floor **28** + Arm A base **30** ⇒ mean **29.0**, pooled seed sd **2.80**. lc 25 sits **−1.43σ** below the comparator — inside the NULL band, indistinguishable from the config it wraps. Board-side: 1.05 vs field-floor config n=5 mean 1.5760 sd 0.2713. One draw; licenses nothing on its own.

**Decisive kill NOT triggered** (needs D2-delivered ≥5 games AND `levels_cleared_by_plan = 0` AND lc ≤23; actual: 5 games, 1 plan-clear, lc 25). The v1 exec-WM class is **NOT dead by its own pre-stated rule.**

## 3. DELIVERY GATES

| gate | bar | actual | result |
|---|---|---|---|
| **D1** reports | ≥20/25 games | **18/25** | **FAIL (as written)** |
| **D2** games reaching PHASE P | ≥3 | **5** (dc22, ka59, ls20, sk48, sp80) | **PASS** |
| **D3** mechanism-refute | ≥10 planned levels ∧ 0 plan-clears | 6 planned, 1 plan-clear | not triggered |
| **D4** graceful degradation | if 100% fallback then lc ∈ 29.0±5.6 | fallback 96.875% (not 100%); lc 25 ∈ [23.4, 34.6] anyway | **PASS** |

Because the "else DELIVERY-LIMITED" clause attaches to **D2** in the sealed text, and D2 passed, a mechanism read is licensed.

### 3a. D1's failure is a GATE-SPECIFICATION DEFECT, and it is recorded as FAILED anyway

Root cause, from the log: **the arm ARMED on 25/25 games** (`grep -c "[execwm] armed v1 game=" = 25`) and **exactly 7 self-disabled** with `[execwm] disabled reason=no-keyboard-actions valid=['ACTION6']`. The 7 games with no report — `ft09, lp85, r11l, s5i5, su15, tn36, vc33` — are **click-only games**, out of the v1 movement rule class by construction. The arm refused them correctly; it just doesn't emit a report when disabled.

So D1 counted **the arm's correct refusal as a delivery failure.** It cannot distinguish "never reached the game" from "reached it and correctly declined." Honest delivery is **25/25 armed, 18/18 in-scope games reported.**

**The gate is NOT rewritten to convert its own FAIL into a PASS.** D1 stands FAILED as sealed. What is recorded is the defect, for the next prereg: *a delivery gate must count arm-reachability, not report-presence, whenever the arm has a legitimate self-disable path.* Same family as the 08-20 arm-mismatch lesson (a scorer defined on a marker's PRESENCE refuses the arm defined by its ABSENCE).

**Companion scorer defect:** `execwm_score.py` reports `"disabled_games": 0` because it counts `disabled_reason` only across the 18 **present** reports. The true count is 7, and it is only recoverable from the log. The scorer undercounts the exact quantity D1 depends on.

## 4. ★★★ THE RESULT THAT SURVIVES: ls20 level 1 CLEARED BY PLAN ON THE REAL RAIL, ZERO LLM TOKENS

`ls20-9607627b`: `llm_calls: 0`, `llm_tokens: 0`, 741 actions, level 1 `phase: "P"`, `cleared_via: "plan"`. Four move rules mined and verified at **precision 1.0** (n = 33/21/23/19, deltas ±5 rows/cols) — byte-for-byte the same model the CPU smoke mined offline. **The CPU proof transferred to Kaggle exactly.** This is the campaign's first level cleared by deterministic search with no LLM in the loop, on the scored rail.

It did not stop there of its own accord: level 2 mined the same 4 rules at precision 1.0 on far more evidence (n = 279/173/17/121) and fell back only on `plan-budget-exhausted` after **96 plans run**.

## 5. ★★★ THE BINDING DEFECT IS UPSTREAM OF THE PLANNER — THE RULE CLASS IS NARROW, NOT THE BUDGET

Fallback taxonomy over 32 in-scope levels (`fallback_rate` 96.875%):

| stratum | levels | evidence |
|---|---|---|
| **no candidate rules at all** (`nrules = 0`) | **14** | ar25×2, cd82×3, cn04×2, lf52×2, re86×2, sb26×2, tr87×1 |
| **candidates mined, ALL refused by the verifier** (`nrules > 0`, `verified = 0`) | **9** | tu93×3, wa30×2, sc25, re86, bp35, m0r0-partial |
| verified rules, planned, did not clear | 5 | dc22 (targets-exhausted), ka59 (sprite-lost), sk48 + sp80 (prediction-breaks), ls20 L2 (plan-budget) |
| **verified, planned, CLEARED** | **1** | ls20 L1 |

**The probe budget was fully spent on the failures.** Probe histogram across 32 levels: `{4: 2, 8: 2, 16: 11, 18: 1, 20: 16}`. The 14 zero-rule levels got the **full 20 (or 16) probes and produced zero candidates** — this is not an early-exit and raising the probe budget does not address it. The failure is **object identification and rule-class coverage**, upstream of both the verifier and the planner.

Read plainly: **v1 assumes a keyboard-controlled sprite translating by a constant delta.** That assumption holds on ls20 (and partially on dc22/g50t/ka59/m0r0). It is false or unidentifiable on the majority of this benchmark. 7/25 games are click-only; of the 18 in-class games, only 5 reach PHASE P. The CPU smoke already flagged bp35/tu93 as genuine non-movers and they reproduced as non-movers here — the smoke was honest; the class is just narrow.

The verifier behaved exactly as designed: **24 rules verified, 29 rejected.** On 9 levels it refused every candidate. A component that refuses two-thirds of what it is offered on hard levels is the one part of this arm that is demonstrably not fooling itself (cf. S8: it refuses deliberately-wrong programs).

**Two early-exits worth root-causing before any v2:** `sb26` got **4 probes** and `bp35` got **8**, against a 16–20 budget everywhere else. sb26 carries **50.4% of the certified field floor's entire `mean_score`** — the single most valuable game on the rail received the smallest exploration budget in the run. That is a bug, not a policy.

## 6. VERDICT

**NULL on the primary, mechanism NOT refuted, delivery gate D1 failed as written on a specification defect, and one genuine positive result banked (ls20 L1 by plan, zero LLM, transferred from CPU).**

The arm is not a failure and it is not a signal. It is a **narrow instrument that worked where its assumptions held and correctly declined where they did not** — and that decline covers ~72% of scored levels, which is why the board draw (1.05) sits below the floor it wraps rather than above it.

**Pre-registered consequence for v2, stated now, before any build:** a v2 may NOT be a parameter tweak of the probe budget — the data above closes that door. Any v2 must widen the **rule class / object model** (click-addressable objects; non-constant deltas; multi-object dynamics) or it is not responsive to this read. The sb26/bp35 probe-truncation bug is a separate, unconditional fix.

---

## 7. BLOCKING GATE DISCHARGED — the winframe defect did NOT cause this draw

**The block (KAOS exp 49, filed the same morning).** `GameState.frame` returns
`raw.frame[-1]`. On an ordinary step `raw.frame` holds one layer; on the step that
**completes a level** it holds two or more, where layer 0 is the terminal board of the
level just cleared and `layer[-1]` is the **opening board of the NEXT level**. Our
exec-WM settled-frame reader takes `[-1]` like everything else in the duck/taaf lineage,
so at a level clear PHASE P's predict-act-compare check would mispredict **by
construction**, be charged a BREAK, and — at `MAX_BREAKS_PER_LEVEL = 3` — latch fallback.
The claim was published in `thtennant/arc3-duck-v25` and **independently replicated on
our own archive across 6 games**. Correctly, exp 49 ruled that the pulled artifact must
be checked before the prereg's decisive-kill clause is allowed to fire.

**The check, run on `runs/kernel_pulls/execwm_v1/`.** The defect can only fire at a
level-**completing** step. So: every break this run recorded, against the engine's own
`levels_completed` for that game.

| game | level | breaks | engine `levels_completed` | fallback reason |
|---|---|---|---|---|
| `ka59` | 1 | 3 | **0** | sprite-lost |
| `sk48` | 1 | 1 | **0** | prediction-breaks |
| `sp80` | 1 | 1 | **0** | prediction-breaks |

**All 5 breaks in the entire run occurred on games that never cleared a single level.**
There was no level-completing step for the defect to fire on. And the one level that
*did* clear by plan — `ls20` L1 — recorded **`breaks: 0`**, so it was not corrupted
either.

**VERDICT: the winframe defect is REAL and is NOT implicated in the exec-WM v1 draw.**
It explains 0 of 5 breaks and 0 of 31 fallbacks. The block is discharged on the artifact
rather than on argument.

**What that does and does not license.**
- It leaves §5 fully intact. 23 of 32 levels failed **upstream of any prediction check**
  — 14 with zero candidate rules after a full probe budget, 9 with every candidate
  refused by the verifier. Not one of those failures involves a level transition. The
  rule class is narrow; that finding does not depend on the frame reader.
- It does **not** retire the defect. Prospectively it is a live hazard for any arm that
  actually clears levels and reads frames across the transition — which is precisely
  what a working exec-WM v2 would do. ls20 L1 escaped it, plausibly because the plan
  terminated at the clear rather than predicting through it. **Fix the reader in v2
  regardless**; the reason to fix it is the next run, not this one.
- The decisive-kill clause never fired anyway (it required `levels_cleared_by_plan = 0`
  **and** lc ≤ 23; actual: 1 and 25), so no rule class was at risk of being killed by a
  mis-instrumented run. The gate did its job before it was needed.

**The generalisable lesson (exp 49's, and it is the right one):** an audit is only as
wide as the transitions it exercised. The 08-24 brief cleared this same accessor —
"our lineage already reads `raw.frame[-1]`, audited, no defect" — on a test that
exercised animation strips **mid-level** and never exercised a level **transition**,
which is the one place where reading `[-1]` *is* the defect. Record which transitions an
audit actually covered, and treat the uncovered ones as **unaudited**, not as clean.
