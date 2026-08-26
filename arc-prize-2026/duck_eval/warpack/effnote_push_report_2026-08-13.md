# EFFNOTE — push report, 2026-08-13

Arm: **EFFNOTE quantified per-turn efficiency note v1** — single flag on the
duck baseline (`EFFNOTE=1`), (f) continuation default riding, **REPORT-ONLY**.
Spec: `learnings/war_room/harness_diff_2026-08-13.md` §4 item **#1**.
Prereg: `learnings/war_room/effnote_prereg_2026-08-13.md` — **SEALED BEFORE
THIS PUSH**, with the **control spread computed first** (§1) so the arm cannot
be read against its own within-run contrast.
Slot: **1 of 2** for 2026-08-13 (no other kernel push today; the 00:07Z
frozen-fork submission is the queue daemon, not a build slot).

---

## 0. Pre-push gates

| gate | requirement | result | verdict |
|---|---|---|---|
| smoke | all green | `effnote_smoke.py` **99 passed / 0 failed** (13 structural, 9 legality/report-only, 30 unit, 10 control-spread replay, 29 real-offline-engine integration, 8 flag-gate/kill-switch subprocess) | OK |
| control spread | computed BEFORE any arm exists | `effnote_replay.py` over 3 block-free runs → `runs/effnote_replay/control_spread.json` + target-100 sensitivity | OK |
| replay fidelity | reproduce the recorded action totals | 5151 / 3492 / 4777 — **exact on all three** | OK |
| module compiles | required | `py_compile` clean | OK |
| module is ASCII-only | required | 31,618 B, **0 non-ASCII bytes** | OK |
| stray `__pycache__` removed pre-push | required | removed | OK |
| other dataset modules untouched | required | **10/10 sha256 unchanged** | OK |
| builder regression | byte-identical + deterministic | `--animation/--compaction/--p1/--w0/--sentinel` rebuilt, hashes stable, cell-diff sets unchanged | OK |
| notebook structural diff | arm-defining cells only | differs from `duckwar-eval` in **exactly cells 2, 12, 14**; metadata delta = `{id,title,code_file}` | OK |
| preflight | **N/A — known runbook debt** | `scripts/preflight.py` has no duck-harness family profile; its K2/K4/K5/K6/K8 test the `arc3-baseline` agent-swarm shape and BLOCK every member of this family (incl. `arc3-duck-animation-eval`, which built COMPLETE). `--mode trusted-fork` is inapplicable (the war-eval baseline itself differs from the Cottaar upstream in 7 code cells). **The structural diff above is the applicable gate.** | documented |

## 1. Dataset version push — `canivel/arc-war-kit`

Pushed `duck_eval/warpack/_kaggle_dataset/` (**11** `.py` files; `effnote_patch.py`
is new). Note: `effnote_patch v1 (EFFNOTE quantified per-turn efficiency note;
REPORT-ONLY, 2 seams, proxy-only target, 700-CHARACTER bound; prereg
effnote_prereg_2026-08-13.md; smoke 99/99)`.

*CLI bug, re-confirmed a third time:* `kaggle datasets version -p` needs a
**Windows absolute path with BACKSLASHES**
(`F:\kaggle\arc-prize-2026\duck_eval\warpack\_kaggle_dataset`). Also note the
CLI writes a cp1252-undecodable byte to stderr on success — capture stderr with
`errors="replace"` or the wrapper raises on a run that actually worked.

## 2. Dataset byte-audit (`feedback_kaggle_dataset_code_sync`)

Downloaded the new version back (`datasets download --unzip`) and re-hashed all
11 modules against staged: **11/11 MATCH.**

| module | bytes | sha256 (16) |
|---|---:|---|
| **effnote_patch.py** (new) | **31,618** | **`25df416f28d42c3b`** |
| animation_patch.py | 24,609 | `0c44e12121c55ec3` |
| budget_sentinel_patch.py | 17,296 | `6a28592c4a0ff637` |
| compaction_patch.py | 41,739 | `a7db3743470d6689` |
| continuation_patch.py | 6,073 | `0ecb33692e436e21` |
| fenced_recovery_patch.py | 6,754 | `90c575417130cf58` |
| ledger_core.py | 22,485 | `671e883ce0542262` |
| ledger_hook_cell.py | 2,456 | `1cfc0975862a4a3b` |
| ledger_patch.py | 13,443 | `73d1950efc7643f0` |
| p1_suppressor_patch.py | 27,969 | `6b3addd587d6e378` |
| warpack_patch.py | 18,750 | `17aa912b4e888cba` |

## 3. Kernel push — `canivel/arc3-duck-effnote-eval`

Version **1** pushed → status **RUNNING**.

Pull-back verification:

* code-cell concat **sha256 `f359ae5e8400df1b` = local, MATCH**
* remote cell 2 carries `os.environ["EFFNOTE"] = "1"`
* remote cell 12 imports `effnote_patch` **and** `continuation_patch` ((f) default)
* remote cell 14 calls `_effnote.canary_report()`
* datasets = wheelhouse + taaf bundle + 27B snapshot + `canivel/arc-war-kit`;
  docker sha `…e16132a8be4cb13c`, `NvidiaRtxPro6000`, GPU on, internet off —
  identical to the family (`feedback_kaggle_env_match`)

## 4. How to read the result

```
# infra alive (K-E0 / K-E4 / K-E5)
grep -n "effnote v1: ACTIVE"                  <log>
grep -n "effnote v1: graft applied"           <log>
grep -c "PATCH FAILED"                        <log>     # must be 0
grep -n "continuation v1"                     <log>     # (f) default rode

# the one summary line (K-E0b / K-E1 / K-E1' / K-E3)
grep -n "EFFNOTE CANARY"                      <log>
#   -> note_rate= chars_mean= chars_max= bound=700 over_rate=
#      stall_rate= nz=N/Ng stag=N/Ng rev=N/Ng errors= target=proxy-only

# per-turn events (delivery on stall/over-target turns, one line each)
grep -c "EFFNOTE v=1 kind=note"               <log>
grep -c "EFFNOTE v=1 kind=game_end"           <log>     # expect 25

# non-harm (K-E2): levels_completed >= 14 (control minimum)
grep -n "^\[finished\]"                       <log>
```

Then score the **behavioural** endpoint by re-deriving the note offline from the
arm's own `artifacts/*_p0_events.jsonl` with the **same** reconstructor used on
the controls:

```
.venv/Scripts/python.exe duck_eval/warpack/effnote_replay.py        # controls
# then add the pulled arm dir to RUNS and re-run; compare B1 against
# runs/effnote_replay/control_spread.json  ->  control_spread.B1_...['min']
```

**PASS requires `B1_post_stall_revisit_rate` < 0.3986** — strictly below the
*minimum* of the control spread (0.3986 – 0.5487). The arm's own
first-half/second-half contrast **may not be cited**: that is exactly the
statistic that made P1's mechanism C look like a 4.4× win when it was
regression to the mean.

## 5. What was deliberately NOT built

* **No real-baseline read of any kind** (no `base_actions_per_level`, no
  `metadata.json` rglob) — the reference prefers them; we deleted the path so
  the eval and the hidden set see the identical mechanism. Asserted by smoke
  L1/L2.
* **No per-game baseline table** — game-specific and, per the 08-12 P1 finding,
  factually wrong on a rerun.
* **No token-fraction canary** — the bound is 700 CHARACTERS. Smoke L6/I6b
  assert that no token metric exists in the module or the canary at all.
* **caoyupeng's `external_game_id=f"{env}-dup"` replay gate — not ported in any
  form.** Smoke L4.
* **No action-path change.** `_execute_action` and `step_env` are the same
  objects after the graft as before (smoke I2g).

---

**No submission fired. Queue untouched. No cloud spend.**

---

# 6. RESULT — v1 COMPLETE, read 2026-08-13 under the sealed prereg

Output pulled to `runs/kernel_pulls/effnote_v1/`. Scored by
`duck_eval/warpack/effnote_replay.py --arm` → `runs/effnote_replay/arm_vs_control.json`.

## 6.1 INFRA — ALIVE (this is a mechanism result, not an infra death)

| canary | requirement | observed | verdict |
|---|---|---|---|
| **K-E0** | `effnote v1: ACTIVE`, 2 seams, report-only banner | present, `(2 seams patched)`, `target=proxy-only`, `cost bound = 700 CHARACTERS` | **PASS** |
| **K-E4** | `errors=0`, no `PATCH FAILED`, no traceback | `errors=0`, `PATCH FAILED` count **0** | **PASS** |
| **K-E5** | graft alone + (f) default rode | `effnote v1: graft applied` ×1, `continuation v1` ×2, no warpack/ledger/sentinel/compaction/animation/p1 | **PASS** |
| events | `kind=game_end` on 25 games | **25/25**, plus 380 `kind=note` lines | **PASS** |

## 6.2 CANARIES — all PASS

```
EFFNOTE CANARY v=1 version=v1 games=25 turns=1444 noted=1355 note_rate=0.9384
  chars_mean=284.0 chars_max=603 bound=700 over_target=332 over_rate=0.2299
  stall_turns=126 stall_rate=0.0873 nz=112/17g stag=5/1g rev=40/6g errors=0
  target=proxy-only
```

| canary | sealed rule | observed | control | verdict |
|---|---|---|---|---|
| **K-E0b** delivery | ≥0.80 on stall-or-over-target turns | `note_rate=0.9384` live; every stall/over-target turn noted (checked in the replay: noted 1036 ≥ union of 92 stall + 268 over) | .961–.967 | **PASS** |
| **K-E3** cost | `chars_max` ≤ 700, no token metric | **603**, mean 284.0 | max 602–603 | **PASS** |
| **K-E1** detectors | nz ≥3 games, rev ≥3 games, **stag ≥1** (re-registered §1.1) | **nz 17g · rev 6g · stag 1g** | nz 14–20 · rev 3–8 · stag 1–2 | **PASS** |
| **K-E1′** nagging | no detector on >40% of turns | `stall_rate=0.0873` | .071–.100 | **PASS** |
| **K-E2** non-harm | `levels_completed` ≥ 14 | **16** | 14 / 17 / 17 | **PASS** |
| over-target rate | descriptive | 0.2299 | .197–.292 | inside |

*Definitional note:* the live canary counts 1444 turns (every
`_build_user_prompt` call, including turns that issued zero actions); the
replay counts 1077 action-carrying turns. The replay definition is applied
identically to arm and controls, so no comparison is affected.

## 6.3 THE SEALED GATE — **FAIL**

**B1 post-stall revisit rate — of the actions issued on a turn whose note fired
a stall detector, the fraction landing on a board state already visited on that
level.**

```
SEALED GATE  B1_post_stall_revisit_rate  arm=0.4971
             threshold = < 0.3986 (control-spread MINIMUM)   ->  FAIL
```

| | animation_v1 | a22_v2_seed1 | a22_compaction_v1 | **ARM** |
|---|---:|---:|---:|---:|
| **B1 post-stall revisit** | 0.3986 | 0.5487 | 0.4751 | **0.4971** |

The arm sits **inside** the sealed control spread (0.3986 – 0.5487), above two
of the three controls. **Kill rule 2 fires ⇒ NO-PROMOTE.**

**Robustness (the failure is not a single-game artefact).** Excluding the
largest contributor (`re86`, 321 of 1022 stall actions) B1 **rises to 0.5578**.
Per-game B1 across the 9 games with ≥20 stall actions: median **0.623** —
ft09 0.981, m0r0 0.705, bp35 0.688, cn04 0.646, wa30 0.623, ar25 0.585,
sp80 0.444, re86 0.364, r11l 0.156. The arm fails on the aggregate and gets
worse under every reasonable trim.

**The arm's own first-half/second-half contrast is NOT cited and was not
computed as an endpoint** — that illusion is what made P1's mechanism C look
like a 4.4× win.

## 6.4 SUPPORTING READS

| metric | ARM | control spread | read |
|---|---:|---|---|
| **B1c** non-stall revisit | 0.2030 | .111 – .199 | **the detectors still work.** Post-stall 0.4971 vs non-stall 0.2030 = **2.4×**, in line with the controls' 2.5–3.6×. The note fires at genuinely wasteful moments. The agent does not act on it. |
| **B2** post-stall no-op | 0.2632 | .078 – .345 | inside |
| **B3** over-target burn, cleared levels | 195 | 32 – 307 | inside |
| **B3** over-target burn, all levels | 2470 | 1267 – 2301 | **above the control max** |
| **B4** actions per **stall** turn | **11.11** | 3.95 – 7.28 | **above the control max — the sharpest datum.** Told "commit to the shortest sequence that tests it", the agent issued ~11 actions on the turns where the note fired (8.15 excluding re86; controls 3.9–7.3). |
| **B4c** actions per non-stall turn | 3.92 | 3.46 – 3.94 | inside — so B4 is not a global batching shift, it is specific to noted turns |
| **M0** median actions/cleared level | 23.5 | 24 – 49 | marginally below the range; see §6.5 |

## 6.5 M1 / M2 — DESCRIPTIVE ONLY, NOT ATTRIBUTABLE

* **M1** `levels_completed` = **16** vs controls {17, 14, 17}. Family is
  **m = 2 ⇒ NOT screenable.** Used only for K-E2 non-harm (PASS).
* **M2** local-25 RHAE = **2.8779** vs controls 1.6352 / 1.4075 / 1.4509 and
  p1_v1 1.9273. **This may NOT be attributed to EFFNOTE.** It is one draw
  against an m = 2 family at sd 0.1513, and **60% of it is a single game**:
  `ft09` alone contributes **43.52 of the 71.95** total, having cleared 4
  levels (L2 in 7 actions vs a baseline of 12 ⇒ the 115 cap). The same `ft09`
  scored **0.000** on two of the three controls and 14.286 on the third. Reading
  2.8779 as an EFFNOTE win would be precisely the error the prereg §3/§6 was
  written to prevent, and the sealed behavioural gate — the endpoint that
  *can* discriminate — **failed**.
* The same caveat applies to M0's 23.5: it is a median over a *different set*
  of cleared levels (16 vs 17/14/17) and animation_v1's 24 is within noise of it.

## 6.6 VERDICT

> ## **NO-PROMOTE.**
> **The note is delivered on 94% of turns, at 284 chars, with the detectors
> firing at genuinely wasteful moments (2.4× the non-stall revisit rate) — and
> the agent's post-stall revisit rate, 0.4971, sits inside the block-free
> control spread. Being shown the scoring rule, its own action count and a
> firing stall detector does not change what it does next.**

Kill rule 2 fired as written. Nothing promoted, no submission, queue untouched.

## 6.7 What this closes

This was the **last convergent lever** in the harness diff: three independent
lineages at ≥1.40 (AGI Boys' prompt line, Helmut's hard `NoopGuard`, Tara Labs'
NET-ZERO/REVISIT lines) all encode the same belief — *the agent wastes actions
re-entering states*. We have now measured that belief **twice**, through two
different mechanisms, and got the same answer:

| arm | mechanism | delivery | behavioural result |
|---|---|---|---|
| P1 mechanism C (08-12) | hand the agent the ground truth it already has | 96.3% of turns | dead-reissue **inside** the control spread |
| **EFFNOTE (08-13)** | hand it the scoring rule + its own action count + a live stall alarm | **93.8%** of turns | post-stall revisit **inside** the control spread |

**The agent does not act on runner-supplied context, whether that context is
what it already tried or what its waste is costing it.** Combined with
RedundancyBench's 24.88% (the best LLM redundancy detector, some below random),
the runner-side push lane is now closed on measured evidence rather than on
argument. The remaining efficiency ideas that are *not* refuted by this pair are
the ones that do not require the model to act on advice: P2 (verified-plan
gating of batch size — note B4 shows the agent batching **11 actions** on stall
turns, which P2 would hard-cap at 1) and P3's cheap primitive-sweep variant.

**Push usage 2026-08-13: 1 of 2** (this arm's dataset version + kernel push).
Slot 2 unspent. **No submissions, no queue changes, no cloud spend.**
