# Animation-awareness eval seed-1 push report — 2026-08-11

**Executed:** 2026-08-11 (weekday build rail). **Sealed intent:**
`learnings/war_room/animation_prereg_2026-08-11.md` (commit `4b57bb0`, sealed
BEFORE `animation_patch.py` existed). **Budget:** kernel push **1 of today's
max 2** (0 used before this; `arc3-duck-repro` and `arc3-duck-compaction-eval`
both terminal-COMPLETE, nothing in flight). **NO competition submission;
submission queue untouched (frozen fork remains tonight's head). $0 cloud.**

**Fresh kernel slug** `canivel/arc3-duck-animation-eval` (v1), per
`feedback_fresh_kernel_slug`.

---

## 0. What this arm is

The sweep's ADOPT #1 — a **perception defect in the artifact we ship**:
`taaf/game.py:170` returns only `raw.frame[-1]`; `all_frames` /
`animation_frames` exist with **zero consumers** in `ARC3-Inference`. Our own
LM-free audit of all 25 offline engines (`runs/animation/frame_audit.md`,
11,104 actions) measures the cost: **17/25 games multi-frame, 401 actions
(3.6% of all, 19.0% of apparent no-ops) had a settled board byte-identical to
the pre-action board while an intermediate frame differed** — the state-aliasing
class. On `ft09`: 281/352 actions, **99.3% of everything that looked like a
no-op**.

**The no-op guard (ADOPT #2) is NOT in this flag and NOT in this module**
(prereg §2.2): it is strictly downstream, separately gated, and *harmful* on
type-1 games without this arm.

---

## Pre-push gate verification (all GREEN)

| gate | expected | observed | verdict |
|---|---|---|---|
| `animation_patch.py` sha256 | staged | `0c44e12121c55ec340e632596167081460b5eb2689ee25fca6aaa5daf4594831` (24,609 B) | OK |
| notebook sha256 | staged | `d232cbc40c179d4d8b9e493dc0f396719e09d81e8a9471cd1f4736d631f0f3e5` | OK |
| `animation_smoke.py` | 100% PASS required | **56/56 PASS** (structural + units + REAL offline ft09/tr87 integration + ToolAgent seams + canary + flag-gate/kill-switch subprocesses) | OK |
| builder regression | byte-identical | `default` / `--w0` / `--sentinel --sentinel-budget 150` / `--compaction` rebuilt **byte-identical**; `--a17-canary` **restored from HEAD** — pre-existing hand-edit drift (v5 dataset route `canivel/qwen25-vl-72b-awq`, per `a17_v5_dataset_route_2026-07-27.md`) that the builder does not reproduce; **not introduced and not fixed here** | OK |
| other dataset modules untouched | required | 8/8 sha256 unchanged (incl. `compaction_patch.py` `a7db3743…`) | OK |
| stray `__pycache__` removed pre-push | required | removed | OK |
| module is ASCII-only | required | 0 non-ASCII bytes | OK |

## 1. Dataset version push — `canivel/arc-war-kit`

Pushed `duck_eval/warpack/_kaggle_dataset/` (9 `.py` files). Notes:
`animation_patch v1 (animation-awareness, sweep 08-11 ADOPT #1; sha
0c44e121...594831; prereg animation_prereg_2026-08-11.md; smoke 56/56)`.

*Operational note for the runbook:* `kaggle datasets version -p` with a
**relative** path fails on this box (`[Errno 2] … Temp\.kaggle/uploads\duck_eval/warpack/_kaggle_dataset_animation_patch.py.json`
— the CLI folds the path into the temp upload filename). **Use an absolute
path.**

## 2. Dataset byte-audit (`feedback_kaggle_dataset_code_sync`)

Downloaded the new version back (`datasets download --unzip`) and re-hashed all
9 modules against staged: **9/9 MATCH**, `animation_patch.py`
`0c44e121…594831` = staged. **DATASET BYTE-AUDIT VERDICT: PASS.**

## 3. Kernel push — `canivel/arc3-duck-animation-eval`

`kernels push` → **"Kernel version 1 successfully pushed."** Status 45 s later:
**RUNNING**.

## 4. Kernel pull-back verify

- **Code cells:** code-cell concat sha256
  `a7670497f5a198bf4dc3f5b00c413198b55acfb8c9a1ee984b9d7689fde82fbb` —
  local == pulled, **MATCH**.
- **Metadata:** `enable_gpu=True`, `enable_tpu=False`, `enable_internet=False`,
  `is_private=True`, docker pinned `sha256:57e612b4…`, `machine_shape
  NvidiaRtxPro6000`, `competition_sources`, `model_sources=[]`,
  `dataset_sources` (4, incl. `canivel/arc-war-kit`) — **ALL OK, no drift**.

**KERNEL PULL-BACK VERDICT: PASS. Pushed version = 1.**

## 5. Runtime canary check (deferred to build COMPLETE, ~2.2 GPU-h)

Read from the v1 build log (prereg §3):

1. **K-A0** `animation v1: ACTIVE (4 seams patched)` banner + the cell-2
   `ANIMATION_AWARE=1` stamp + `animation-eval: SEED=1` line. Absent or
   `PATCH FAILED` ⇒ ran VANILLA ⇒ **VOID** (not a FAIL).
   Also expect the `(f)` banner `continuation v1: … ACTIVE (2 modules patched)`
   and **no** `warpack:` / `LEDGER` / `SENTINEL` / `COMPACTION ` lines (P1).
2. **K-A1** ≥1 `ANIMATION ` event line, on **≥5 distinct games**;
   `<game>_animation_events.jsonl` sidecars present.
3. **K-A2** the single `ANIMATION CANARY` line must show `invisible>0` and
   `audit_type1_engaged` naming ≥1 of `ft09/cd82/sc25/ls20`. Zero across all
   four ⇒ **VOID**, and the audit method goes back under review.
4. **K-A3** `token_fraction < 0.01` (local smoke measured 0.00243).
5. **K-A4** `errors=0`. Nonzero ⇒ arm KILLED regardless of Δlc.

Then the read-out: **M0 (primary, mechanism)** = invisible/executed actions,
overall and per game. **M1 (Δlc) is DESCRIPTIVE ONLY** — the legal comparator
family `duck-harness-kaggle-continuation-v1` has **m = 2** (`w0_eval_s1` lc 16,
`w0_cont_eval` lc 10), so under `duck_eval/SCREEN_PROTOCOL.md` §1 P2 this arm
is **NOT SCREENABLE**; the prereg invokes the §4.6 power-honesty clause, so
**no PASS may be reported as non-harm** — only "uninformative in both
directions". M3 (repeated-no-op rate on the four type-1 games) is measured here
to size the downstream no-op-guard arm at zero extra cost.

*No submission, no queue change, $0 cloud. Kernel-push budget: 1/2 used today.*
