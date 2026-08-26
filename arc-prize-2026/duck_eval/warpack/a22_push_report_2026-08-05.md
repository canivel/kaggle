# A22 compaction-eval v2 push report — 2026-08-05

**Executed:** 2026-08-05 (weekday build-rail; sealed intent
`learnings/war_room/a22_compaction_v2_prereg_2026-08-04.md`; verify gate
`duck_eval/warpack/a22_v2_verify_report_2026-08-05.md` → VERDICT: GO).
**Budget:** kernel push 1 of today's max-2 (exactly ONE performed). Dataset
version does NOT count vs kernel budget. NO competition submission; queue
unchanged (frozen filler armed separately at 12:38). $0 cloud.

---

## Pre-push gate verification (all GREEN)

| gate | expected | observed | verdict |
|---|---|---|---|
| compaction_patch.py sha256 (v2, post verify-agent edits) | `5d8579ad…e1804f` | `5d8579ad0960312629c4804a27e99a905e6ffec601673b81a6b26e13ace1804f` | MATCH |
| notebook sha256 (byte-identical to pushed v1 — v2 swap is dataset-side only) | `1c4e51eb…50edb` | `1c4e51ebad3e8a371d3b8aad525930a60590c357b6f183f51be7b5e32e750edb` | MATCH |
| compaction_smoke_v2.py | 100% PASS required | **142/142 PASS** (verify agent, `uv run`) | OK |
| builder regression (default / --sentinel 150 / --w0 / --compaction) | byte-identical | byte-identical; `--a17-canary` ruled N/A (dedicated a17 lane, base-builder output never committed) | OK |
| ledger_core.py untouched | required | byte-identical to canonical twin, mtime 07-16 | OK |
| stray `__pycache__` removed pre-push | required | removed | OK |

## 1. Dataset version push — `canivel/arc-war-kit`

- Pushed `duck_eval/warpack/_kaggle_dataset/` as-is (absolute `-p` via
  PowerShell per the kaggle-2.0.0 Windows path bug). 8 `.py` files uploaded.
- Version notes: `A22 compaction_patch v2 region-aware eviction (sha
  5d8579ad...e1804f; prereg 2026-08-04; smoke 142/142)`.
- Server shows lastUpdated **2026-08-05 12:45:34** (version live).

## 2. Dataset byte-audit (feedback_kaggle_dataset_code_sync)

Downloaded the new version back (`datasets download --unzip`) and compared:
downloaded `compaction_patch.py` sha256 = `5d8579ad…e1804f`, `cmp` staged vs
downloaded → IDENTICAL. **DATASET BYTE-AUDIT VERDICT: PASS.**

## 3. Kernel push — `canivel/arc3-duck-compaction-eval`

- `kernels push` → **"Kernel version 2 successfully pushed."** Status ~1 min
  later: **RUNNING**. This is the v2 seed-1 eval (same notebook bytes as v1;
  the mechanism swap arrives via the arc-war-kit dataset version).

## 4. Kernel pull-back verify

- **Code cells:** 8/8 code-cell concat sha256
  `3fc85449e60de40da8bf5b335bddca8fd7f2e88077c3de0f929f047006246519` —
  local == pulled, MATCH.
- **Metadata:** enable_gpu/tpu/internet, is_private, docker pinning,
  dataset_sources (incl. arc-war-kit), competition_sources, model_sources —
  ALL OK, no drift.

**KERNEL PULL-BACK VERDICT: PASS. Pushed version = 2.**

## 5. Runtime banner check (deferred to build COMPLETE, ~2.2 GPU-h)

Prereg §3 canary, to read from the v2 build log:
1. `compaction v2: ACTIVE` banner — a **v1** banner ⇒ stale dataset served ⇒
   run VOID; absent/`PATCH FAILED` ⇒ VANILLA, VOID (K2).
2. `COMPACTION=1` stamp (cell-2 banner).
3. ≥1 `COMPACTION ` event line + per-game `*_compaction_events.jsonl` (K1).
4. **RETAIN-OFF canary (inverted vs v1): `retained_reasoning_msgs=0` on every
   event** AND banner shows mirroring OFF; any nonzero ⇒ sub-arm leak ⇒ VOID.

Then the seed-1 screen: paired M1 vs `runs/kernel_pulls/war_eval_v1/` seed 1
(mean Δlc ≥ −0.128 AND worst-game ≥ −1.0); M2 budget-relief attribution
split; M3 refuted re-proposal rate. K3: FAIL ⇒ v2 PAUSED (lane one FAIL from
DEAD given the v1 record).

*No submission, no queue change, $0 cloud. Kernel-push budget: 1/2 used today.*
