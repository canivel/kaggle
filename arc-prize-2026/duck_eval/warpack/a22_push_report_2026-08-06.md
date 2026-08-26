# A22 compaction-eval v2.1 push report — 2026-08-06

**Executed:** 2026-08-06 (weekday build-rail; sealed intent
`learnings/war_room/a22_compaction_v2_1_prereg_2026-08-06.md`; build agent
report green: smoke 165/165, builder regression byte-identical, runtime
graft test PASS). **Budget:** kernel push 1 of today's max-2. NO competition
submission; queue unchanged (frozen filler is tonight's head). $0 cloud.

**Lane stakes at push:** v1 K3 FAIL + v2 K3 FAIL ⇒ a v2.1 seed-1 K3 FAIL
makes the A22 lane DEAD (sealed in the intent §4, incl. post-death
disposition).

---

## Pre-push gate verification (all GREEN)

| gate | expected | observed | verdict |
|---|---|---|---|
| compaction_patch.py sha256 (v2.1) | `a7db3743…7fa8ab` (build agent) | `a7db3743470d668905a43a9cfd3bbf9697158f299d304ecd6195c9a74d7fa8ab` (independent re-hash) | MATCH |
| notebook sha256 (byte-identical to pushed kernel — v2.1 swap is dataset-side only) | `1c4e51eb…50edb` | `1c4e51ebad3e8a371d3b8aad525930a60590c357b6f183f51be7b5e32e750edb` | MATCH |
| compaction_smoke_v2.py | 100% PASS required | **165/165 PASS** (incl. new digest-OFF section D + COMPACTION_DIGEST=1 restore regression) | OK |
| builder regression (default / --sentinel 150 / --w0) | byte-identical | byte-identical; `--a17-canary` N/A per standing 08-05 ruling (base-builder output hashed to the 08-05 reference, artifacts restored from git) | OK |
| ledger_core.py untouched | required | byte-identical to canonical twin | OK |
| stray `__pycache__` removed pre-push | required | removed | OK |

## 1. Dataset version push — `canivel/arc-war-kit`

- Pushed `duck_eval/warpack/_kaggle_dataset/` (8 `.py` files uploaded).
- Version notes: `A22 compaction_patch v2.1 pure eviction digest-OFF (sha
  a7db3743...7fa8ab; prereg 2026-08-06; smoke 165/165)`.

## 2. Dataset byte-audit (feedback_kaggle_dataset_code_sync)

Downloaded the new version back (`datasets download --unzip`): downloaded
`compaction_patch.py` sha256 = `a7db3743…7fa8ab` = staged.
**DATASET BYTE-AUDIT VERDICT: PASS.**

## 3. Kernel push — `canivel/arc3-duck-compaction-eval`

- `kernels push` → **"Kernel version 3 successfully pushed."** Status 60s
  later: **RUNNING**. This is the v2.1 seed-1 eval (same notebook bytes;
  mechanism arrives via the arc-war-kit dataset version).

## 4. Kernel pull-back verify

- **Code cells:** code-cell concat sha256
  `fc94726d378f32d333ff63dd6497b66b725e893f73fa27e7db3dc1dfbeebe579` —
  local == pulled, MATCH.
- **Metadata:** enable_gpu/tpu/internet, is_private, docker pinning,
  dataset_sources (incl. arc-war-kit), competition_sources, model_sources,
  machine_shape — ALL OK, no drift (source order differs, content identical).

**KERNEL PULL-BACK VERDICT: PASS. Pushed version = 3.**

## 5. Runtime canary check (deferred to build COMPLETE, ~2.2 GPU-h)

Intent §2, to read from the v3 build log:
1. `compaction v2.1: ACTIVE` banner showing **digest=OFF** and mirroring=OFF —
   a v2 banner ⇒ stale dataset served ⇒ VOID; absent/`PATCH FAILED` ⇒
   VANILLA, VOID (K2).
2. `COMPACTION=1` stamp (cell-2 banner).
3. ≥1 `COMPACTION ` event line + per-game sidecars (K1).
4. **Digest-OFF canary: `digest_tokens=0` AND `reserve_applied=0` on EVERY
   event**; any nonzero ⇒ injection leak ⇒ VOID.
5. RETAIN-OFF canary: `retained_reasoning_msgs=0` on every event.

Then the v2.1 seed-1 screen: paired M1 vs `runs/kernel_pulls/war_eval_v1/`
(mean Δlc ≥ −0.128 AND worst ≥ −1.0); M2 pure-eviction attribution; M3
recorded, not a kill criterion (intent §3). **K3: FAIL ⇒ A22 lane DEAD.**

*No submission, no queue change, $0 cloud. Kernel-push budget: 1/2 used today.*
