# A22 compaction-eval push report — 2026-08-02

**Executed:** 2026-08-02 (R23-ratified staged pushes; item 2 eval plan ratified,
NO scored draws; item 3 compaction lane NOT serialized behind the A/B).
**Authorities:** `learnings/war_room/a22_compaction_prereg_2026-08-01.md`;
`ITERATION_LOG.md` L508-513 (STAGED 08-01, slot-1 08-02 AFTER R23 look).
**Budget:** kernel push 1 of today's max-2 (exactly ONE performed). Dataset
version does NOT count vs kernel budget. NO competition submission, NO queue
change, $0 cloud.

---

## Pre-push gate verification (all GREEN — no abort)

| gate | expected | observed | verdict |
|---|---|---|---|
| compaction_patch.py sha256 | `6af5fce1…17d2` | `6af5fce1b4b678e72e523ca132b09a3821fd23bf9ccb20995487ac3c4a1217d2` | MATCH |
| notebook sha256 | `1c4e51eb…50edb` | `1c4e51ebad3e8a371d3b8aad525930a60590c357b6f183f51be7b5e32e750edb` | MATCH |
| dataset slug (from dataset-metadata.json) | canivel/arc-war-kit | canivel/arc-war-kit | OK |
| kernel enable_gpu | true | true | OK |
| kernel dataset_sources contains arc-war-kit | yes | yes (+ 3 wheelhouse/taaf/qwen datasets) | OK |
| metadata vs frozen family (duckgate-eval-s1) | byte-match modulo id/title/code_file + arc-war-kit add | docker_image, machine_shape (RtxPro6000), competition_sources, enable_gpu/tpu/internet, model_sources(empty) all identical | NO DRIFT |
| compaction_smoke.py | 41/41 PASS | 41/41 PASS (re-run 08-02) | OK |
| preflight.py --mode structural --host-gates | BLOCK expected (K1 unpushed slug) | BLOCK: K1 "kaggle pull failed" (slug had never been pushed) — expected per a17 eval-family precedent (gate = build-script proof + smoke + pull-back, not structural T4 on an unpushed slug) | EXPECTED |

Full compaction_patch.py sha (was truncated in the 08-01 log/prereg as
`6af5fce1…17d2`; recorded here in full for the ledger):
`6af5fce1b4b678e72e523ca132b09a3821fd23bf9ccb20995487ac3c4a1217d2`.

---

## 1. Dataset version push — `canivel/arc-war-kit`

- Pushed folder `duck_eval/warpack/_kaggle_dataset/` **as-is** (no tarball;
  Kaggle auto-extracts). Removed stray `__pycache__/` before push so no stale
  bytecode shipped. New file: `compaction_patch.py` (24238 B).
- Version notes: `A22 compaction_patch v1 (sha 6af5fce1...)`.
- Note: kaggle 2.0.0 CLI has a Windows relative-path temp-file bug; push
  succeeded when invoked with an **absolute** `-p` path via PowerShell.
- Result: "Dataset version is being created" — upload successful for all 9 files.

## 2. Dataset byte-audit (feedback_kaggle_dataset_code_sync)

Downloaded the new version back (`kaggle datasets download canivel/arc-war-kit
--unzip`) and byte-compared:

- downloaded compaction_patch.py sha256 = `6af5fce1…17d2` (full match)
- `cmp` staged vs downloaded → **IDENTICAL (exit 0)**

**DATASET BYTE-AUDIT VERDICT: PASS** — served bytes == staged bytes.

## 3. Kernel push — `canivel/arc3-duck-compaction-eval`

- `kaggle kernels push -p notebooks/duckcompaction-eval` →
  **"Kernel version 1 successfully pushed."**
- Post-push status: QUEUED (build starting).
- This is **kernel push 1 of 2** for 2026-08-02. Budget respected: exactly one
  kernel push performed; no further kernel pushes today.

## 4. Kernel pull-back verify

Pulled back (`kaggle kernels pull -m`) and compared:

- **Notebook:** raw-byte `cmp` DIFFERS (34662 B pulled vs 39404 B local) — this
  is Kaggle's standard `.ipynb` round-trip re-serialization (JSON reformat /
  empty-field strip), same cosmetic diff seen on every prior eval-family pull,
  NOT a content change. Semantic verify: **all 17 code-cell sources
  byte-identical** (concat-src sha256 `b7354aed…3cb5` matches local exactly).
  Markers present in pulled: `COMPACTION=1`, `compaction_patch`,
  `WARPACK_FORCE_OFFLINE_BENCH` (offline bench), seed banner.
- **Metadata round-trip:** all keys OK, NO drift — enable_gpu true, enable_tpu
  /enable_internet false, is_private true, docker_image byte-match,
  machine_shape NvidiaRtxPro6000, competition_sources match, model_sources
  empty, dataset_sources set-match (canivel/arc-war-kit +
  arc3-vllm-h100-wheelhouse-v3 + taaf-kaggle-source-share +
  vrfai-qwen3-6-27b-fp8-hf-snapshot).

**KERNEL PULL-BACK VERDICT: PASS** — code-cell content byte-identical, metadata
byte-matched, arc-war-kit correctly attached. **Pushed version = 1.**

## 5. Runtime banner check (NOT waited on — read later today/tomorrow)

On build **COMPLETE**, the kernel log MUST show (prereg §3 canary; K1/K2 kill):
1. `compaction v1: ACTIVE …` banner (dataset-version proof) — absent or
   `PATCH FAILED` ⇒ run is VANILLA, VOID for this arm.
2. `COMPACTION=1` engaged.
3. **≥1 `COMPACTION ` event line** (+ per-game `*_compaction_events.jsonl`
   sidecars). K1: 0 events ⇒ mechanism never engaged ⇒ VOID + no further pushes
   until root-caused.
4. (RETAIN sub-canary) `retained_reasoning_msgs>0` in event lines.

This read happens later today or tomorrow after build COMPLETE; NOT part of
this push session.

---

## Summary verdicts

- **Dataset:** `canivel/arc-war-kit` new version pushed; byte-audit **PASS**
  (compaction_patch.py sha `6af5fce1…17d2`, cmp identical).
- **Kernel:** `canivel/arc3-duck-compaction-eval` **version 1** pushed;
  pull-back **PASS** (17/17 cell sources identical, metadata no-drift).
- **Expected build duration:** ~2.2 GPU-h (per gate_eval_readiness_2026-07-30
  §4 eval-family estimate; ~2h12m wall observed on the gate-eval seed-1 build
  08-01). Against the ~30 GPU-h/wk free quota.

*No submission, no queue change, $0 cloud. Kernel-push budget: 1/2 used today.*
