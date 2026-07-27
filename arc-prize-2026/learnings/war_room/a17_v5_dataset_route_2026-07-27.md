# A17 canary v5 — DATASET-weights route (build memo, 2026-07-27)

Executes the PRIORITY-PINNED addendum (daily_iterate_prompt.md, 2026-07-27):
unblock the A17 72B-VL bench via dataset-served weights. C4 deadline: numbers
by Aug 3.

## Why the route changed (do NOT retry model_sources)

- v1/v2 (07-24/25): kernel ERROR at boot — Kaggle save-kernel API silently
  dropped `model_sources` pinned to `.../72b-instruct-awq/1` (probe-isolated,
  `runs/model_attach_probe/`, reproduced 3x across CLI versions).
- v3 (07-25): /2 pin attached and served — 72B booted, all cell-8 serve
  asserts PASSED (tool-call round-trip, MM probe) — but 0 actions in-game
  (fenced-python pathology; `runs/kernel_pulls/a17_canary_v3/analysis.md`).
- v4 (07-26): v3 + fenced-recovery adapter; ERRORed — model mount dropped
  again at push (root-caused in daily_iterate.log 07-26). The Model-mount API
  is nondeterministically lossy; route declared DEAD.
- v5 (today): weights served from a DATASET — the exact pattern the duck
  harness already uses for the 27B (`driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot`).
  Dataset mounts have never dropped in this campaign (kaggle_env_match
  discipline, 49/49 recoveries).

## The weights dataset

- No public Kaggle dataset carries Qwen2.5-VL-72B-Instruct-AWQ (searched
  qwen2.5-vl-72b / 72b-instruct-awq / qwen72b / qwen2 vl 72b: only 3B/7B/32B
  VL variants and one text-only 72B AWQ from 2025 — text-only is banned by
  the multimodal contract, scope v1 §1).
- Created: **`canivel/qwen25-vl-72b-awq`** = full HF snapshot of
  `Qwen/Qwen2.5-VL-72B-Instruct-AWQ` (24 files, 43,021,048,004 B total; 11
  safetensors shards; verified file-by-file against the HF API manifest with
  sizes). Under the ~100 GB dataset cap. Disk math: F: had 5.9 TB free —
  43 GB down + stage cleared trivially.
- `.cache/huggingface` (hf transfer state) deleted before
  `kaggle datasets create`; `dataset-metadata.json` id `canivel/qwen25-vl-72b-awq`.

## v5 composition (staged notebook = v4, three deltas)

Patch script: `duck_eval/a17/build_v5_boot_canary.py` (exact-string anchors,
match-once-or-die, idempotence-guarded). Smoke:
`duck_eval/a17/a17_v5_smoke.py` — validates compile, pins, untouched cell-8
serve graft, v4 cell-12 composition, cell-14 window deltas, metadata
env-match, and a live model-finder replay against the real downloaded
dataset layout (exactly 1 hit; 27B-decoy refused).

1. **metadata**: `model_sources` REMOVED; `canivel/qwen25-vl-72b-awq`
   appended to `dataset_sources` (all other fields byte-equal to the duckwar
   family). The cell-8 finder is already mount-path agnostic (rglob of
   /kaggle/input for arch marker + quantization_config + safetensors), so no
   notebook path change is needed — v3 proved this finder against a model
   mount; the same code finds a dataset mount.
2. **cell 2**: banner → `mode=boot-canary-v5-dataset-weights`, notes route
   death + MEASUREMENT ONLY.
3. **cell 14**: `A17_WINDOW_S` 7920 → **1500 s** and offline `soft_end` =
   `now() + A17_WINDOW_S` at bm.run entry (load time excluded). This push is
   a SHORT serve-config test of the dataset route: vLLM boot + /v1/models
   identity + forced tool-call round-trip + MM probe (all FAIL-LOUD cell-8
   boot asserts) + a ~25-min in-game slice that exercises the v4
   fenced-recovery adapter. **NOT the sealed bench**; per
   `learnings/a17_error_model.md` k=1 false-NO-GO = 1.0, output is
   measurement only — no GO/NO-GO reading.

Pre-noted short-window side effects: zero-action-abort arming range
[1800 s, window−600 s] is empty at 1500 s (never fires); stall-kill armed
only first 900 s; window-drift WARN compares vs 1500 s.

## Post-run verdict greps (pull the build log)

    A17-CANARY gpu=                                  MUST be RTX PRO 6000
    A17-CANARY model_path=                           MUST be under /kaggle/input/qwen25-vl-72b-awq  <- THE dataset-route verdict line
    A17-CANARY setup-commands rewrite OK             all 10 anchors hit
    A17-CANARY: model=Qwen2.5-VL-72B-Instruct-AWQ    72B actually served
    A17-CANARY tool-call-roundtrip=OK                risk D discharged on dataset route
    A17-CANARY mm-image-roundtrip=OK                 risk E boot probe
    fenced-recovery v1                               adapter applied (cell 12)
    A17-CANARY games=                                expect n=4 of 4
    A17-CANARY HEARTBEAT                             liveness trace
    A17-CANARY N(                                    per-game actions in the 1500 s slice
    tool_calls_recovered_from_markup                 fenced-recovery hits (H) evidence

PASS = boot asserts all green on the dataset mount + kernel COMPLETE.
Any `A17-CANARY FATAL` = the route needs the retry slot (slot 2 today).

## Contingent next push (tomorrow, slot 1)

On v5 PASS: build v6 = v5 with cell-14 window restored to the sealed bench
config (`A17_WINDOW_S = 7920.0`, original budget-derived soft_end block —
`build_v5_boot_canary.py` documents both original strings) and banner
`mode=throughput-canary-v6-dataset-weights`. That is the FULL-window canary
re-run under the 07-26 v4 pre-registration
(`learnings/war_room/a17_v4_prereg_2026-07-26.md`: G1 recovery ≥ 0.95, G2
≥ 100 executed actions, G3 cadence measurement, G4 = NO capability
interpretation), which also delivers the ρ_action denominator (scope v2 §3:
480 / Σ N₇₂B). Sequence after that stays sealed: freeze null_adj at measured
ρ_action → seed-1 scored bench. ~2.5 GPU-h per full-window push; remaining
screen budget ≈ 7.5 GPU-h. Gate arithmetic stays sealed
(`runs/sealed/r17_thresholds.json`); numbers go to the sealed walk +
Sunday panel, never interpreted at k=1.
