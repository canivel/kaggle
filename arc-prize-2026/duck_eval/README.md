# duck_eval — score-parity rig for GPU/LLM kernels (ZERO-COST, Kaggle-quota-first)

**Purpose:** predict a kernel's Kaggle score BEFORE spending a submission slot.

**Primary rig (free): Kaggle build-time evals.** A `kaggle kernels push` costs
zero submissions and runs the notebook on the ACTUAL competition SKU
(RTX PRO 6000 Blackwell, native FP8) inside the 30 GPU-h/week quota. The duck
notebook's non-submission path already runs a diagnostics benchmark (verified:
our duck-repro build printed `mean score: 0.11`). Procedure:
1. Fork the duck notebook; in the customization-hook cell (cell 6, "tweak bm")
   set the seed + game subset + full-benchmark mode instead of quick diagnostics.
2. `kaggle kernels push` → build runs the seeded sweep on the real SKU.
3. `kaggle kernels output <slug>` → pull the results JSON.
   **If the kernel ERRORed, do NOT start with `kernels output`.** That command downloads
   `/kaggle/working` FIRST, and every vLLM kernel's working dir holds the multi-GB
   `vllm-site-packages` tree, so the log effectively never arrives. Use instead:

       kaggle kernels logs <slug> > post_mortem.log     # CLI 2.2.3 ONLY

   `kernels logs` streams the full stdout as JSON (`stream_name`/`time`/`data`) and never
   touches the working dir: 238 KB in seconds on the 2026-08-15 Q38 kernel, including
   per-line timestamps that give you the wallclock of every boot milestone. **Two CLIs are
   installed and only one has it:** pushes use 2.0.1 at
   `~/AppData/Roaming/Python/Python313/Scripts/kaggle.exe` (no `logs` subcommand); post-mortems
   use **2.2.3** at `/f/kaggle/march-madness-2026/.venv/Scripts/kaggle`.
   `kernels files <slug>` returns EMPTY for *every* errored kernel (verified against
   `arc3-b122-boot-canary` and `arc3-lora-serve-canary`) - it is not evidence of anything.

   **This corrects a standing note in `learnings/war_room/lora_lane_2026-08-13.md`**, which
   recorded that "the log never arrived usefully on either CLI" and diagnosed that lane's
   death blind by bracketing artifact presence/absence. The bracketing was good work on a
   false premise; the log was retrievable the whole time.
4. Save to runs/duck_eval/<tag>_seed<N>.json; attach as `evidence` in the queue.

Budgeting: 30 GPU-h/week caps this at ~2-5 seeded sweeps/week depending on
sweep depth (12h session max). The panel's 8-10-seed null accumulates over
2-3 weeks or uses per-seed game subsets — slower than renting, but $0 and
ZERO quantization gap (it IS the scoring hardware; amendments A1/A2's anchor
battery becomes unnecessary for builds run this way).

**Optional accelerator (declined for now, user decision 2026-07-07): RunPod
A40 rig below. Keep the scripts; if the weekly quota becomes the bottleneck
the option remains.**

Per the panel-approved plan (learnings/winning_solution_FINAL.md):
- Phase 0b: 8-10-seed frozen-baseline null on dev-18 → the identical-build
  null every later merge gate tests against.
- Tokens/s parity vs the Kaggle RTX PRO 6000 measured here; all budgets are
  token-denominated, never wall-clock.
- Quantization caveat: A40 (Ampere) has no FP8 — pilot runs FP8-dequant→BF16
  or AWQ-int8. 3-game anchor battery on the actual Kaggle SKU bounds the gap
  (amendments A1/A2: game-level t at df=2, TOST equivalence; default to
  kernel-anchored entry if unsatisfiable).

## Layout
- `taaf_bundle/` — snapshot of jeroencottaar/taaf-kaggle-source-share
  (ARC3-Inference + tufa-arc-agi-framework + pickled Benchmark)
- `provision_a40.sh` — RunPod pod setup (run on the pod)
- `run_local_eval.sh` — one seeded eval sweep → results JSON
- Results land in `runs/duck_eval/<tag>_seed<N>.json`

## Evidence gate
`scripts/daily_submit.py` refuses to submit any queue entry carrying
`"requires_evidence": true` unless `"evidence"` points to a results JSON with
>=3 seeds. Frozen known-good builds (baseline redraws, byte-identical forks
of proven kernels) don't need evidence; anything experimental does.

## Workflow
1. Provision: `runpodctl create pod --gpuType "NVIDIA A40" --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 --containerDiskSize 100 --volumeSize 100`
2. Sync this dir + kaggle-data/environment_files to the pod
3. On pod: `bash provision_a40.sh` (installs taaf + ARC3-Inference, downloads
   Qwen3 27B snapshot from HF, starts vLLM)
4. `bash run_local_eval.sh <tag> <seed>` per seed
5. Pull results JSONs back to runs/duck_eval/; attach path as `evidence` in
   the submission queue entry.
