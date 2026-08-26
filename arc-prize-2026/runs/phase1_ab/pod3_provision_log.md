# Pod-3: null + v2 eval session — RunPod H100 PCIe provision log
Started: 2026-07-11
Pod: root@216.81.245.97 -p 14631 (H100 PCIe 80GB, /workspace network-mounted 873T, 128 cores, ~30h / ~$85 budget)

GOAL: PART A = 10 vanilla (null) seeds 101-110 (`run_ab.py --no-phase1`), PART B = 3 phase1-v2
seeds 201-203. 25 public games each, local competition-arcade simulator, vLLM Qwen3.6-27B-FP8.
Outputs: /workspace/ab_results/null/null_seed{N}.json + seed{N}/benchmark.json, /workspace/ab_results/v2/.

## Pod-1 lessons applied (all mandatory)
- L1: ninja symlinked to /usr/local/bin after venv install (engine subprocess PATH).
- L2: readiness gate = real /v1/chat/completions returning non-empty content (chat_gate.py), not /v1/models.
- L3: each seed = fresh process; master.sh reaps stale `[r]un_ab.py` / `[c]ompetition_arcade` between seeds.
- L4: pkill patterns use bracket trick; pkill lives inside master.sh (cmdline never contains pattern).
- L5: all 25 games in, no AB_SKIP_GAMES.
- L6: everything nohup'd; per-seed logs at /workspace/logs/{label}_seed{N}.log; [finished] markers -> /workspace/ab.log.

## Timeline
- 12:10 Recon: pod up (H100 PCIe 80GB, /workspace mfs mount, empty). Read pod-1 provision_log for
  exact PYSETUP vLLM serve flags; read v2 bundle run_ab.py — vanilla flag confirmed as `--no-phase1`
  (phase1_patch never imported/applied on that path; PHASE1_* env vars inert without the patch).
- 12:13 Shipped via scp: taaf_bundle.tar.gz (700KB re-tar), env_files.tar.gz (874KB, 25 games —
  required by run_ab.py ENV_FILES), phase1_v2_bundle.tar.gz (29KB), ~/.kaggle/access_token
  (no kaggle.json exists locally — same OAuth token as pod-1; -> /root/.kaggle/access_token chmod 600,
  never printed).
- 12:15 First extract failed exit-2: tar chown errors on network FS (uid 197609 from Windows tar).
  Re-extracted everything with --no-same-owner: OK. 25 env-file game dirs confirmed.
- 12:16 uv 0.11.28 installed; venv at /workspace/venv (python 3.12.3, same as pod-1). kaggle CLI 2.2.3.
  Model download nohup'd: kaggle datasets download driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot
  -p /workspace/models --unzip (27.1GB zip; started at ~130MB/s). Log: /workspace/logs/model_download.log.
- 12:18 Dep install nohup'd (uv pip --torch-backend=cu128): vllm==0.19.0 flashinfer-python==0.6.6
  ninja arc_agi==0.9.9 arcengine>=0.9.3 matplotlib==3.10.6 python-dotenv requests imageio
  imageio-ffmpeg scipy pillow numpy. (Editable installs skipped per pod-1: taaf pins
  Python==3.12.12 + private re-arc-3 git dep; run_ab.py sys.path-inserts the bundle srcs.)
- 12:20 Shipped pod scripts: /workspace/run_ab.py (v2 bundle copy + 2-line patch: OUT_DIR from
  $AB_OUT_DIR; output filename suffix = --label, so Part A writes null_seed{N}.json),
  /workspace/start_vllm.sh (exact PYSETUP flags, port 8000, model dir auto-resolved via
  find config.json), /workspace/chat_gate.py (L2), /workspace/master.sh (10 null + 3 v2, markers).
- 12:25 All scripts syntax-checked locally (ast.parse / bash -n) before shipping. Arcade recon:
  CompetitionArcadeServer runs as an in-process daemon thread, so pod-1's "lingering
  competition_arcade" = a wedged run_ab.py host process; master.sh reaps both patterns.
